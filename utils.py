from __future__ import division
import os, scipy.io
import re
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import glob
import cv2
from scipy.stats import poisson
from skimage.measure import compare_psnr,compare_ssim
import time

def reduce_mean(out_im, gt_im):
    return torch.abs(out_im - gt_im).mean()

def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'model_epoch*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_epoch(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch

def preprocess(raw):
    input_full = raw.transpose((0, 3, 1, 2))
    input_full = torch.from_numpy(input_full)
    input_full = input_full.cuda()
    return input_full

def postprocess(output):
    output = output.cpu()
    output = output.detach().numpy().astype(np.float32)
    output = np.transpose(output, (0, 2, 3, 1))
    output = np.clip(output,0,1)
    return output

def F_loss(img1, img2, vgg_model):
    output_1_vgg_0, output_1_vgg_1, output_1_vgg_2, output_1_vgg_3, output_1_vgg_4 = vgg_model(img1)
    output_2_vgg_0, output_2_vgg_1, output_2_vgg_2, output_2_vgg_3, output_2_vgg_4 = vgg_model(img2)
    loss_0 = reduce_mean(output_1_vgg_0, output_2_vgg_0)
    loss_1 = reduce_mean(output_1_vgg_1, output_2_vgg_1)
    loss_2 = reduce_mean(output_1_vgg_2, output_2_vgg_2)
    loss_3 = reduce_mean(output_1_vgg_3, output_2_vgg_3)
    loss_4 = reduce_mean(output_1_vgg_4, output_2_vgg_4)

    return loss_0 + loss_1 + loss_2 + loss_3 + loss_4

def test_big_size_image(input_data, denoiser, patch_h = 256, patch_w = 256, patch_h_overlap = 64, patch_w_overlap = 64):

    H = input_data.shape[1]
    W = input_data.shape[2]
    
    test_result = np.zeros((input_data.shape[0],H,W,3))
    t0 = time.clock()
    h_index = 1
    while (patch_h*h_index-patch_h_overlap*(h_index-1)) < H:
        test_horizontal_result = np.zeros((input_data.shape[0],patch_h,W,3))
        h_begin = patch_h*(h_index-1)-patch_h_overlap*(h_index-1)
        h_end = patch_h*h_index-patch_h_overlap*(h_index-1) 
        w_index = 1
        while (patch_w*w_index-patch_w_overlap*(w_index-1)) < W:
            w_begin = patch_w*(w_index-1)-patch_w_overlap*(w_index-1)
            w_end = patch_w*w_index-patch_w_overlap*(w_index-1)
            test_patch = input_data[:,h_begin:h_end,w_begin:w_end,:]               
            test_patch = preprocess(test_patch)               
            with torch.no_grad():
                output_patch = denoiser(test_patch)
            test_patch_result = postprocess(output_patch)
            if w_index == 1:
                test_horizontal_result[:,:,w_begin:w_end,:] = test_patch_result
            else:
                for i in range(patch_w_overlap):
                    test_horizontal_result[:,:,w_begin+i,:] = test_horizontal_result[:,:,w_begin+i,:]*(patch_w_overlap-1-i)/(patch_w_overlap-1)+test_patch_result[:,:,i,:]*i/(patch_w_overlap-1)
                test_horizontal_result[:,:,w_begin+patch_w_overlap:w_end,:] = test_patch_result[:,:,patch_w_overlap:,:]
            w_index += 1                   

        test_patch = input_data[:,h_begin:h_end,-patch_w:,:]         
        test_patch = preprocess(test_patch)
        with torch.no_grad():
            output_patch = denoiser(test_patch)
        test_patch_result = postprocess(output_patch)       
        last_range = w_end-(W-patch_w)       
        for i in range(last_range):
            test_horizontal_result[:,:,W-patch_w+i,:] = test_horizontal_result[:,:,W-patch_w+i,:]*(last_range-1-i)/(last_range-1)+test_patch_result[:,:,i,:]*i/(last_range-1)
        test_horizontal_result[:,:,w_end:,:] = test_patch_result[:,:,last_range:,:]       

        if h_index == 1:
            test_result[:,h_begin:h_end,:,:] = test_horizontal_result
        else:
            for i in range(patch_h_overlap):
                test_result[:,h_begin+i,:,:] = test_result[:,h_begin+i,:,:]*(patch_h_overlap-1-i)/(patch_h_overlap-1)+test_horizontal_result[:,i,:,:]*i/(patch_h_overlap-1)
            test_result[:,h_begin+patch_h_overlap:h_end,:,:] = test_horizontal_result[:,patch_h_overlap:,:,:] 
        h_index += 1

    test_horizontal_result = np.zeros((input_data.shape[0],patch_h,W,3))
    w_index = 1
    while (patch_w*w_index-patch_w_overlap*(w_index-1)) < W:
        w_begin = patch_w*(w_index-1)-patch_w_overlap*(w_index-1)
        w_end = patch_w*w_index-patch_w_overlap*(w_index-1)
        test_patch = input_data[:,-patch_h:,w_begin:w_end,:]               
        test_patch = preprocess(test_patch)               
        with torch.no_grad():
            output_patch = denoiser(test_patch)
        test_patch_result = postprocess(output_patch)
        if w_index == 1:
            test_horizontal_result[:,:,w_begin:w_end,:] = test_patch_result
        else:
            for i in range(patch_w_overlap):
                test_horizontal_result[:,:,w_begin+i,:] = test_horizontal_result[:,:,w_begin+i,:]*(patch_w_overlap-1-i)/(patch_w_overlap-1)+test_patch_result[:,:,i,:]*i/(patch_w_overlap-1)
            test_horizontal_result[:,:,w_begin+patch_w_overlap:w_end,:] = test_patch_result[:,:,patch_w_overlap:,:]   
        w_index += 1

    test_patch = input_data[:,-patch_h:,-patch_w:,:]         
    test_patch = preprocess(test_patch)
    with torch.no_grad():
        output_patch = denoiser(test_patch)
    test_patch_result = postprocess(output_patch)
    last_range = w_end-(W-patch_w)       
    for i in range(last_range):
        test_horizontal_result[:,:,W-patch_w+i,:] = test_horizontal_result[:,:,W-patch_w+i,:]*(last_range-1-i)/(last_range-1)+test_patch_result[:,:,i,:]*i/(last_range-1) 
    test_horizontal_result[:,:,w_end:,:] = test_patch_result[:,:,last_range:,:] 

    last_last_range = h_end-(H-patch_h)
    for i in range(last_last_range):
        test_result[:,H-patch_w+i,:,:] = test_result[:,H-patch_w+i,:,:]*(last_last_range-1-i)/(last_last_range-1)+test_horizontal_result[:,i,:,:]*i/(last_last_range-1)
    test_result[:,h_end:,:,:] = test_horizontal_result[:,last_last_range:,:,:]
   
    t1 = time.clock()
    print('Total running time: %s s' % (str(t1 - t0)))

    return test_result