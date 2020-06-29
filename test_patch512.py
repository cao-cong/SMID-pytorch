from __future__ import division
import os, scipy.io
import re
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import glob
import cv2
from PIL import Image
from skimage.measure import compare_psnr,compare_ssim
import math
import time
import rawpy
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

save_dir = './model'

latest_epoch = findLastCheckpoint(save_dir=save_dir)
print('latest denoiser train epoch: {}'.format(latest_epoch))
denoiser = torch.load(os.path.join(save_dir, 'model_epoch{}.pth'.format(latest_epoch)))
denoiser = denoiser.cuda()

test_result_dir = 'test_result_static_video'
if not os.path.isdir(test_result_dir):
    os.makedirs(test_result_dir)

f = open('test_list.txt')
scene_ids = []
for line in f.readlines():           
    line = line.strip()
    number = int(line)              
    scene_ids.append(number)  
print(scene_ids)

f = open('frame5_test_psnr_and_ssim_on_DRV_dataset_patch512.txt', 'w')

avg_srgb_psnr = 0
avg_srgb_ssim = 0

for scene_id in scene_ids:

    test_gt_path = glob.glob('../../other_dataset/DRV_dataset/long/%04d/half*.png'%(scene_id))[0]
    test_gt = cv2.imread(test_gt_path).astype(np.float32)/255.0 

    test_raw_path = '../../other_dataset/DRV_dataset/VBM4D_rawRGB/%04d/0005.png'%(scene_id)
    raw = cv2.imread(test_raw_path, cv2.IMREAD_UNCHANGED).astype(np.float32)/65535.0 
    input_full = np.expand_dims(raw, axis=0)
    
    test_result = test_big_size_image(input_full, denoiser, patch_h = 512, patch_w = 512, patch_h_overlap = 64, patch_w_overlap = 64)
    cv2.imwrite(test_result_dir+'/scene_%04d_denoised.png'%scene_id, np.uint8(test_result[0]*255))

    test_srgb_psnr = compare_psnr(test_gt, np.uint8(test_result[0]*255).astype(np.float32)/255, data_range=1.0)
    test_srgb_ssim = compare_ssim(test_gt, np.uint8(test_result[0]*255).astype(np.float32)/255, data_range=1.0, multichannel=True)

    print('test srgb psnr : {}, test srgb ssim : {} '.format(test_srgb_psnr,test_srgb_ssim))
    context = 'scene {}  srgb psnr/ssim: {}  {}'.format(scene_id,test_srgb_psnr,test_srgb_ssim) + '\n'
    f.write(context)
        
    avg_srgb_psnr += test_srgb_psnr
    avg_srgb_ssim += test_srgb_ssim

avg_srgb_psnr = avg_srgb_psnr/49
avg_srgb_ssim = avg_srgb_ssim/49

print('average test srgb psnr : {}, test srgb ssim : {} '.format(avg_srgb_psnr,avg_srgb_ssim))
context = 'average srgb psnr/ssim: {}  {}'.format(avg_srgb_psnr,avg_srgb_ssim) + '\n'
f.write(context)




