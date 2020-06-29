from __future__ import division
import os, time, scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import glob
import re
import cv2
from PIL import Image
from skimage.measure import compare_psnr,compare_ssim
from scipy.stats import poisson
from models import SeeMotionInDarkNet
from tensorboardX import SummaryWriter
import random
import rawpy
from vgg import VGG19_Extractor
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if not os.path.isdir('model'):
    os.makedirs('model')
if not os.path.isdir('logs'):
    os.makedirs('logs')
if not os.path.isdir('visualize'):
    os.makedirs('visualize')

f = open('train_list.txt')
scene_ids = []
for line in f.readlines():           
    line = line.strip()
    #if line[1]!=0:
    number = int(line)              
    scene_ids.append(number)  
print(scene_ids)

ps = 512 # patch size for training
batch_size = 1

writer = SummaryWriter('logs')

learning_rate = 1e-4

vgg_model = VGG19_Extractor(output_layer_list=[3,8,13,22]).cuda()

model = SeeMotionInDarkNet().cuda()

initial_epoch = findLastCheckpoint(save_dir='model')
if initial_epoch > 0:
    print('resuming by loading epoch %d' % initial_epoch)
    model = torch.load('model/model_epoch%d.pth' % initial_epoch)
    initial_epoch += 1

opt = optim.Adam(model.parameters(), lr = learning_rate)

# Raw data takes long time to load. Keep them in memory after loaded.
gt_srgbs = [None] * len(scene_ids)

if initial_epoch==0:
    step=0
else:
    step = (initial_epoch-1)*int(len(scene_ids)/batch_size)
for epoch in range(initial_epoch, 1001):
    cnt = 0
    if epoch >500:
        for g in opt.param_groups:
            g['lr'] = 1e-5

    for batch_id in range(int(len(scene_ids)/batch_size)):
        gt_raw_batch_list = []
        input_raw_batch_list = []
        batch_num = 0
        while batch_num<batch_size:
            batch_num += 1

            scene_id = random.sample(scene_ids,1)[0]
            index_id = scene_ids.index(scene_id)

            gt_path = glob.glob('../dataset/DRV_dataset/long/%04d/half*.png'%(scene_id))[0]
            if gt_srgbs[index_id] is None:
                gt_srgb = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
                gt_srgb = np.expand_dims(gt_srgb.astype(np.float32)/65535.0, axis=0)
                gt_srgbs[index_id] = gt_srgb
            gt_srgb_full = gt_srgbs[index_id]

            in_files = sorted(glob.glob('../dataset/DRV_dataset/VBM4D_rawRGB/%04d/*.png'%scene_id))
            #choose two random frames from the same video
            ind_seq = np.random.random_integers(0,len(in_files)-2)
            in_path = in_files[ind_seq]
            im = cv2.imread(in_path,cv2.IMREAD_UNCHANGED)
            in_im1 = np.expand_dims(np.float32(im/65535.0),axis = 0)
            ind_seq2 = np.random.random_integers(0,len(in_files)-2)
            if ind_seq2 == ind_seq:
                ind_seq2 += 1
            in_path = in_files[ind_seq2]
            im = cv2.imread(in_path,cv2.IMREAD_UNCHANGED)
            in_im2 = np.expand_dims(np.float32(im/65535.0),axis = 0)
            in_im = np.concatenate([in_im1,in_im2],axis=3)
       
            H = 918
            W = 1374

            xx = np.random.randint(0, W - ps+1)
            yy = np.random.randint(0, H - ps+1)

            gt_patch = gt_srgb_full[:,yy:yy + ps, xx:xx + ps,:]
            input_patch = in_im[:,yy:yy + ps, xx:xx + ps,:]
               
            gt_raw_batch_list.append(gt_patch)
            input_raw_batch_list.append(input_patch)

        gt_raw_batch = np.concatenate(gt_raw_batch_list, axis=0)
        input_raw_batch = np.concatenate(input_raw_batch_list, axis=0)

        gt_data = torch.from_numpy(gt_raw_batch.copy()).permute(0,3,1,2).cuda()
        in_data = torch.from_numpy(input_raw_batch.copy()).permute(0,3,1,2).cuda()
        
        model.zero_grad()

        denoised_out1 = model(in_data[:,:3,:,:])
        denoised_out2 = model(in_data[:,3:,:,:])

        l1_loss = F_loss(denoised_out1, gt_data, vgg_model) + F_loss(denoised_out2, gt_data, vgg_model)
        self_consistency_loss = F_loss(denoised_out1, denoised_out2, vgg_model) 

        loss = l1_loss + 0.05*self_consistency_loss
        loss.backward()

        opt.step()

        cnt += 1
        step += 1
        writer.add_scalar('loss', loss.item(), step)
        writer.add_scalar('l1_loss', l1_loss.item(), step)
        writer.add_scalar('self_consistency_loss', self_consistency_loss.item(), step)

        print("epoch:%d iter%d loss=%.3f" % (epoch, cnt, loss.data))

    if epoch%100==0:
        torch.save(model, os.path.join('model/model_epoch%d.pth' % epoch))
