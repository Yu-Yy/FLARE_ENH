# -*- encoding: utf-8 -*-
'''
@File    :   deploy_enh.py
@Time    :   2024/10/09 16:20:23
@Author  :   panzhiyu 
@Version :   1.0
@Contact :   pzy20@mails.tsinghua.edu.cn
@License :   Copyright (c) 2024, Zhiyu Pan, Tsinghua University. All rights reserved
@Function :  deploy the model to a specific dataset
'''
import torch
import os.path as osp
import os
import cv2
from model.network import SqueezeUNet
from utils.misc import load_model
import numpy as np
from tqdm import tqdm

def image_read(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # process the img size for 4 
    h_org, w_org = img.shape
    # find the nearest 4 times of the img size, for the larger direction
    h = int(np.ceil(h_org / 16) * 16)
    w = int(np.ceil(w_org / 16) * 16)
    img = cv2.resize(img, (w, h)) 
    img = (img.astype(np.float32) - 127.5) / 127.5
    return img[None], (w_org, h_org)


def deploy_enh(folder, model_path, method_name, pre_enh=False):
    # load the model
    folder = os.path.normpath(folder)
    model = SqueezeUNet(input_channels=1, num_classes=2, pre_enh=pre_enh)
    model.cuda()
    load_model(model, model_path)
    model.eval()
    # Load the dataset path
    img_lst = os.listdir(folder)
    save_path = osp.join('output', method_name)
    os.makedirs(save_path, exist_ok=True)
    
    for img_name in tqdm(img_lst):
        img_path = osp.join(folder, img_name)
        # img = image_read(img_path)
        img, org_shape = image_read(img_path)
        img = torch.from_numpy(img).unsqueeze(0).cuda()
        with torch.no_grad():
            pred = model(img)
        enh, _ = torch.split(pred, [1, 1], dim=1)
        # enh = pred['enh']
        enh = enh.squeeze().cpu().numpy()
        enh = (enh * 255).astype(np.uint8)
        enh = cv2.resize(enh, org_shape) 
        cv2.imwrite(osp.join(save_path, img_name), enh)
    
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Deploy the Frequency Enhancement model to the fingerprints")
    parser.add_argument('--gpu', '-g', type=str, default='0', help='The GPU id')
    parser.add_argument('--folder', '-f', type=str, required=True, help='The dataset folder')
    parser.add_argument('--pre_enh', '-e', action='store_true', help='Whether to pre-enhance the images')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    ckpt_path = 'pretrained_model/unetenh/unetenh.pth'
    method_name = 'UNetEnh'
    deploy_enh(args.folder, ckpt_path, method_name, pre_enh=args.pre_enh)