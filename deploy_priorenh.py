import torch
import os.path as osp
import os
import cv2
from model.network import VQFPEnhancer_PCNN
from utils.misc import load_model
import numpy as np
from tqdm import tqdm
import yaml
from easydict import EasyDict as edict

def image_read(img_path, size=512):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    h_org, w_org = img.shape
    if h_org > w_org:
        img = cv2.copyMakeBorder(img, 0, 0, 0, h_org - w_org, cv2.BORDER_CONSTANT, value=255)
        padding_direction = 0
        padding_size = h_org - w_org
    else:
        img = cv2.copyMakeBorder(img, 0, w_org - h_org, 0, 0, cv2.BORDER_CONSTANT, value=255)
        padding_direction = 1
        padding_size = w_org - h_org
    
    img = cv2.resize(img, (size, size)) # try to directly resize the image to 256 x 256
    img = (img.astype(np.float32) - 127.5) / 127.5
    return img[None], (w_org, h_org), padding_direction, padding_size

def inverse_image(image, org_size, padding_direction, padding_size):
    max_size = max(org_size)
    image = cv2.resize(image, (max_size, max_size))
    if padding_direction == 0:
        image = image[:, :max_size - padding_size]
    else:
        image = image[:max_size - padding_size, :]
    return image

def deploy_enh(folder, ckpt_path, w=0, method_name='PriorEnh', pre_enh=False):
    # standard the folder path 
    folder = os.path.normpath(folder)
    model_path = osp.join(ckpt_path, 'priorenh.pth') # 
    cfg_path = osp.join(ckpt_path, 'vq.yaml')
    config = edict(yaml.safe_load(open(cfg_path, 'r')))
    model = VQFPEnhancer_PCNN(config.hdconfig, config.ldconfig, n_embed=config.n_codebook, 
                              embed_dim=config.embed_dim, pcn_embed=config.pcn_embed, ckpt_path=config.ckpt_path, pre_enh=pre_enh)
    model.cuda()
    load_model(model, model_path)
    model.eval()
    # Load the dataset path
    img_lst = os.listdir(folder)
    save_path = osp.join('output', method_name)
    os.makedirs(save_path, exist_ok=True)

    for img_name in tqdm(img_lst):
        img_path = osp.join(folder, img_name)
        img, org_shape, padding_dr, padding_size = image_read(img_path)
        img = torch.from_numpy(img).unsqueeze(0).cuda()
        with torch.no_grad():
            try:
                enh = model.module.enhance(img, w=w)
            except:
                enh = model.enhance(img, w=w)
        enh = torch.clamp(enh, -1, 1)   
        enh = enh.squeeze().cpu().numpy()
        enh = ((enh + 1) * 127.5).astype(np.uint8)
        enh = inverse_image(enh, org_shape, padding_dr, padding_size)
        cv2.imwrite(osp.join(save_path, img_name), enh)

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Deploy the Frequency Enhancement model to the fingerprints")
    parser.add_argument('--gpu', '-g', type=str, default='0', help='The GPU id')
    parser.add_argument('--folder', '-f', type=str, required=True, help='The dataset folder')
    parser.add_argument('--w', '-w', type=float, default=0.5, help='fusing')
    parser.add_argument('--pre_enh', '-e', action='store_true', help='Whether to pre-enhance the images')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    model_path = 'pretrained_model/priorenh' 
    deploy_enh(args.folder, model_path, args.w, method_name='priorenh', pre_enh=args.pre_enh)