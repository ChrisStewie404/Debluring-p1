import torch
from network import GKMNet
from data import TestDataset
from torch.utils.data import DataLoader
import numpy as np
import os
import cv2
import train_config as config


def get_files(img_dir:str):
    list=[]
    for filepath,dirnames,filenames in os.walk(img_dir):
        for filename in filenames:
            list.append(os.path.join(filepath,filename))
    return list

if __name__ == '__main__':
    if torch.cuda.is_available():
        net = GKMNet().cuda()
    else:
        net = GKMNet().cpu()

    img_dir='./test_img'
    img_path = get_files(img_dir)
    for i in img_path:
        img = cv2.imread(i)
        h,w,_ = img.shape
        img_2 = cv2.resize(img,(w*2,h*2))
        img_4 = cv2.resize(img,(w*4,h*4))
        img = np.transpose(img,(2,0,1))
        img_2 = np.transpose(img_2,(2,0,1))
        img_4 = np.transpose(img_4,(2,0,1))
        img = np.expand_dims(img,axis=0)
        img_2 = np.expand_dims(img_2,axis=0)
        img_4 = np.expand_dims(img_4,axis=0)
        t_img = torch.from_numpy(img)
        t_img_2 = torch.from_numpy(img_2)
        t_img_4 = torch.from_numpy(img_4)
        with torch.no_grad():
            if torch.cuda.is_available():
                inputs = [t_img_4.cuda().float(),t_img_2.cuda().float(),t_img.cuda().float()]
            else:
                inputs = [t_img_4.cpu().float(),t_img_2.cpu().float(),t_img.cpu().float()]
            pred = net(*inputs)

