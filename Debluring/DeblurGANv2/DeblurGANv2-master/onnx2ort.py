import  onnxruntime as ort
import onnx
import numpy as np
import pandas as pd
import cv2
import os
from glob import glob
from typing import Optional
import albumentations as albu

def get_normalize():
    normalize = albu.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    normalize = albu.Compose([normalize], additional_targets={'target': 'image'})

    def process(a, b):
        r = normalize(image=a, target=b)
        return r['image'], r['target']

    return process

def main(img_pattern: str,
        mask_pattern: Optional[str] = None,
        out_dir='submit_ort'
        ):
    def sorted_glob(pattern):
        return sorted(glob(pattern))
    imgs = sorted_glob(img_pattern)
    masks = sorted_glob(mask_pattern) if mask_pattern is not None else [None for _ in imgs]
    names = sorted([os.path.basename(x) for x in glob(img_pattern)])
    pairs = zip(imgs, masks)

    os.makedirs(out_dir, exist_ok = True)
    for name, pair in zip(names,pairs):
        f_img, f_mask = pair
        img, mask = map(cv2.imread, (f_img, f_mask))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

def sorted_glob(pattern):
    return sorted(glob(pattern))


        

def get_files(img_dir:str):
    list=[]
    for filepath,dirnames,filenames in os.walk(img_dir):
        for filename in filenames:
            list.append(os.path.join(filepath,filename))
    return list

if __name__ == '__main__':

    img_dir='./test_img'
    out_dir = 'submit_ort'
    model_path = 'FPNInception_736_1312.onnx'
    onnx_model = onnx.load(model_path)
    mask_pattern = None
    img_path = get_files(img_dir)
    onnx.checker.check_model(onnx_model)
    for img_pattern in img_path:
        imgs = sorted_glob(img_pattern)
        masks = sorted_glob(mask_pattern) if mask_pattern is not None else [None for _ in imgs]
        names = sorted([os.path.basename(x) for x in glob(img_pattern)])
        pairs = zip(imgs, masks)

        normalize_fn = get_normalize()

        os.makedirs(out_dir, exist_ok = True)
        for name, pair in zip(names,pairs):
            f_img, f_mask = pair
            img_o = cv2.imread(f_img)
            img = cv2.cvtColor(img_o,cv2.COLOR_BGR2RGB)
            img,_ = normalize_fn(img,img)
            # mask = np.ones_like(img,dtype=np.float32)
            h, w, _ = img.shape
            block_size = 32
            min_h = (h // block_size + 1) * block_size
            min_w = (w // block_size + 1) * block_size
            pad_params = {
                'mode' : 'constant',
                'constant_values' : 0,
                'pad_width' : ((0,min_h - h), (0,min_w - w),(0, 0))
            }
            img = np.pad(img, **pad_params)
            
            # mask = np.pad(mask, **pad_params)

            # (H,W,C) to (C,H,W)
            img_t = img.transpose((2,0,1))
            # add batch size dimension
            img_n = np.expand_dims(img_t,axis=0)
            ort_cess = ort.InferenceSession(model_path)
            outputs = ort_cess.run(None,{'input.1': img_n})
            output,  = outputs
            d_imgnp = output[0]
            d_imgnp = (np.transpose(d_imgnp,(1,2,0)) + 1) / 2.0 * 255.0
            d_imgnp = d_imgnp.astype(np.uint8)
            d_img = cv2.cvtColor(d_imgnp,cv2.COLOR_RGB2BGR)[:h,:w,:]
            cv2.imwrite(os.path.join(out_dir,name),d_img)
            

            