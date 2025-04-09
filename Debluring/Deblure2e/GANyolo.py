import os
from glob import glob
from typing import Optional
from tqdm import tqdm
import numpy as np

from predict import Predictor
from fpn_inception import FPNInception
import cv2
from ultralytics import YOLO


def get_files(img_dir:str):
    list=[]
    for filepath,dirnames,filenames in os.walk(img_dir):
        for filename in filenames:
            list.append(os.path.join(filepath,filename))
    return list

def main(img_pattern: str,
         mask_pattern: Optional[str] = None,
         weights_path='fpn_inception.h5',
         out_dir='submit/',
         side_by_side: bool = False,
         video: bool = False):
    def sorted_glob(pattern):
        return sorted(glob(pattern))
    imgs = sorted_glob(img_pattern)
    masks = sorted_glob(mask_pattern) if mask_pattern is not None else [None for _ in imgs]
    pairs = zip(imgs, masks)
    names = sorted([os.path.basename(x) for x in glob(img_pattern)])
    predictor = Predictor(weights_path=weights_path)

    os.makedirs(out_dir, exist_ok=True)
    for name, pair in tqdm(zip(names, pairs), total=len(names)):
        f_img, f_mask = pair
        img, mask = map(cv2.imread, (f_img, f_mask))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        pred = predictor(img, mask)
        if side_by_side:
            pred = np.hstack((img, pred))
        pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(out_dir, name),
                    pred)

if __name__ == '__main__':
    img_dir = './test_img'
    img_path = get_files(img_dir)

    for i in img_path:
        main(i)