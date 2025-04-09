import os
from glob import glob
from typing import Optional
from tqdm import tqdm
import numpy as np

from predict import Predictor
import cv2
from ultralytics import YOLO


def get_files(img_dir:str):
    list=[]
    for filepath,dirnames,filenames in os.walk(img_dir):
        for filename in filenames:
            list.append(os.path.join(filepath,filename))
    return list

def process_video(pairs, predictor, output_dir):
    for video_filepath, mask in tqdm(pairs):
        video_filename = os.path.basename(video_filepath)
        output_filepath = os.path.join(output_dir, os.path.splitext(video_filename)[0]+'_deblur.mp4')
        video_in = cv2.VideoCapture(video_filepath)
        fps = video_in.get(cv2.CAP_PROP_FPS)
        width = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frame_num = int(video_in.get(cv2.CAP_PROP_FRAME_COUNT))
        video_out = cv2.VideoWriter(output_filepath, cv2.VideoWriter_fourcc(*'MP4V'), fps, (width, height))
        tqdm.write(f'process {video_filepath} to {output_filepath}, {fps}fps, resolution: {width}x{height}')
        for frame_num in tqdm(range(total_frame_num), desc=video_filename):
            res, img = video_in.read()
            if not res:
                break
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pred = predictor(img, mask)
            pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
            video_out.write(pred)

def main(img_pattern: str,
         mask_pattern: Optional[str] = None,
         weights_path='fpn_inception.h5',
         out_dir='demo_submit/',
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
    if not video:
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

    else:
        process_video(pairs, predictor, out_dir)

if __name__ == '__main__':
    # img_dir = './test_img'
    # img_path = get_files(img_dir)
    # for i in img_path:
    #     main(i)
    model = YOLO('yolo11x.pt')
    # write to fin_dir
    fin_dir  = './demo_pred/room1b'
    # blurred img from ori_dir
    ori_dir = '../../BAD-NeRF/data/nerf_llff_data/blurcozy2room/images'
    ori_path = get_files(ori_dir)
    # deblurred img from pred_dir
    pred_dir = './submit'
    pred_path =  get_files(pred_dir)

    for p in ori_path:
        result = model(p)[0]
        boxes = result.boxes
        masks = result.masks
        keypoints = result.keypoints
        probs = result.probs
        obb = result.obb
        result.save(filename = os.path.join(fin_dir,os.path.basename(p)))
    # for  p in pred_path:
    #     result = model(p)[0]
    #     boxes = result.boxes
    #     masks = result.masks
    #     keypoints = result.keypoints
    #     probs = result.probs
    #     obb = result.obb
    #     result.show()
    #     result.save(filename = fin_dir+"result.jpg")
