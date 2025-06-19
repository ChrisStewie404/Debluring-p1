import onnxruntime as ort
import onnx
import numpy as np
import os
from pathlib import Path
import yaml
from glob import glob
import cv2
import albumentations as albu
import re

def get_normalize():
    normalize = albu.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    normalize = albu.Compose([normalize], additional_targets={'target': 'image'})
    def process(a, b):
        r = normalize(image=a, target=b)
        return r['image'], r['target']

    return process

def sorted_glob(pattern):
    return sorted(glob(pattern))

def get_files(img_dir:str):
    list=[]
    for filepath,dirnames,filenames in os.walk(img_dir):
        for filename in filenames:
            list.append(os.path.join(filepath,filename))
    return list

def yaml_load(file="data.yaml", append_filename=False):
    assert Path(file).suffix in {".yaml", ".yml"}, f"Attempting to load non-YAML file {file} with yaml_load()"
    with open(file, errors="ignore", encoding="utf-8") as f:
        s = f.read()  # string

        # Remove special characters
        if not s.isprintable():
            s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+", "", s)

        # Add YAML filename to dict and return
        data = yaml.safe_load(s) or {}  # always return a dict (yaml.safe_load() may return None for empty files)
        if append_filename:
            data["yaml_file"] = str(file)
        return data

class yolo11onnx:
    def __init__(self):
        self.color_palette = np.random.uniform(0,255,size=(80,3))
        # load name classes
        self.name_classes = yaml_load("coco8.yaml")["names"]

    def draw_detections(self,img,box,score,class_id):
        x1,y1,w,h = box
        color = self.color_palette[class_id]
        cv2.rectangle(img,(int(x1),int(y1)),(int(x1+w),int(y1+h)),color,2)
        label = f"{self.name_classes[class_id]}: {score:.2f}"

        (label_w, label_h), _ = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.5,1)
        label_x = x1
        label_y = y1-10 if y1-10 > label_h else y1+10

        cv2.rectangle(
            img, (label_x,label_y+label_h),(label_x+label_w,label_y+label_h),color,
        )
        cv2.putText(img,label,(label_x,label_y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)


if __name__ == '__main__':

    img_dir='./test_img'
    final_out_dir = 'submit_pipeline'
    deblur_out_dir = 'submit_deblur'
    fpn_path = 'FPNInception_736_1312.onnx'
    yolo_path = 'yolo11n.onnx'
    fpn_model = onnx.load(fpn_path)
    yolo_model = onnx.load(yolo_path)
    mask_pattern = None

    img_path = get_files(img_dir)
    onnx.checker.check_model(fpn_model)
    onnx.checker.check_model(yolo_model)
    yolo_processor = yolo11onnx()
    for img_pattern in img_path:
        imgs = sorted_glob(img_pattern)
        masks = sorted_glob(mask_pattern) if mask_pattern is not None else [None for _ in imgs]
        names = sorted([os.path.basename(x) for x in glob(img_pattern)])
        pairs = zip(imgs, masks)

        normalize_fn = get_normalize()

        os.makedirs(final_out_dir, exist_ok = True)
        os.makedirs(deblur_out_dir, exist_ok= True)
        for name, pair in zip(names,pairs):
            # fpn preprocess
            f_img, f_mask = pair
            img_o = cv2.imread(f_img)
            img = cv2.cvtColor(img_o,cv2.COLOR_BGR2RGB)
            print(img[0].size)
            # img,_ = normalize_fn(img,img)
            # print(img)
            # img_h, img_w, _ = img.shape
            # block_size = 32
            # min_h = (img_h // block_size + 1) * block_size
            # min_w = (img_w // block_size + 1) * block_size
            # pad_params = {
            #     'mode' : 'constant',
            #     'constant_values' : 0,
            #     'pad_width' : ((0,min_h - img_h), (0,min_w - img_w),(0, 0))
            # }
            # img = np.pad(img, **pad_params)

            # # (H,W,C) to (C,H,W)
            # img_t = img.transpose((2,0,1))
            # # add batch size dimension
            # img_n = np.expand_dims(img_t,axis=0)
            # fpn_cess = ort.InferenceSession(fpn_path)
            # outputs = fpn_cess.run(None,{'input.1': img_n})

            # # fpn postprocess
            # output,  = outputs
            # d_imgnp = output[0]
            # d_imgnp = (np.transpose(d_imgnp,(1,2,0)) + 1) / 2.0 * 255.0
            # d_imgnp = d_imgnp.astype(np.uint8)
            # d_img = cv2.cvtColor(d_imgnp,cv2.COLOR_RGB2BGR)[:img_h,:img_w,:]
            # cv2.imwrite(os.path.join(deblur_out_dir,name),d_img)

            # # yolo preprocess
            # h,w,_ = d_img.shape
            # p_img = cv2.cvtColor(d_img,cv2.COLOR_BGR2RGB)
            # p_img = cv2.resize(p_img,(640,640))
            # p_imgnp = np.array(p_img) / 255.0
            # p_imgnp = np.transpose(p_imgnp,(2,0,1))
            # p_imgnp = np.expand_dims(p_imgnp,axis=0).astype(np.float32)

            # yolo_cess = ort.InferenceSession(yolo_path)
            # outputs = yolo_cess.run(None,{'images': p_imgnp})
            

            # # yolo postprocess
            # output = np.transpose(np.squeeze(outputs[0]))
            # rows = output.shape[0]

            # thrshld = 0.4
            # iou_thrshld = 0.5
            # boxes = []
            # scores = []
            # class_ids = []

            # x_fac = img_w / 640
            # y_fac = img_h / 640

            # for i in range(rows):
            #     classes_scores = output[i][4:]
            #     max_score = np.amax(classes_scores)
            #     if max_score >= thrshld:
            #         class_id = np.argmax(classes_scores)
            #         x,y,w,h = output[i][0], output[i][1], output[i][2], output[i][3]

            #         left = int((x-w/2)*x_fac)
            #         top = int((y-h/2)*y_fac)
            #         width = int(w*x_fac)
            #         height = int(h*y_fac)

            #         class_ids.append(class_id)
            #         scores.append(max_score)
            #         boxes.append([left,top,width,height])

            # indices = cv2.dnn.NMSBoxes(boxes,scores,thrshld,iou_thrshld)
            # for i in indices:
            #     box = boxes[i]
            #     score = scores[i]
            #     class_id = class_ids[i]
            #     # print(yolo_processor.name_classes[class_id])
            #     yolo_processor.draw_detections(d_img,box,score,class_id)

            # cv2.imwrite(os.path.join(final_out_dir,name),d_img)

            









            