import torch
import torch.fx
import torch.nn as nn
from typing import Optional
from models.fpn_inception import FPNInception
from models.networks import get_norm_layer
from models.networks import get_generator
from predict import Predictor
import numpy as np
import cv2
from glob import glob
import yaml
import os
from tqdm import tqdm
from aug import get_normalize
from fire import Fire

class InstanceNormAlternative(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps=1e-6

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        assert (len(inp.shape)==4), "InstanceNorm Shape Error!"
        desc = 1 / (inp.var(axis=[2, 3], keepdim=True, unbiased=False) + self.eps) ** 0.5
        retval = (inp - inp.mean(axis=[2, 3], keepdim=True)) * desc
        return retval
    
class Predictor:
    def __init__(self, weights_path: str, model_name: str = ''):

        # Generate model
        with open('config/config.yaml',encoding='utf-8') as cfg:
            config = yaml.load(cfg,Loader=yaml.FullLoader)
        model_config = config['model']
        model_g = FPNInception(norm_layer=get_norm_layer(norm_type=model_config['norm_layer']))
        new_state_dict = {}
            
        if torch.cuda.is_available():
            state_dict = torch.load(weights_path,weights_only=False)['model']
            for k,v in state_dict.items():
                k = k.replace('module.','')
                new_state_dict[k] = v
            model_g.load_state_dict(new_state_dict)
        else:
            state_dict = torch.load(weights_path,weights_only=False,map_location=torch.device('cpu'))['model']
            for k,v in state_dict.items():
                k = k.replace('module.','')
                new_state_dict[k] = v
            model_g.load_state_dict(new_state_dict)
        
        replace_list =[k.split('.') 
                        for k, m in model_g.named_modules(remove_duplicate=False) 
                        if isinstance(m, torch.nn.InstanceNorm2d)]
        for *parent, id in replace_list:
            model_g.get_submodule('.'.join(parent))[int(id)] = InstanceNormAlternative()

        model = model_g
        if torch.cuda.is_available():
            # model.load_state_dict(torch.load(weights_path)['model'])
            self.model = model.cuda()
        else:
            # model.load_state_dict(torch.load(weights_path,map_location=torch.device('cpu'))['model'])
            self.model = model.cpu()

        self.model.train(True)
        # GAN inference should be in train mode to use actual stats in norm layers,
        # it's not a bug
        self.normalize_fn = get_normalize()

    @staticmethod
    def _array_to_batch(x):
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, 0)
        return torch.from_numpy(x)

    def _preprocess(self, x: np.ndarray, mask: Optional[np.ndarray]):
        x, _ = self.normalize_fn(x, x)
        if mask is None:
            mask = np.ones_like(x, dtype=np.float32)
        else:
            mask = np.round(mask.astype('float32') / 255)

        h, w, _ = x.shape
        block_size = 32
        min_height = (h // block_size + 1) * block_size
        min_width = (w // block_size + 1) * block_size

        pad_params = {'mode': 'constant',
                      'constant_values': 0,
                      'pad_width': ((0, min_height - h), (0, min_width - w), (0, 0))
                      }
        x = np.pad(x, **pad_params)
        mask = np.pad(mask, **pad_params)

        return map(self._array_to_batch, (x, mask)), h, w

    @staticmethod
    def _postprocess(x: torch.Tensor) -> np.ndarray:
        x, = x
        x = x.detach().cpu().float().numpy()
        x = (np.transpose(x, (1, 2, 0)) + 1) / 2.0 * 255.0
        return x.astype('uint8')

    def __call__(self, img: np.ndarray, mask: Optional[np.ndarray], ignore_mask=True) -> np.ndarray:
        (img, mask), h, w = self._preprocess(img, mask)
        with torch.no_grad():
            if torch.cuda.is_available():
                inputs = [img.cuda()]
            else:
                inputs = [img.cpu()]
            if not ignore_mask:
                inputs += [mask]
            pred = self.model(*inputs)
        return self._postprocess(pred)[:h, :w, :]

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

def getfiles():
    filenames = os.listdir(r'.\dataset1\blur')
    print(filenames)
def get_files(img_dir:str):
    list=[]
    for filepath,dirnames,filenames in os.walk(img_dir):
        for filename in filenames:
            list.append(os.path.join(filepath,filename))
    return list



if __name__ == "__main__": 
    img_dir = './test_img'
    weights_path = 'fpn_inception.h5'
    
    # Generate model
    with open('config/config.yaml',encoding='utf-8') as cfg:
        config = yaml.load(cfg,Loader=yaml.FullLoader)
    model_config = config['model']
    model_g = FPNInception(norm_layer=get_norm_layer(norm_type=model_config['norm_layer']))
    new_state_dict = {}
        
    if torch.cuda.is_available():
        state_dict = torch.load(weights_path,weights_only=False)['model']
        for k,v in state_dict.items():
            k = k.replace('module.','')
            new_state_dict[k] = v
        model_g.load_state_dict(new_state_dict)
    else:
        state_dict = torch.load(weights_path,weights_only=False,map_location=torch.device('cpu'))['model']
        for k,v in state_dict.items():
            k = k.replace('module.','')
            new_state_dict[k] = v
        model_g.load_state_dict(new_state_dict)
    
    replace_ins_list =[k.split('.') 
                    for k, m in model_g.named_modules(remove_duplicate=False) 
                    if isinstance(m, torch.nn.InstanceNorm2d)]
    for *parent, id in replace_ins_list:
        model_g.get_submodule('.'.join(parent))[int(id)] = InstanceNormAlternative()

    with torch.no_grad():
        example_input = (torch.randn((1,3,736,1312)),)   
        model_onnx = torch.onnx.export(
            model_g,
            example_input,
            "FPNInception_736_1312.onnx",
            # dynamo=True,
            opset_version=15,
            # autograd_inlining=False,
            # dynamic_axes=None,
        )
