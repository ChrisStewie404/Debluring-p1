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


if __name__ == "__main__":
    # weights_path = 'fpn_inception.h5'
    
    # model_config = config['model']
    # model = FPNInception(norm_layer=MN.get_norm_layer(norm_type=model_config['norm_layer']))
    # skeleton = get_generator(config['model'])
    # if(torch.cuda.is_available()):
    #     skeleton.load_state_dict(torch.load(weights_path)['model'])
    #     model = skeleton.cuda()
    # else:
    #     skeleton.load_state_dict(torch.load(weights_path,map_location=torch.device('cpu'))['model'])
    #     model = skeleton.cpu()
    

    img_dir = r'./test_img/*.png'
    weights_path = 'fpn_inception.h5'



    predictor = Predictor(weights_path=weights_path)

    # Generate model
    with open('config/config.yaml',encoding='utf-8') as cfg:
        config = yaml.load(cfg,Loader=yaml.FullLoader)
    model_config = config['model']
    model_g = FPNInception(norm_layer=get_norm_layer(norm_type=model_config['norm_layer']))
    new_state_dict = {}
        
    if torch.cuda.is_available():
        # model_dp.load_state_dict(torch.load(weights_path,weights_only=False)['model'])
        state_dict = torch.load(weights_path,weights_only=False)['model']
        for k,v in state_dict.items():
            k = k.replace('module.','')
            new_state_dict[k] = v
        model_g.load_state_dict(new_state_dict)
    else:
        # model_dp.load_state_dict(torch.load(weights_path,weights_only=False,map_location=torch.device('cpu'))['model'])
        state_dict = torch.load(weights_path,weights_only=False,map_location=torch.device('cpu'))['model']
        for k,v in state_dict.items():
            k = k.replace('module.','')
            new_state_dict[k] = v
        model_g.load_state_dict(new_state_dict)
    # model_g.train(True)
    for module in model_g.modules():
        if isinstance(module, torch.nn.InstanceNorm2d):
            module.track_running_stats = False
    model_g.eval()
    # model_g = get_generator(config['model'])

    # # Generate example input
    # f_img = glob(img_dir)[0]
    # f_mask = None
    # img, mask = map(cv2.imread, (f_img, f_mask))       
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # (img, mask), h, w = predictor._preprocess(img,mask)
    # inputs = []
    # with torch.no_grad():
    #     if torch.cuda.is_available():
    #         inputs = [img.cuda()]
    #     else:
    #         inputs = [img.cpu()]
        
    example_input = (torch.randn((1,3,736,1312)),)   
    model_onnx = torch.onnx.export(
        model_g,
        example_input,
        "FPNInception.onnx",
        # dynamo=True,
        opset_version=14,
        autograd_inlining=False
    )
    model_onnx.save("FPNInception14.onnx")
    # # Load weight
    # if torch.cuda.is_available():
    #     model_g.load_state_dict(torch.load(weights_path)['model'])
    #     model = model_g.cuda()
    #     model.train(True)

        
    #     torch.onnx.export(
    #         model,
    #         example_input,
    #         "FPNInception.onnx",
    #         dynamo=True
    #     )
    # else:
    #     model_g.load_state_dict(torch.load(weights_path,map_location=torch.device('cpu'))['model'])
    #     model = model_g.cpu()
    #     model.train(True)
    
    
        # torch.onnx.export(
        #     model,
        #     example_input,
        #     "FPNInception.onnx",
        #     dynamo=True
        # )

    # Set to evaluation mode
    
    # torch.onnx.export(
    #     model,
    #     (input_tensors[0],),
    #     "FPNInception.onnx",
    #     input_names=["blurred1"],
    #     dynamo=True
    # )

