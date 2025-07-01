import _tools as MNNTools
import onnx

def export_mnn2onnx(onnx_path,mnn_path,quant=False):
    convert_args = [
        '',
        '-f',
        'ONNX',
        '--modelFile',
        onnx_path,
        '--MNNModel',
        mnn_path,
        '--keepInputFormat',
        '--bizCode',
        'biz'
    ]
    if quant:
        convert_args.extend(['--weightQuantBits','8'])
    MNNTools.mnnconvert(convert_args)

def slim_onnx(onnx_model, slim_model):
    import onnxslim
    model = onnxslim.slim(onnx_model)
    onnx.save(model, slim_model)

if  __name__ == "__main__":
    fpn_onnx_path = 'FPNInception_736_1312.onnx'
    fpn_slim_path = 'FPNInception_736_1312_slim.onnx'
    yolo_onnx_path = 'yolo11n.onnx'
    fpn_mnn_path = 'FPNInception_736_1312_Q8.mnn'
    yolo_mnn_path = 'yolo11n.mnn'
    # slim_onnx(fpn_onnx_path, fpn_slim_path)
    # export_mnn2onnx(fpn_slim_path, fpn_mnn_path, quant=True)
    export_mnn2onnx(fpn_onnx_path, fpn_mnn_path, quant=True)
    # export_mnn2onnx(yolo_onnx_path,yolo_mnn_path) 
