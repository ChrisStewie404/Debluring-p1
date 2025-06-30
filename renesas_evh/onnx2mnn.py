import _tools as MNNTools

def export_mnn2onnx(onnx_path,mnn_path):
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
        'biz',
        '--weightQuantBits','8',
    ]
    MNNTools.mnnconvert(convert_args)

if  __name__ == "__main__":
    fpn_onnx_path = 'FPNInception_736_1312.onnx'
    yolo_onnx_path = 'yolo11n.onnx'
    fpn_mnn_path = 'FPNInception_736_1312_Q8.mnn'
    yolo_mnn_path = 'yolo11n.mnn'
    export_mnn2onnx(fpn_onnx_path,fpn_mnn_path)
    # export_mnn2onnx(yolo_onnx_path,yolo_mnn_path) 
