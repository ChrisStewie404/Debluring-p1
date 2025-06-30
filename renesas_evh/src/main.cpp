#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/Executor.hpp>
#include <tools/cv/include/cv/cv.hpp>
#include <MNN/expr/Executor.hpp>
#include <tools/cv/include/cv/imgproc/color.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>
#include <MNN/ImageProcess.hpp>
#include <MNN/Interpreter.hpp>
#include <iostream>
using namespace MNN;
#define CHNL        (3)
#define PAD_H       (736)
#define PAD_W       (1312)
#define PAD_IMGSIZE (PAD_H*PAD_W*CHNL)
#define YOLO_H      (640)
#define YOLO_W      (640)
#define YOLO_SIZE   (YOLO_H*YOLO_W*CHNL)
#define THRSHLD     (0.30f)
#define IOU_THRSHLD (0.55f)

#define MAX_DETECTIONS (20)
size_t argmax(const float *arr, size_t len);
std::vector<std::string> name_classes = {
  "person",
  "bicycle",
  "car",
  "motorcycle",
  "airplane",
  "bus",
  "train",
  "truck",
  "boat",
  "traffic light",
  "fire hydrant",
  "stop sign",
  "parking meter",
  "bench",
  "bird",
  "cat",
  "dog",
  "horse",
  "sheep",
  "cow",
  "elephant",
  "bear",
  "zebra",
  "giraffe",
  "backpack",
  "umbrella",
  "handbag",
  "tie",
  "suitcase",
  "frisbee",
  "skis",
  "snowboard",
  "sports ball",
  "kite",
  "baseball bat",
  "baseball glove",
  "skateboard",
  "surfboard",
  "tennis racket",
  "bottle",
  "wine glass",
  "cup",
  "fork",
  "knife",
  "spoon",
  "bowl",
  "banana",
  "apple",
  "sandwich",
  "orange",
  "broccoli",
  "carrot",
  "hot dog",
  "pizza",
  "donut",
  "cake",
  "chair",
  "couch",
  "potted plant",
  "bed",
  "dining table",
  "toilet",
  "tv",
  "laptop",
  "mouse",
  "remote",
  "keyboard",
  "cell phone",
  "microwave",
  "oven",
  "toaster",
  "sink",
  "refrigerator",
  "book",
  "clock",
  "vase",
  "scissors",
  "teddy bear",
  "hair drier",
  "toothbrush"
};
std::vector<CV::Scalar> colors = {
    {240,46,117},
    {100,191,120},
    {49,172,146},
    {8,13,55},
    {165,205,39},
    {59,147,71},
    {220,150,211},
    {81,142,146},
    {95,53,221},
    {96,240,241},
    {16,150,195},
    {196,221,233},
    {140,145,44},
    {158,202,200},
    {163,27,101},
    {105,85,70},
    {186,82,191},
    {15,175,153},
    {147,124,211},
    {46,39,68},
    {229,35,149},
    {2,118,138},
    {86,145,169},
    {209,146,26},
    {24,25,9},
    {15,198,254},
    {206,13,234},
    {6,71,131},
    {17,217,121},
    {212,102,90},
    {171,221,20},
    {146,191,78},
    {240,7,76},
    {88,113,38},
    {169,59,239},
    {8,160,79},
    {161,165,120},
    {61,37,57},
    {67,212,193},
    {196,118,239},
    {61,35,30},
    {142,201,79},
    {31,195,27},
    {137,63,154},
    {52,90,154},
    {190,93,35},
    {208,196,250},
    {85,98,109},
    {153,96,241},
    {57,168,197},
    {154,12,221},
    {251,123,35},
    {11,233,12},
    {192,47,118},
    {181,67,227},
    {144,133,153},
    {47,9,0},
    {124,182,156},
    {195,111,148},
    {103,101,153},
    {222,96,208},
    {193,224,0},
    {162,80,157},
    {111,72,39},
    {4,133,149},
    {196,24,246},
    {163,11,205},
    {25,8,169},
    {93,201,221},
    {99,233,83},
    {99,27,61},
    {144,93,103},
    {71,208,166},
    {28,234,247},
    {63,16,177},
    {3,249,35},
    {66,81,181},
    {102,70,86},
    {69,249,116},
    {250,66,99},
};
int main(){
    BackendConfig backend_config;
    Express::Executor::getGlobalExecutor()->setGlobalExecutorConfig(MNN_FORWARD_CPU,backend_config,4);
    const std::string file = "020.png";
    auto img = CV::imread("../test_img/"+file); // img path modified later

    auto rgb = CV::cvtColor(img,CV::COLOR_BGR2RGB);
    
    
    float mean[3]={0.5, 0.5, 0.5};
    float std[3]={0.5, 0.5, 0.5};
    auto info = rgb->getInfo();
    auto shape = info->dim;
    const auto size = info->size;
    auto ori_format = CV::NHWC;
    auto format = CV::NCHW;

    // imgarr in NHWC format
    // const uint8_t *imgarr = rgb->readMap<uint8_t>();
    const int height = shape[0];
    const int width = shape[1];
    const auto pshape = Express::INTS({height,width,CHNL});
    const auto ori_padshape = Express::INTS({PAD_H,PAD_W,CHNL});  // in NHWC format
    const auto padshape = Express::INTS({CHNL,PAD_H,PAD_W});      // in NCHW format
    // float *pc = new float[PAD_IMGSIZE];

    // // padding with 0 -> (H = 736, W = 1312, C = 3)
    // uint8_t *tmpimg = new uint8_t[PAD_IMGSIZE];
    
    // for(size_t c=0;c<3;c++){
    //     for(size_t h=0;h<PAD_H;h++){  
    //         for(size_t w=0;w<PAD_W;w++){                  
    //             if(h<height && w<width) tmpimg[((c*PAD_H)+h)*PAD_W+w] = imgarr[((h*width)+w)*CHNL+c];
    //             else tmpimg[((c*PAD_H)+h)*PAD_W+w] = 0;
    //         }
    //     }
    // }

    // for(size_t i=0;i<PAD_IMGSIZE;i++) pc[i] = tmpimg[i];
    // auto frgb = CV::_Const(pc,padshape,format,halide_type_of<float>());
    // free(tmpimg);

    std::vector<int> pdavals {0,PAD_H-height,PAD_W-width,0,0};
    auto pads = Express::_Const(static_cast<void*>(pdavals.data()),{3,2},Express::NCHW,halide_type_of<int>());
    auto frgb = Express::_Pad(rgb,pads,CV::CONSTANT);
    for(auto &dsize:frgb->getInfo()->dim) std::cout<<dsize<<' ';
    std::cout<<'\n';

    // // img * 2.0f
    // Express::VARP _2f = Express::_Scalar<float>(2.0);
    // Express::VARP crgb = frgb*_2f;        
    // // img /= 255.0f
    // Express::VARP _255f = Express::_Scalar<float>(255.0f);
    // Express::VARP crgbd = crgb/_255f;
    // // img -= 1.0f
    // Express::VARP _1f = Express::_Scalar<float>(1.0f);
    // Express::VARP norm_rgb = crgbd - _1f;

    // auto normimg_rgb = norm_rgb->readMap<float>();
    // for(size_t c=0;c<CHNL;c++){
    //     for(size_t h=0;h<PAD_H;h++){
    //         for(size_t w=0;w<PAD_W;w++){
    //             if(h<height && w<width) pc[((c*PAD_H)+h)*PAD_W+w] = normimg_rgb[((c*PAD_H)+h)*PAD_W+w];
    //             else pc[((c*PAD_H)+h)*PAD_W+w] = 0;
    //         }
    //     }
    // }
    // norm_rgb = CV::_Const(pc,padshape,format,halide_type_of<float>());

    // free(pc);

    // // Expand dims 
    // auto exp_rgb = Express::_ExpandDims(norm_rgb,0);
    
    // // load model
    // const std::string FPN_file = "../FPNInception_736_1312.mnn";
    // const std::vector<std::string> fpn_input_names{"input.1"};
    // const std::vector<std::string> fpn_output_names{"3695"};    // rediculous output name! (look it up on NETRON)
    // Express::Module::Config fpn_mdconfig;
    // std::unique_ptr<Express::Module> fpn_module;
    // fpn_module.reset(Express::Module::load(fpn_input_names,fpn_output_names,FPN_file.c_str(),nullptr,&fpn_mdconfig));
    // std::vector<Express::VARP> fpn_inputs(1);
    // fpn_inputs[0] = exp_rgb;
    // std::vector<Express::VARP> fpn_outputs = fpn_module->onForward(fpn_inputs);
    // if (fpn_outputs[0]->getInfo()->order == MNN::Express::NC4HW4) {
    //     fpn_outputs[0] = _Convert(fpn_outputs[0], MNN::Express::NCHW);
    // }
    // auto deblur_rgb = Express::_Squeeze(fpn_outputs[0],{0});

    // auto deblur_fpimg = deblur_rgb->readMap<float>(); 
    // uint8_t *yolo_preinput = new uint8_t[size];
    // for(size_t h=0;h<PAD_H;h++){  
    //     for(size_t w=0;w<PAD_W;w++){  
    //         for(size_t c=0;c<3;c++){
    //             if(h<height && w<width){
    //                 yolo_preinput[((h*width)+w)*CHNL+c] = (uint8_t)((deblur_fpimg[((c*PAD_H)+h)*PAD_W+w]+1.0f)/2.0f*255.0f);
    //             }
    //         }
    //     }
    // }    
    // auto deblur_final_img = Express::_Const(yolo_preinput,pshape,ori_format,halide_type_of<uint8_t>());
    // deblur_final_img = CV::cvtColor(deblur_final_img,CV::COLOR_RGB2BGR);
    
    // auto yolo_rgb = CV::_Const(yolo_preinput,pshape,ori_format,halide_type_of<uint8_t>());
    // CV::Size_<int> yolo_rgb_size_(640,640);
    // CV::Size yolo_rgb_size(yolo_rgb_size_);
    // yolo_rgb = CV::resize(yolo_rgb,yolo_rgb_size);
    // auto yolo_input_rgb = Express::_Cast<float>(yolo_rgb);
    // yolo_input_rgb = yolo_input_rgb / _255f;
    // auto yolo_final_input_rgb = Express::_ExpandDims(yolo_input_rgb,0);
    // free(yolo_preinput);
    
    // const std::string YOLO_file = "../yolo11n.mnn";
    // const std::vector<std::string> yolo_input_names{"images"};
    // const std::vector<std::string> yolo_output_names{"output0"};
    // Express::Module::Config yolo_mdconfig;
    // std::unique_ptr<Express::Module> yolo_module;
    // yolo_module.reset(Express::Module::load(yolo_input_names,yolo_output_names,YOLO_file.c_str(),nullptr,&yolo_mdconfig));
    // std::vector<Express::VARP> yolo_inputs(1);

    // yolo_inputs[0] = yolo_final_input_rgb;
    // std::vector<Express::VARP> yolo_outputs = yolo_module->onForward(yolo_inputs);
    
    // Express::VARP yolo_output = Express::_Transpose(Express::_Squeeze(yolo_outputs[0],{0}),{1,0});

    // auto yolo_arr = yolo_output->readMap<float>();
    // const size_t rows = yolo_output->getInfo()->dim[0];
    // const size_t cols = yolo_output->getInfo()->dim[1];

    // const float x_fac = width *1.0f / YOLO_W;
    // const float y_fac = height *1.0f / YOLO_H;

    // std::vector<size_t> class_ids;
    // float *rects = new float[rows*4];
    // float *score_arr = new float[rows];
    // int total = 0;
    // for(size_t i=0;i<rows;i++){
    //     size_t class_id = argmax(yolo_arr+(i*cols+4),cols-4);
    //     float max_score = yolo_arr[i*cols+4+class_id];
    //     if(max_score > THRSHLD){
            
    //         auto x = yolo_arr[i*cols];
    //         auto y = yolo_arr[i*cols+1];
    //         auto w = yolo_arr[i*cols+2];
    //         auto h = yolo_arr[i*cols+3];

    //         float *rect = rects+total*4;
    //         rect[0] = ((x-w/2.0f)*x_fac);
    //         rect[1] = ((y-h/2.0f)*y_fac);
    //         rect[2] = (w*x_fac);
    //         rect[3] = (h*y_fac);
    //         score_arr[total] = max_score;
    //         class_ids.push_back(class_id);
            
    //         total++;
    //     }
    // }  
    // Express::VARP boxes = Express::_Const(rects,{total,4});
    // Express::VARP scores = Express::_Const(score_arr,{total});
    // auto indices_varp = Express::_Nms(boxes,scores,MAX_DETECTIONS,IOU_THRSHLD,THRSHLD);
   
   
    // size_t obj_size = indices_varp->getInfo()->size;
    // if(obj_size > 0){
    //     auto indices = indices_varp->readMap<int>();
    //     for(int i=0;i<obj_size;i++){
    //         if(indices[i] == -1) continue;
    //         std::cout<<indices[i]<<' ';
    //         std::cout<<name_classes[class_ids[indices[i]]]<<'\n';
    //         auto index = indices[i];
    //         CV::Point pt1,pt2;
    //         pt1.set(rects[index*4],rects[index*4+1]);
    //         pt2.set(rects[index*4]+rects[index*4+2],rects[index*4+1]+rects[index*4+3]);
    //         CV::rectangle(deblur_final_img,pt1,pt2,colors[class_ids[indices[i]]],2);
    //     }        
    // }
    
    // if(CV::imwrite("../final_detected_"+file,deblur_final_img,{})) std::cout<<"write final img success\n";
    // else std::cout<<"write final img fail\n";   

    // class_ids.clear();
    // rgb = CV::resize(rgb,yolo_rgb_size);
    // yolo_input_rgb = Express::_Cast<float>(rgb);
    // yolo_input_rgb = yolo_input_rgb / _255f;
    // yolo_final_input_rgb = Express::_ExpandDims(yolo_input_rgb,0);

    // yolo_inputs[0] = yolo_final_input_rgb;
    // yolo_outputs = yolo_module->onForward(yolo_inputs);

    // yolo_output = Express::_Transpose(Express::_Squeeze(yolo_outputs[0],{0}),{1,0});

    // yolo_arr = yolo_output->readMap<float>();
    // const size_t rows1 = yolo_output->getInfo()->dim[0];
    // const size_t cols1 = yolo_output->getInfo()->dim[1];

    // total = 0;
    // for(size_t i=0;i<rows1;i++){
    //     size_t class_id = argmax(yolo_arr+(i*cols1+4),cols1-4);
    //     float max_score = yolo_arr[i*cols1+4+class_id];
    //     if(max_score > THRSHLD){
            
    //         auto x = yolo_arr[i*cols1];
    //         auto y = yolo_arr[i*cols1+1];
    //         auto w = yolo_arr[i*cols1+2];
    //         auto h = yolo_arr[i*cols1+3];

    //         float *rect = rects+total*4;
    //         rect[0] = ((x-w/2.0f)*x_fac);
    //         rect[1] = ((y-h/2.0f)*y_fac);
    //         rect[2] = (w*x_fac);
    //         rect[3] = (h*y_fac);
    //         score_arr[total] = max_score;
    //         class_ids.push_back(class_id);
            
    //         total++;
    //     }
    // }  
    // boxes = Express::_Const(rects,{total,4});
    // scores = Express::_Const(score_arr,{total});
    // indices_varp = Express::_Nms(boxes,scores,MAX_DETECTIONS,IOU_THRSHLD,THRSHLD);

    
    // obj_size = indices_varp->getInfo()->size;
    // if(obj_size > 0){
    //     auto indices = indices_varp->readMap<int>();
    //     for(int i=0;i<obj_size;i++){
    //         if(indices[i] == -1) continue;
    //         std::cout<<name_classes[class_ids[indices[i]]]<<'\n';
    //         auto index = indices[i];
    //         CV::Point pt1,pt2;
    //         pt1.set(rects[index*4],rects[index*4+1]);
    //         pt2.set(rects[index*4]+rects[index*4+2],rects[index*4+1]+rects[index*4+3]);
    //         CV::rectangle(img,pt1,pt2,colors[class_ids[indices[i]]],2);
    //     }
    // }
    // if(CV::imwrite("../original_detected_"+file,img,{})) std::cout<<"write original img success\n";
    // else std::cout<<"write original img fail\n";   

    // free(score_arr);
    // free(rects);
    return 0;
}
size_t argmax(const float *arr, size_t len){
    size_t argmax = 0;
    float max = arr[0];
    for(size_t i=0;i<len;i++){
        if(max < arr[i]){
            argmax = i;
            max = arr[i];
        }
    }
    return argmax;
}