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

int main(){
    BackendConfig backend_config;
    Express::Executor::getGlobalExecutor()->setGlobalExecutorConfig(MNN_FORWARD_CPU,backend_config,4);
    auto img = CV::imread("../test_img/000027.png"); // img path modified later

    auto rgb = CV::cvtColor(img,CV::COLOR_BGR2RGB);
    
    
    float mean[3]={0.5, 0.5, 0.5};
    float std[3]={0.5, 0.5, 0.5};
    auto info = rgb->getInfo();
    auto shape = info->dim;
    const auto size = info->size;
    auto ori_format = CV::NHWC;
    auto format = CV::NCHW;
    // std::cout<<"image shape: \n";
    // for(auto &dim: shape){
    //     std::cout<<dim<<' ';
    // }
    // std::cout<<std::endl;

    // imgarr in NHWC format
    const uint8_t *imgarr = rgb->readMap<uint8_t>();
    const int height = shape[0];
    const int width = shape[1];
    const auto pshape = Express::INTS({height,width,CHNL});
    const auto ori_padshape = Express::INTS({PAD_H,PAD_W,CHNL});  // in NHWC format
    const auto padshape = Express::INTS({CHNL,PAD_H,PAD_W});      // in NCHW format
    float *pc = new float[PAD_IMGSIZE];

    // padding with 0 -> (H = 736, W = 1312, C = 3)
    uint8_t *tmpimg = new uint8_t[PAD_IMGSIZE];
    
    for(size_t c=0;c<3;c++){
        for(size_t h=0;h<PAD_H;h++){  
            for(size_t w=0;w<PAD_W;w++){                  
                if(h<height && w<width) tmpimg[((c*PAD_H)+h)*PAD_W+w] = imgarr[((h*width)+w)*CHNL+c];
                else tmpimg[((c*PAD_H)+h)*PAD_W+w] = 0;
            }
        }
    }

    // // write image
    // uint8_t *ori_tmpimg = new uint8_t[PAD_IMGSIZE];
    //     for(size_t h=0;h<PAD_H;h++){  
    //     for(size_t w=0;w<PAD_W;w++){  
    //         for(size_t c=0;c<3;c++){
    //             ori_tmpimg[((h*PAD_W)+w)*CHNL+c] = tmpimg[((c*PAD_H)+h)*PAD_W+w];
    //         }
    //     }
    // }    
    // auto padimg = CV::_Const(ori_tmpimg,ori_padshape,ori_format,halide_type_of<uint8_t>());
    // auto fpadimg = CV::cvtColor(padimg,CV::COLOR_RGB2BGR);
    // if(CV::imwrite("../submit_pad_img/000027.png",fpadimg,{})) std::cout<<"write tmp img success\n";
    // else std::cout<<"write tmp img fail\n";
    // free(ori_tmpimg);

    for(size_t i=0;i<PAD_IMGSIZE;i++) pc[i] = tmpimg[i];
    auto frgb = CV::_Const(pc,padshape,format,halide_type_of<float>());

    // // img * 2.0f
    for(size_t i=0;i<PAD_IMGSIZE;i++) pc[i] = 2.0f;
    auto imgvarc = CV::_Const(pc,padshape,format,halide_type_of<float>());
    Express::VARP crgb = frgb*imgvarc;
    
    // // img /= 255.0f
    for(size_t i=0;i<PAD_IMGSIZE;i++) pc[i] = 255.0f;
    auto imgvard = CV::_Const(pc,padshape,format,halide_type_of<float>());
    Express::VARP crgbd = crgb/imgvard;

    // // img -= 1.0f
    for(size_t i=0;i<PAD_IMGSIZE;i++) pc[i] = 1.0f;
    auto imgvarb = CV::_Const(pc,padshape,format,halide_type_of<float>());
    Express::VARP norm_rgb = crgbd - imgvarb;

    free(pc);
    free(tmpimg);

    // Expand dims 
    auto exp_rgb = Express::_ExpandDims(norm_rgb,0);
    // Deblur
    // load model
    const std::string FPN_file = "../FPNInception_736_1312.mnn";
    const std::vector<std::string> fpn_input_names{"input.1"};
    const std::vector<std::string> fpn_output_names{"3695"};    // rediculous output name! (look it up on NETRON)
    Express::Module::Config fpn_mdconfig;
    std::unique_ptr<Express::Module> fpn_module;
    fpn_module.reset(Express::Module::load(fpn_input_names,fpn_output_names,FPN_file.c_str(),nullptr,&fpn_mdconfig));
    std::vector<Express::VARP> fpn_inputs(1);
    fpn_inputs[0] = exp_rgb;
    std::vector<Express::VARP> fpn_outputs = fpn_module->onForward(fpn_inputs);
    auto deblur_imgshape = fpn_outputs[0]->getInfo()->dim;
    
    auto deblur_rgb = Express::_Squeeze(fpn_outputs[0],{0});
    auto deblur_fpimg = deblur_rgb->readMap<float>();

    uint8_t *deblur_uimg = new uint8_t[PAD_IMGSIZE];
    uint8_t *yolo_preinput = new uint8_t[size];
    for(size_t h=0;h<PAD_H;h++){  
        for(size_t w=0;w<PAD_W;w++){  
            for(size_t c=0;c<3;c++){
                deblur_uimg[((h*PAD_W)+w)*CHNL+c] = (float)((deblur_fpimg[((c*PAD_H)+h)*PAD_W+w]+1.0f)/2.0f*255.0f);
                if(h<height && w<width) yolo_preinput[((h*width)+w)*CHNL+c] = (float)((deblur_fpimg[((c*PAD_H)+h)*PAD_W+w]+1.0f)/2.0f*255.0f);
                // deblur_uimg[((c*PAD_H)+h)*PAD_W+w] = (deblur_fpimg[((c*PAD_H)+h)*PAD_W+w]+1)/2.0*255;
            }
        }
    }    
    auto imgdeblur = CV::_Const(deblur_uimg,ori_padshape,ori_format,halide_type_of<uint8_t>());
    auto deblur_final_img = CV::cvtColor(imgdeblur,CV::COLOR_RGB2BGR);
    if(CV::imwrite("../submit_deblur_img/000027.png",deblur_final_img,{})) std::cout<<"write deblur img success\n";
    else std::cout<<"write deblur img fail\n";   
    free(deblur_uimg);
    // DON'T FORGET PERMUTATION!!!
    
    auto yolo_rgb = CV::_Const(yolo_preinput,pshape,ori_format,halide_type_of<uint8_t>());
    yolo_rgb = CV::cvtColor(yolo_rgb,CV::COLOR_BGR2RGB);
    CV::Size_<int> yolo_rgb_size_(640,640);
    CV::Size yolo_rgb_size(yolo_rgb_size_);
    yolo_rgb = CV::resize(yolo_rgb,yolo_rgb_size);
    auto resize_shape = yolo_rgb->getInfo()->dim;
    for(auto &dsize:resize_shape) std::cout<<dsize<<' ';
    std::cout<<'\n';
    auto yolo_img = yolo_rgb->readMap<uint8_t>();
    float *yolo_input_hwc = new float[YOLO_SIZE];
    for(size_t i=0;i<YOLO_SIZE;i++) yolo_input_hwc[i]=yolo_img[i] / 255.0f;
    float *yolo_input_chw = new float[YOLO_SIZE];
    for(size_t c=0;c<CHNL;c++){
        for(size_t h=0;h<YOLO_H;h++){
            for(size_t w=0;w<YOLO_W;w++){
                yolo_input_chw[((c*YOLO_H)+h)*YOLO_W+w] = yolo_input_hwc[((h*YOLO_W)+w)*CHNL+c];
            }
        }
    }
    auto yolo_input_rgb = Express::_Const(yolo_input_chw,{CHNL,YOLO_H,YOLO_W},format,halide_type_of<uint8_t>());
    auto yolo_final_input_rgb = Express::_ExpandDims(yolo_input_rgb,0);
    const std::string YOLO_file = "../yolo11n.mnn";
    const std::vector<std::string> yolo_input_names{"images"};
    const std::vector<std::string> yolo_output_names{"output0"};
    Express::Module::Config yolo_mdconfig;
    std::unique_ptr<Express::Module> yolo_module;
    yolo_module.reset(Express::Module::load(yolo_input_names,yolo_output_names,YOLO_file.c_str(),nullptr,&yolo_mdconfig));
    std::vector<Express::VARP> yolo_inputs(1);
    yolo_inputs[0] = yolo_final_input_rgb;
    std::vector<Express::VARP> yolo_outputs = yolo_module->onForward(yolo_inputs);
    std::cout<<"yolo outputs size "<<yolo_outputs.size();
    free(yolo_input_chw);
    free(yolo_input_hwc);    
    free(yolo_preinput);
    return 0;
}