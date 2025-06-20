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
#define PAD_H   (736)
#define PAD_W   (1312)
#define CHNL (3)
#define PAD_IMGSIZE (PAD_H*PAD_W*3)
int main(){
    BackendConfig backend_config;
    Express::Executor::getGlobalExecutor()->setGlobalExecutorConfig(MNN_FORWARD_CPU,backend_config,4);
    auto img = CV::imread("../test_img/000027.png"); // img path modified later

    auto rgb = CV::cvtColor(img,CV::COLOR_BGR2RGB);
    
    
    float mean[3]={0.5, 0.5, 0.5};
    float std[3]={0.5, 0.5, 0.5};
    auto info = rgb->getInfo();
    auto shape = info->dim;
    auto size = info->size;
    auto ori_format = CV::NHWC;
    auto format = CV::NCHW;
    // std::cout<<"image shape: \n";
    // for(auto &dim: shape){
    //     std::cout<<dim<<' ';
    // }
    // std::cout<<std::endl;

    // imgarr in NHWC format
    const uint8_t *imgarr = rgb->readMap<uint8_t>();
    int height = shape[0];
    int width = shape[1];
    auto pshape = Express::INTS({height,width,CHNL});
    auto ori_padshape = Express::INTS({PAD_H,PAD_W,CHNL});  // in NHWC format
    auto padshape = Express::INTS({CHNL,PAD_H,PAD_W});      // in NCHW format
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
    const std::vector<std::string> input_names{"input.1"};
    const std::vector<std::string> output_names{"3695"};    // rediculous output name! (look it up on NETRON)
    Express::Module::Config mdconfig;
    std::unique_ptr<Express::Module> module;
    module.reset(Express::Module::load(input_names,output_names,FPN_file.c_str(),nullptr,&mdconfig));
    std::vector<Express::VARP> inputs(1);
    inputs[0] = exp_rgb;
    std::vector<Express::VARP> outputs = module->onForward(inputs);
    auto deblur_imgshape = outputs[0]->getInfo()->dim;
    for(auto &dsize:deblur_imgshape) std::cout<<dsize<<' ';
    std::cout<<'\n';
    
    auto deblur_rgb = Express::_Squeeze(outputs[0],{0});
    auto deblur_fpimg = deblur_rgb->readMap<float>();

    uint8_t *deblur_uimg = new uint8_t[PAD_IMGSIZE];
    for(size_t h=0;h<PAD_H;h++){  
        for(size_t w=0;w<PAD_W;w++){  
            for(size_t c=0;c<3;c++){
                deblur_uimg[((h*PAD_W)+w)*CHNL+c] = (deblur_fpimg[((c*PAD_H)+h)*PAD_W+w]+1.0f)/2.0f*255.0f;
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
   

    return 0;
}