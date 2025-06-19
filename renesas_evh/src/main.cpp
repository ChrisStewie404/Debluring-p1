#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/Executor.hpp>
#include <tools/cv/include/cv/cv.hpp>
#include <MNN/expr/Executor.hpp>
#include <tools/cv/include/cv/imgproc/color.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>
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
    auto padshape = Express::INTS({PAD_H,PAD_W,CHNL});
    float *pc = new float[PAD_IMGSIZE];

    // padding with 0 -> (H = 736, W = 1312, C = 3)
    uint8_t *tmpimg = new uint8_t[PAD_IMGSIZE];
    
    for(size_t h=0;h<PAD_H;h++){  
        for(size_t w=0;w<PAD_W;w++){  
            for(size_t c=0;c<3;c++){
                if(h<height && w<width) tmpimg[((h*PAD_W)+w)*CHNL+c] = imgarr[((h*width)+w)*CHNL+c];
                else tmpimg[((h*PAD_W)+w)*CHNL+c] = 0;
            }
        }
    }
    auto padimg = CV::_Const(tmpimg,padshape,ori_format,halide_type_of<uint8_t>());
    
    // write image
    // auto fpadimg = CV::cvtColor(padimg,CV::COLOR_RGB2BGR);
    // if(CV::imwrite("../commit_img/000027.png",fpadimg,{})) std::cout<<"write tmp img success\n";
    // else std::cout<<"write tmp img fail\n";

    for(size_t i=0;i<PAD_IMGSIZE;i++) pc[i] = tmpimg[i];
    auto frgb = CV::_Const(pc,padshape,format,halide_type_of<float>());

    // // img * 2.0f
    for(size_t i=0;i<PAD_IMGSIZE;i++) pc[i] = 2.0f;
    auto imgvarc = CV::_Const(pc,padshape,format,halide_type_of<float>());
    Express::VARP crgb = frgb*imgvarc;
    // std::cout<<"crgb: "<<crgb->readMap<float>()[0]<<std::endl;
    
    
    // // img /= 255.0f
    for(size_t i=0;i<PAD_IMGSIZE;i++) pc[i] = 255.0f;
    auto imgvard = CV::_Const(pc,padshape,format,halide_type_of<float>());
    Express::VARP crgbd = crgb/imgvard;
    // std::cout<<"crgbd: "<<crgbd->readMap<float>()[0]<<std::endl;

    // // img -= 1.0f
    for(size_t i=0;i<PAD_IMGSIZE;i++) pc[i] = 1.0f;
    auto imgvarb = CV::_Const(pc,padshape,format,halide_type_of<float>());
    Express::VARP norm_rgb = crgbd - imgvarb;
    // std::cout<<"norm: "<<norm_rgb->readMap<float>()[0]<<std::endl;
    
    // // DON'T FORGET PERMUTATION!!!
    // // rgb = CV::_Permute(rgb,{2,0,1});
    free(pc);
    free(tmpimg);
    return 0;
}