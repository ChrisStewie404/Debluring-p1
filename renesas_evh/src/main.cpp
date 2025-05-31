#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/Executor.hpp>
#include <tools/cv/include/cv/cv.hpp>
#include <MNN/expr/Executor.hpp>
#include <tools/cv/include/cv/imgproc/color.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>
#include <iostream>
using namespace MNN;
int main(){
    BackendConfig backend_config;
    Express::Executor::getGlobalExecutor()->setGlobalExecutorConfig(MNN_FORWARD_CPU,backend_config,4);
    auto img = CV::imread("../test_img/000027.png"); // img path modified later
    auto rgb = CV::cvtColor(img,CV::COLOR_BGR2RGB);
    rgb = CV::_Permute(rgb,{2,0,1});
    auto info = rgb->getInfo();
    std::cout << rgb->readMap<float>()[0] << std::endl;
    
    float mean[3]={0.5, 0.5, 0.5};
    float std[3]={0.5, 0.5, 0.5};
    return 0;
}