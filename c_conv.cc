#include "ctorch.cnn.h"

Cnn::C_ConvNd::C_ConvNd(int in_channels, int out_channels, int stride, int padding, int dilation, bool bias) : in_channels{in_channels}, out_channels{out_channels}, stride{stride}, padding{padding}, dilation{dilation}, bias{bias}, weights{nullptr} {
    if (bias) {
        this->biases = CTorch::zeros(out_channels);
    }
}
Cnn::C_ConvNd::~C_ConvNd() {
    biases = nullptr;
    input = nullptr;
}

Cnn::CConv1d::CConv1d(int in_channels, int out_channels, int kernel_size, int stride, int padding, int dilation, bool bias) : C_ConvNd(in_channels, out_channels, stride, padding, dilation, bias), kernel_size{kernel_size} {
    this->weights = CTorch::randn(out_channels, in_channels, kernel_size);
}
Cnn::CConv1d::~CConv1d() {
    weights = nullptr;
}
std::unique_ptr<CTorch::CTensor> Cnn::CConv1d::forward(std::unique_ptr<CTorch::CTensor>& input) {

}


Cnn::CConv2d::CConv2d(int in_channels, int out_channels, std::tuple<int, int> kernel_size, int stride, int padding, int dilation, bool bias) : C_ConvNd(in_channels, out_channels, stride, padding, dilation, bias), kernel_size{kernel_size} {
    this->weights = CTorch::randn(out_channels, in_channels, std::get<0>(kernel_size), std::get<1>(kernel_size));
}
Cnn::CConv2d::~CConv2d() {
    weights = nullptr;
}

Cnn::CConv3d::CConv3d(int in_channels, int out_channels, std::tuple<int, int, int> kernel_size, int stride, int padding, int dilation, bool bias) : C_ConvNd(in_channels, out_channels, stride, padding, dilation, bias), kernel_size{kernel_size} {
    this->weights = CTorch::randn(out_channels, in_channels, std::get<0>(kernel_size), std::get<1>(kernel_size), std::get<2>(kernel_size));
}
Cnn::CConv3d::~CConv3d() {
    weights = nullptr;
}
