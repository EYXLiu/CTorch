#ifndef CTORCH_CNN_H
#define CTORCH_CNN_H

#include "ctorch.h"
#include <iostream>
#include <any>
#include <cstdint>
#include <vector>
#include <map>
#include <tuple>

class Cnn {
    public:
        class CModule {
            public:
                virtual ~CModule();
                virtual std::unique_ptr<CTorch::CTensor> forward(std::unique_ptr<CTorch::CTensor>& input) = 0;
                virtual std::unique_ptr<CTorch::CTensor> backward(std::unique_ptr<CTorch::CTensor>& grad) = 0;
        };
        class CSequential : public CModule {
            std::vector<std::unique_ptr<CModule>> modules;
            public:
                CSequential(std::vector<std::unique_ptr<CModule>>& modules);
                ~CSequential();
                std::unique_ptr<CTorch::CTensor> forward(std::unique_ptr<CTorch::CTensor>& input);
                std::unique_ptr<CTorch::CTensor> backward(std::unique_ptr<CTorch::CTensor>& grad);
                void append(std::unique_ptr<CModule> module);
        };
        class CModuleList : public CModule {
            std::vector<std::unique_ptr<CModule>> modules;
            public:
                CModuleList(std::vector<std::unique_ptr<CModule>>& modules);
                ~CModuleList();
                std::unique_ptr<CTorch::CTensor> forward(std::unique_ptr<CTorch::CTensor>& input);
                std::unique_ptr<CTorch::CTensor> backward(std::unique_ptr<CTorch::CTensor>& grad);
                void append(std::unique_ptr<CModule> module);
                void extend(std::vector<std::unique_ptr<CModule>> modules);
                void insert(int index, std::unique_ptr<CModule> module);
        };
        class CModuleDict : public CModule {
            std::map<std::string, std::unique_ptr<CModule>> modules;
            public:
                CModuleDict(std::map<std::string, std::unique_ptr<CModule>>& modules);
                ~CModuleDict();
                std::unique_ptr<CTorch::CTensor> forward(std::unique_ptr<CTorch::CTensor>& input);
                std::unique_ptr<CTorch::CTensor> backward(std::unique_ptr<CTorch::CTensor>& grad);
                void clear();
                std::unique_ptr<CModule> pop(std::string key);
                void update(std::map<std::string, std::unique_ptr<CModule>> modules);
        };
        class C_ConvNd : public CModule {
            protected:
                std::unique_ptr<CTorch::CTensor> weights;
                std::unique_ptr<CTorch::CTensor> biases;
                std::unique_ptr<CTorch::CTensor> input;
                int in_channels;
                int out_channels;
                int stride; 
                int padding;
                int dilation; 
                bool bias;
            public: 
                C_ConvNd(int in_channels, int out_channels, int stride, int padding, int dilation, bool bias);
                virtual ~C_ConvNd();
        };
        class CConv1d : public C_ConvNd {
            int kernel_size;
            public:
                CConv1d(int in_channels, int out_channels, int kernel_size, int stride=1, int padding=0, int dilation=1, bool bias=true);
                ~CConv1d();
                std::unique_ptr<CTorch::CTensor> forward(std::unique_ptr<CTorch::CTensor>& input);
                std::unique_ptr<CTorch::CTensor> backward(std::unique_ptr<CTorch::CTensor>& grad);
        };
        class CConv2d : public C_ConvNd {
            std::tuple<int, int> kernel_size;
            public:
                CConv2d(int in_channels, int out_channels, std::tuple<int, int> kernel_size, int stride=1, int padding=0, int dilation=1, bool bias=true);
                ~CConv2d();
                std::unique_ptr<CTorch::CTensor> forward(std::unique_ptr<CTorch::CTensor>& input);
                std::unique_ptr<CTorch::CTensor> backward(std::unique_ptr<CTorch::CTensor>& grad);
        };
        class CConv3d : public C_ConvNd {
            std::tuple<int, int, int> kernel_size;
            public:
                CConv3d(int in_channels, int out_channels, std::tuple<int, int, int> kernel_size, int stride=1, int padding=0, int dilation=1, bool bias=true);
                ~CConv3d();
                std::unique_ptr<CTorch::CTensor> forward(std::unique_ptr<CTorch::CTensor>& input);
                std::unique_ptr<CTorch::CTensor> backward(std::unique_ptr<CTorch::CTensor>& grad);
        };
        class CLinear : public CModule {
            std::unique_ptr<CTorch::CTensor> weights;
            std::unique_ptr<CTorch::CTensor> biases;
            std::unique_ptr<CTorch::CTensor> input;
            std::unique_ptr<CTorch::CTensor> output;
            int in_features;
            int out_features;
            bool bias;
            float learning_rate;
            public:
                CLinear(int in_features, int out_features, bool bias=true);
                ~CLinear();
                std::unique_ptr<CTorch::CTensor> forward(std::unique_ptr<CTorch::CTensor>& input);
                std::unique_ptr<CTorch::CTensor> backward(std::unique_ptr<CTorch::CTensor>& grad);
        };

};

#endif // CTORCH_CNN_H