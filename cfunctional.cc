#include "ctorch.cnn.functional.h"
#include <cmath>
#include <cassert>

void ReLU(CTorch::CTensor*& ct) {
    assert(ct->size() == 1 && "Expected input of size 1");
    std::vector<std::any> new_arr;
    int type = ct->getType();
    std::vector<std::any> arr = std::any_cast<std::vector<std::any>>(ct->getValue());
    for (auto& a : arr) {
        if (type == CTorch::Int32) {
            new_arr.push_back(std::max(std::any_cast<std::int32_t>(a), 0));
        } else if (type == CTorch::Int64) {
            new_arr.push_back(std::max(std::any_cast<std::int64_t>(a), static_cast<std::int64_t>(0)));
        } else if (type == CTorch::Float32) {
            new_arr.push_back(std::max(std::any_cast<float>(a), 0.0f));
        } else if (type == CTorch::Float64) {
            new_arr.push_back(std::max(std::any_cast<double>(a), 0.0));
        } else {
            throw std::logic_error("invalid data type");
        }
    }
    delete ct;
    ct = new CTorch::CTensor(new_arr, static_cast<CTorch::ScalarType>(type));
}

void exp(CTorch::CTensor*& ct) {
    assert(ct->size() == 1 && "Expected input of size 1");
    std::vector<std::any> new_arr;
    int type = ct->getType();
    std::vector<std::any> arr = std::any_cast<std::vector<std::any>>(ct->getValue());
    for (auto& a : arr) {
        if (type == CTorch::Int32) {
            new_arr.push_back(static_cast<std::int32_t>(std::exp(std::any_cast<std::int32_t>(a))));
        } else if (type == CTorch::Int64) {
            new_arr.push_back(static_cast<std::int64_t>(std::exp(std::any_cast<std::int64_t>(a))));
        } else if (type == CTorch::Float32) {
            new_arr.push_back(static_cast<float>(std::exp(std::any_cast<float>(a))));
        } else if (type == CTorch::Float64) {
            new_arr.push_back(static_cast<double>(std::exp(std::any_cast<double>(a))));
        } else {
            throw std::logic_error("invalid data type");
        }
    }
    delete ct; 
    ct = new CTorch::CTensor(new_arr, static_cast<CTorch::ScalarType>(type));
}

void softmax(CTorch::CTensor*& ct) {
    assert(ct->size() == 1 && "Expected input of size 1");
    exp(ct);
    std::vector<std::any> new_arr;
    int type = ct->getType();
    std::vector<std::any> arr = std::any_cast<std::vector<std::any>>(ct->getValue());
    float total = 0;
    for (auto& a : arr) {
        if (type == CTorch::Int32) {
            total += static_cast<float>(std::any_cast<std::int32_t>(a));
        } else if (type == CTorch::Int64) {
            total += static_cast<float>(std::any_cast<std::int64_t>(a));
        } else if (type == CTorch::Float32) {
            total += std::any_cast<float>(a);
        } else if (type == CTorch::Float64) {
            total += static_cast<float>(std::any_cast<double>(a));
        } else {
            throw std::logic_error("invalid data type");
        }
    }
    for (auto& a : arr) {
        if (type == CTorch::Int32) {
            new_arr.push_back(static_cast<std::int32_t>(std::round(std::any_cast<std::int32_t>(a) / total)));
        } else if (type == CTorch::Int64) {
            new_arr.push_back(static_cast<std::int64_t>(std::round(std::any_cast<std::int64_t>(a) / total)));
        } else if (type == CTorch::Float32) {
            new_arr.push_back(std::any_cast<float>(a) / total);
        } else if (type == CTorch::Float64) {
            new_arr.push_back(static_cast<double>(std::round(std::any_cast<double>(a) / total)));
        }
    }
    delete ct;
    ct = new CTorch::CTensor(new_arr, static_cast<CTorch::ScalarType>(type));
}

void sigmoid(CTorch::CTensor*& ct) {
    assert(ct->size() == 1 && "Expected input of size 1");
    std::vector<std::any> new_arr;
    int type = ct->getType();
    std::vector<std::any> arr = std::any_cast<std::vector<std::any>>(ct->getValue());
    for (auto& a : arr) {
        if (type == CTorch::Int32) {
            new_arr.push_back(static_cast<std::int32_t>(1 / (1 + std::exp(std::any_cast<std::int32_t>(a)))));
        } else if (type == CTorch::Int64) {
            new_arr.push_back(static_cast<std::int64_t>(1 / (1 + std::exp(std::any_cast<std::int64_t>(a)))));
        } else if (type == CTorch::Float32) {
            new_arr.push_back(static_cast<float>(1 / (1 + std::exp(std::any_cast<float>(a)))));
        } else if (type == CTorch::Float64) {
            new_arr.push_back(static_cast<double>(1 / (1 + std::exp(std::any_cast<double>(a)))));
        } else {
            throw std::logic_error("invalid data type");
        }
    }
    delete ct; 
    ct = new CTorch::CTensor(new_arr, static_cast<CTorch::ScalarType>(type));
}