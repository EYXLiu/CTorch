#include "ctorch.cnn.h"
#include <cassert>

Cnn::CLinear::CLinear(int in_features, int out_features, bool bias): in_features{in_features}, out_features{out_features}, bias{bias} {
    this->weights = CTorch::randn(out_features, in_features);
    if (bias) {
        this->biases = CTorch::zeros(out_features);
    };
    this->learning_rate = 0.001f;
}
Cnn::CLinear::~CLinear() {
    biases = nullptr;
    weights = nullptr;
    input = nullptr;
    output = nullptr;
}

std::unique_ptr<CTorch::CTensor> Cnn::CLinear::forward(std::unique_ptr<CTorch::CTensor>& ct) {
    assert(ct->size() == 1 && "Expects input of size 1");
    input = std::move(ct);
    std::unique_ptr<CTorch::CTensor> new_ct = input->matmul(*weights->t());
    output = input->matmul(*weights->t());
    if (bias) {
        new_ct = new_ct->add(*biases);
        output = output->add(*biases);
    };
    return new_ct;
}
std::unique_ptr<CTorch::CTensor> Cnn::CLinear::backward(std::unique_ptr<CTorch::CTensor>& ct) {
    assert(ct->size() == 1 && "Expects error of size 1");

    std::unique_ptr<CTorch::CTensor> sig_ct = ct->sigmoid(true);
    std::unique_ptr<CTorch::CTensor> delta = input->t()->matmul(*ct);

    std::unique_ptr<CTorch::CTensor> weight_gradient = input->t()->matmul(*delta);
    weights = weights->add(*weight_gradient->mul(learning_rate));

    if (bias) {
        std::unique_ptr<CTorch::CTensor> bias_gradient = delta->sumLin();
        biases = biases->add(*bias_gradient->mul(learning_rate));
    }

    return delta->matmul(*weights->t());
}