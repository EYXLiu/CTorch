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
    activated = output->sigmoid(true);
    if (bias) {
        new_ct = new_ct->add(*biases);
        output = output->add(*biases);
    };
    return new_ct;
}

std::tuple<std::unique_ptr<CTorch::CTensor>, std::unique_ptr<CTorch::CTensor>> Cnn::CLinear::backgrad(std::unique_ptr<CTorch::CTensor>& grad, std::unique_ptr<CTorch::CTensor>& target) {
    std::unique_ptr<CTorch::CTensor> new_grad = std::move(grad);
    std::unique_ptr<CTorch::CTensor> prev;
    if ((*new_grad->shape().get())[1] == 0) {

        std::unique_ptr<CTorch::CTensor> output_grad = output->lossDerivative(target);

        std::unique_ptr<CTorch::CTensor> sigmoid_deriv = output->sigmoidDerivative();
        std::unique_ptr<CTorch::CTensor> delta = output_grad->matmul(*sigmoid_deriv);
        prev = delta->mul(1);
        new_grad->append(delta);
    } else {
        std::unique_ptr<CTorch::CTensor> delta = target->matmul(*weights->t())->hadamard(*output->sigmoidDerivative());
        prev = delta->mul(1);
        new_grad->append(delta);
    }
    return std::make_tuple(std::move(new_grad), std::move(prev));
}

std::unique_ptr<CTorch::CTensor> Cnn::CLinear::backpass(std::unique_ptr<CTorch::CTensor>& grad) {
    std::unique_ptr<CTorch::CTensor> new_grad = std::move(grad);
    std::unique_ptr<CTorch::CTensor> delta = new_grad->pop();

    std::unique_ptr<CTorch::CTensor> weights_grad = input->t()->matmul(*delta);
    weights = weights->add(*weights_grad->mul(learning_rate));

    if (bias) {
        std::unique_ptr<CTorch::CTensor> biases_grad = delta->sumLin();
        biases = biases->add(*biases_grad->mul(learning_rate));
    }

    return new_grad;

}