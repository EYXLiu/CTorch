#include "ctorch.h"
#include "ctorch.cnn.h"
#include "ctorch.cnn.functional.h"
#include <iostream>

int main() {
    CTorch torch;
    std::unique_ptr<CTorch::CTensor> t = torch.zeros(1, 0);
    std::unique_ptr<CTorch::CTensor> l = std::make_unique<CTorch::CTensor>(std::vector<std::any>{std::vector<std::any>{}}, CTorch::Float32);
    std::cout << *l << std::endl;
    std::cout << *l->shape() << std::endl;
    std::cout << l->size() << std::endl;
}