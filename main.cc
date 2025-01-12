#include "ctorch.h"
#include "ctorch.cnn.h"
#include "ctorch.cnn.functional.h"
#include <iostream>

int main() {
    CTorch torch;
    std::unique_ptr<CTorch::CTensor> t = torch.randn(3, 3);
    std::cout << *t << std::endl;
    std::cout << *t->shape() << std::endl;
    std::cout << t->size() << std::endl;
}