#include "ctorch.cnn.h"

Cnn::CModule::~CModule() {}

Cnn::CSequential::CSequential(std::vector<std::unique_ptr<CModule>>& modules) : modules{std::move(modules)} {}
Cnn::CSequential::~CSequential() {};
std::unique_ptr<CTorch::CTensor> Cnn::CSequential::forward(std::unique_ptr<CTorch::CTensor>& input) {
    std::unique_ptr<CTorch::CTensor> ct = std::move(input);
    for (auto& m : modules) {
        ct = m->forward(ct);
    }
    return ct;
};
std::unique_ptr<CTorch::CTensor> Cnn::CSequential::backward(std::unique_ptr<CTorch::CTensor>& grad) {
    std::unique_ptr<CTorch::CTensor> ct = std::move(grad);
    for (int i = modules.size() - 1; i >= 0; i--) {
        ct = modules[i]->backward(ct);
    }
    return ct;
};
std::unique_ptr<CTorch::CTensor> Cnn::CSequential::backgrad(std::unique_ptr<CTorch::CTensor>& grad) {
    std::unique_ptr<CTorch::CTensor> ct = std::move(grad);
    for (auto& m : modules) {
         ct = m->backgrad(ct);
    }
    return ct;
}
std::unique_ptr<CTorch::CTensor> Cnn::CSequential::backprop(std::unique_ptr<CTorch::CTensor>& grad) {
    std::unique_ptr<CTorch::CTensor> ct = std::move(grad);
    for (auto& m : modules) {
        ct = m->backprop(ct);
    }
    return ct;
}
void Cnn::CSequential::append(std::unique_ptr<Cnn::CModule> module) {
    modules.push_back(std::move(module));
};

Cnn::CModuleList::CModuleList(std::vector<std::unique_ptr<CModule>>& modules) : modules{std::move(modules)} {}
Cnn::CModuleList::~CModuleList() {};
std::unique_ptr<CTorch::CTensor> Cnn::CModuleList::forward(std::unique_ptr<CTorch::CTensor>& input) {
    std::unique_ptr<CTorch::CTensor> ct = std::move(input);
    for (auto& m : modules) {
        ct = m->forward(ct);
    }
    return ct;
};
std::unique_ptr<CTorch::CTensor> Cnn::CModuleList::backward(std::unique_ptr<CTorch::CTensor>& grad) {
    std::unique_ptr<CTorch::CTensor> ct = std::move(grad);
    for (int i = modules.size() - 1; i >= 0; i--) {
        ct = modules[i]->backward(ct);
    }
    return ct;
};
void Cnn::CModuleList::append(std::unique_ptr<Cnn::CModule> module) {
    modules.push_back(std::move(module));
};
void Cnn::CModuleList::extend(std::vector<std::unique_ptr<Cnn::CModule>> modules) {
    for (auto& m : modules) {
        this->modules.push_back(std::move(m));
    }
};
void Cnn::CModuleList::insert(int index, std::unique_ptr<Cnn::CModule> module) {
    modules.insert(modules.begin() + index, std::move(module));
};


Cnn::CModuleDict::CModuleDict(std::map<std::string, std::unique_ptr<Cnn::CModule>>& modules) : modules{std::move(modules)} {}
Cnn::CModuleDict::~CModuleDict() {};
std::unique_ptr<CTorch::CTensor> Cnn::CModuleDict::forward(std::unique_ptr<CTorch::CTensor>& input) {
    std::unique_ptr<CTorch::CTensor> ct = std::move(input);
    for (auto& m : modules) {
        ct = m.second->forward(ct);
    }
    return ct;
};
std::unique_ptr<CTorch::CTensor> Cnn::CModuleDict::backward(std::unique_ptr<CTorch::CTensor>& grad) {
    std::unique_ptr<CTorch::CTensor> ct = std::move(grad);
    for (auto& m : modules) {
        ct = m.second->backward(ct);
    }
    return ct;
};
void Cnn::CModuleDict::clear() {
    modules.clear();
};
std::unique_ptr<Cnn::CModule> Cnn::CModuleDict::pop(std::string key) {
    std::unique_ptr<Cnn::CModule> m = std::move(modules[key]);
    modules.erase(key);
    return m;
};
void Cnn::CModuleDict::update(std::map<std::string, std::unique_ptr<Cnn::CModule>> other) {
    for (auto& m : other) {
        modules[m.first] = std::move(m.second);
    }
};
