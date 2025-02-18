#include "ctorch.h"
#include <tuple>
#include <ctime> 
#include <cassert>
#include <utility>

CTorch::CTensor::CTensor(std::vector<std::any> arr, CTorch::ScalarType dtype) : dtype{dtype} {
    try {
        if (arr.size() == 0) {
            return;
        }
        for (auto& a : arr) {
            try {
                this->arr.push_back(std::make_unique<CTorch::CTensor>(std::any_cast<std::vector<std::any>>(a), dtype));
            } catch (const std::bad_any_cast &e) {
                if (dtype == CTorch::Int32) {
                    this->arr.push_back(std::make_unique<CTorch::CInt32>(std::any_cast<std::int32_t>(a)));
                } else if (dtype == CTorch::Int64) {
                    this->arr.push_back(std::make_unique<CTorch::CInt64>(std::any_cast<std::int64_t>(a)));
                } else if (dtype == CTorch::Float32) {
                    try {
                        this->arr.push_back(std::make_unique<CTorch::CFloat32>(std::any_cast<float>(a)));
                    } catch (const std::bad_any_cast &e) {
                        try {
                            this->arr.push_back(std::make_unique<CTorch::CFloat32>(static_cast<float>(std::any_cast<int32_t>(a))));
                        } catch (const std::bad_any_cast &e) {
                            throw std::logic_error("invalid input type");
                        }
                    }
                } else if (dtype == CTorch::Float64) {
                    this->arr.push_back(std::make_unique<CTorch::CFloat64>(std::any_cast<double>(a)));
                } else {
                    throw std::logic_error("invalid data type");
                }
            }
        }
    } catch (const std::logic_error &e) {
        std::cerr << "ValueError: " << e.what() << std::endl;
    } catch (const std::bad_any_cast &e) {
        std::cerr << "Bad Cast: " <<  e.what() << std::endl;
    }
};

CTorch::CTensor::CTensor(const CTensor& other) : dtype{other.dtype} {
    for (auto& a : other.arr) {
        if (dtype == CTorch::Int32) {
            arr.push_back(std::make_unique<CInt32>(std::any_cast<std::int32_t>(a->getValue())));
        } else if (dtype == CTorch::Int64) {
            arr.push_back(std::make_unique<CInt64>(std::any_cast<std::int64_t>(a->getValue())));
        } else if (dtype == CTorch::Float32) {
            arr.push_back(std::make_unique<CFloat32>(std::any_cast<float>(a->getValue())));
        } else if (dtype == CTorch::Float64) {
            arr.push_back(std::make_unique<CFloat64>(std::any_cast<double>(a->getValue())));
        } else {
            throw std::logic_error("invalid data type");
        }
    }
}

CTorch::CTensor::~CTensor() {}

int CTorch::CTensor::getType() const {
    return dtype;
}

std::unique_ptr<CTorch::CTensor> CTorch::CTensor::add(const CTorch::CTensor& other) const {
    try {
        if (this->dtype != other.dtype) {
            throw std::logic_error("expected " + std::to_string(this->dtype) + " (got " + std::to_string(other.dtype) + ")");
        }
        if (this->arr.size() != other.arr.size()) {
            throw std::logic_error("expected sequence of length " + std::to_string(this->arr.size()));
        }
        std::vector<std::any> new_arr;
        for (int i = 0; i < this->arr.size(); i++) {
            auto* ct = dynamic_cast<CTorch::CTensor*>(this->arr[i].get());
            if (ct != nullptr) {
                auto* other_ct = dynamic_cast<CTorch::CTensor*>(other.arr[i].get());
                if (other_ct == nullptr || ct->arr.size() != other_ct->arr.size()) {
                    throw std::logic_error("type mismatch, expected matrix received scalar");
                }
                new_arr.push_back((ct->add(*other_ct))->getValue());
            } else if (this->dtype == CTorch::Int32) {
                new_arr.push_back(std::any_cast<std::int32_t>(this->arr[i].get()->getValue()) + std::any_cast<std::int32_t>(other.arr[i].get()->getValue()));
            } else if (this->dtype == CTorch::Int64) {
                new_arr.push_back(std::any_cast<std::int64_t>(this->arr[i].get()->getValue()) + std::any_cast<std::int64_t>(other.arr[i].get()->getValue()));
            } else if (this->dtype == CTorch::Float32) {
                new_arr.push_back(std::any_cast<float>(this->arr[i].get()->getValue()) + std::any_cast<float>(other.arr[i].get()->getValue()));
            } else if (this->dtype == CTorch::Float64) {
                new_arr.push_back(std::any_cast<double>(this->arr[i].get()->getValue()) + std::any_cast<double>(other.arr[i].get()->getValue()));
            } else {
                throw std::logic_error("invalid data type");
            }
        }
        return std::make_unique<CTorch::CTensor>(new_arr, this->dtype);
    } catch (const std::logic_error &e) {
        std::cerr << "ValueError: " << e.what() << std::endl;
        return std::make_unique<CTorch::CTensor>(std::vector<std::any>{}, this->dtype);
    } catch (const std::bad_variant_access &e) {
        std::cerr << "MismatchedType: " <<  e.what() << std::endl;
        return std::make_unique<CTorch::CTensor>(std::vector<std::any>{}, this->dtype);
    }
}

std::unique_ptr<CTorch::CTensor> CTorch::CTensor::hadamard(const CTorch::CTensor& other) const {
    try {
        if (this->dtype != other.dtype) {
            throw std::logic_error("expected " + std::to_string(this->dtype) + " (got " + std::to_string(other.dtype) + ")");
        }
        if (this->arr.size() != other.arr.size()) {
            throw std::logic_error("expected sequence of length " + std::to_string(this->arr.size()));
        }
        std::vector<std::any> new_arr;
        for (int i = 0; i < this->arr.size(); i++) {
            auto* ct = dynamic_cast<CTorch::CTensor*>(this->arr[i].get());
            if (ct != nullptr) {
                auto* other_ct = dynamic_cast<CTorch::CTensor*>(other.arr[i].get());
                if (other_ct == nullptr || ct->arr.size() != other_ct->arr.size()) {
                    throw std::logic_error("type mismatch, expected matrix received scalar");
                }
                new_arr.push_back((ct->hadamard(*other_ct))->getValue());
            } else if (this->dtype == CTorch::Int32) {
                new_arr.push_back(std::any_cast<std::int32_t>(this->arr[i]->getValue()) * std::any_cast<std::int32_t>(other.arr[i]->getValue()));
            } else if (this->dtype == CTorch::Int64) {
                new_arr.push_back(std::any_cast<std::int64_t>(this->arr[i]->getValue()) * std::any_cast<std::int64_t>(other.arr[i]->getValue()));
            } else if (this->dtype == CTorch::Float32) {
                new_arr.push_back(std::any_cast<float>(this->arr[i]->getValue()) * std::any_cast<float>(other.arr[i]->getValue()));
            } else if (this->dtype == CTorch::Float64) {
                new_arr.push_back(std::any_cast<double>(this->arr[i]->getValue()) * std::any_cast<double>(other.arr[i]->getValue()));
            } else {
                throw std::logic_error("invalid data type");
            }
        }
        return std::make_unique<CTorch::CTensor>(new_arr, this->dtype);
    } catch (const std::logic_error &e) {
        std::cerr << "ValueError: " << e.what() << std::endl;
        return std::make_unique<CTorch::CTensor>(std::vector<std::any>{}, this->dtype);
    } catch (const std::bad_variant_access &e) {
        std::cerr << "MismatchedType: " <<  e.what() << std::endl;
        return std::make_unique<CTorch::CTensor>(std::vector<std::any>{}, this->dtype);
    }
}

std::unique_ptr<CTorch::DType> CTorch::CTensor::dot(const CTorch::CTensor& other) const {
    try {
        if (auto* other_ct = dynamic_cast<CTorch::CTensor*>(other.arr[0].get())) {
            return this->matmul(*other_ct);
        }
        if (dtype == CTorch::Int32) {
            std::int32_t val = 0;
            for (int i = 0; i < arr.size(); i++) {
                val += (std::any_cast<std::int32_t>(arr[i].get()->getValue()) * std::any_cast<std::int32_t>(other.arr[i].get()->getValue()));
            }
            return std::make_unique<CTorch::CFloat32>(val);
        } else if (dtype == CTorch::Int64) {
            std::int64_t val = 0;
            for (int i = 0; i < arr.size(); i++) {
                val += (std::any_cast<std::int64_t>(arr[i].get()->getValue()) * std::any_cast<std::int64_t>(other.arr[i].get()->getValue()));
            }
            return std::make_unique<CTorch::CFloat32>(val);
        } else if (dtype == CTorch::Float32) {
            float val = 0;
            for (int i = 0; i < arr.size(); i++) {
                val += (std::any_cast<float>(arr[i].get()->getValue()) * std::any_cast<float>(other.arr[i].get()->getValue()));
            }
            return std::make_unique<CTorch::CFloat32>(val);
        } else if (dtype == CTorch::Float64) {
            double val = 0;
            for (int i = 0; i < arr.size(); i++) {
                val += (std::any_cast<double>(arr[i].get()->getValue()) * std::any_cast<double>(other.arr[i].get()->getValue()));
            }
            return std::make_unique<CTorch::CFloat32>(val);
        } else {
            throw std::logic_error("invalid data type");
        }
    } catch (const std::logic_error &e) {
        std::cerr << "ValueError: " << e.what() << std::endl;
        return std::make_unique<CTorch::CFloat32>(0);
    } catch (const std::bad_variant_access &e) {
        std::cerr << "MismatchedType: " <<  e.what() << std::endl;
        return std::make_unique<CTorch::CFloat32>(0);
    } catch (const std::bad_any_cast &e) {
        std::cerr << "Bad Cast: " <<  e.what() << std::endl;
        return std::make_unique<CTorch::CFloat32>(0);
    }
}

std::unique_ptr<CTorch::CTensor> CTorch::CTensor::matmul(const CTorch::CTensor& other) const {
    try {
        if (this->dtype != other.dtype) {
            throw std::logic_error("expected " + std::to_string(this->dtype) + " (got " + std::to_string(other.dtype) + ")");
        }
        assert (arr.size() > 0 && other.arr.size() > 0 && "expected non-empty sequence");
        auto* ct = dynamic_cast<CTorch::CTensor*>(arr[0].get());
        auto* other_ct = dynamic_cast<CTorch::CTensor*>(other.arr[0].get());
        std::vector<std::any> new_arr;
        if (ct != nullptr) {
            if (ct->arr.size() != other.size() && other.size() != 1) {
                throw std::logic_error("expected sequence of length " + std::to_string(ct->arr.size()));
            }
            if (other_ct != nullptr) {
                for (int i = 0; i < arr.size(); i++) {
                    new_arr.push_back((dynamic_cast<CTorch::CTensor*>(arr[i].get())->matmul(*dynamic_cast<CTorch::CTensor*>(other.arr[other.arr.size() == 1 ? 0 : i].get())))->getValue());
                }
            } else {
                if (arr.size() == 1) {
                    return dynamic_cast<CTorch::CTensor*>(arr[0].get())->matmul(other);
                }
                for (int i = 0; i < arr.size(); i++) {
                    new_arr.push_back((dynamic_cast<CTorch::CTensor*>(arr[i].get())->matmul(other))->getValue());
                }
            }
        } else if (other_ct == nullptr) {
            for (int i = 0; i < arr.size(); i++) {
                new_arr.push_back(other.mul(std::any_cast<float>(arr[i]->getValue()))->getValue());
            }
        } else {
            for (int i = 0; i < (*other_ct->shape())[0]; i++) {
                std::vector<std::any> other_arr;
                for (int j = 0; j < arr.size(); j++) {
                    other_arr.push_back(std::any_cast<std::vector<std::any>>(dynamic_cast<CTorch::CTensor*>(other.arr[j].get())->getValue())[i]);
                }
                CTorch::CTensor* s = new CTorch::CTensor(other_arr);
                new_arr.push_back((this->dot(*s))->getValue());
                delete s;
            }
        }
        return std::make_unique<CTorch::CTensor>(new_arr, this->dtype);
    } catch (const std::logic_error &e) {
        std::cerr << "ValueError: " << e.what() << std::endl;
        return std::make_unique<CTorch::CTensor>(std::vector<std::any>{}, this->dtype);
    } catch (const std::bad_variant_access &e) {
        std::cerr << "MismatchedType:  " <<  e.what() << std::endl;
        return std::make_unique<CTorch::CTensor>(std::vector<std::any>{}, this->dtype);
    }
}

std::unique_ptr<CTorch::CTensor> CTorch::CTensor::mul(float i) const {
    try {
        auto* ct = dynamic_cast<CTorch::CTensor*>(arr[0].get());
        std::vector<std::any> new_arr;
        if (ct == nullptr) {
            for (int j = 0; j < arr.size(); j++) {
                if (dtype == CTorch::Float64) {
                    new_arr.push_back(static_cast<float>(i) * std::any_cast<float>(arr[j]->getValue()));
                } else {
                    new_arr.push_back(i * std::any_cast<float>(arr[j].get()->getValue()));
                }
            }
        } else {
            for (int j = 0; j < arr.size(); j++) {
                new_arr.push_back(std::any_cast<CTorch::CTensor*>(arr[j].get())->mul(i)->getValue());
            }
        }
        return std::make_unique<CTorch::CTensor>(new_arr, this->dtype);
    } catch (const std::logic_error &e) {
        std::cerr << "ValueError: " << e.what() << std::endl;
        return std::make_unique<CTorch::CTensor>(std::vector<std::any>{}, this->dtype);
    } catch (const std::bad_variant_access &e) {
        std::cerr << "MismatchedType:  " <<  e.what() << std::endl;
        return std::make_unique<CTorch::CTensor>(std::vector<std::any>{}, this->dtype);
    }
}

std::vector<std::any> CTorch::CTensor::unsqueezeHelp(int dim, std::vector<std::any> arr) const {
    std::vector<std::any> new_arr;
    if (dim == 0) {
        for (auto& a : arr) {
            new_arr.push_back(std::vector<std::any>(1, a));
        }
    } else {
        for (auto& a : arr) {
            new_arr.push_back(unsqueezeHelp(dim - 1, std::any_cast<std::vector<std::any>>(a)));
        }
    }
    return new_arr;
}

std::unique_ptr<CTorch::CTensor> CTorch::CTensor::unsqueeze(int dim) const {
    try {
        if (dim < 0) {
            throw std::logic_error("index " + std::to_string(dim) + " is out of bounds");
        }
        if (dim == 0) {
            return std::make_unique<CTorch::CTensor>(std::vector<std::any>(1, getValue()), dtype);
        }
        std::vector<std::any> new_arr = std::any_cast<std::vector<std::any>>(getValue());
        return std::make_unique<CTorch::CTensor>(unsqueezeHelp(dim - 1, new_arr), dtype);
    } catch (const std::logic_error &e) {
        std::cerr << "IndexError: " << e.what() << std::endl;
        return std::make_unique<CTorch::CTensor>(std::vector<std::any>{}, dtype);
    } catch (const std::bad_any_cast &e) {
        std::cerr << "IndexError: index " << dim << " is out of bounds" << std::endl;
        return std::make_unique<CTorch::CTensor>(std::vector<std::any>{}, dtype);
    }
}

std::unique_ptr<CTorch::CTensor> CTorch::CTensor::operator[](std::size_t dim) const {
    try {
        if (dim < 0 || dim >= arr.size()) {
            throw std::logic_error("index " + std::to_string(dim) + " is out of bounds");
        }
        std::unique_ptr<CTorch::CTensor> ct = std::unique_ptr<CTorch::CTensor>(dynamic_cast<CTorch::CTensor*>(arr[dim].get()));
        if (ct != nullptr) { return ct; };
        if (dtype == CTorch::Int32) {
            return std::make_unique<CTorch::CTensor>(std::vector<std::any>{std::any_cast<std::int32_t>(arr[dim].get()->getValue())}, dtype);
        } else if (dtype == CTorch::Int64) {
            return std::make_unique<CTorch::CTensor>(std::vector<std::any>{std::any_cast<std::int64_t>(arr[dim].get()->getValue())}, dtype);
        } else if (dtype == CTorch::Float32) {
            return std::make_unique<CTorch::CTensor>(std::vector<std::any>{std::any_cast<float>(arr[dim].get()->getValue())}, dtype);
        } else if (dtype == CTorch::Float64) {
            return std::make_unique<CTorch::CTensor>(std::vector<std::any>{std::any_cast<double>(arr[dim].get()->getValue())}, dtype);
        } else {
            return std::make_unique<CTorch::CTensor>(std::vector<std::any>{}, dtype);
            throw std::logic_error("invalid data type");
        }
    } catch (const std::logic_error &e) {
        std::cerr << "IndexError: " << e.what() << std::endl;
        return std::make_unique<CTorch::CTensor>(std::vector<std::any>{}, dtype);
    } catch (const std::bad_any_cast &e) {
        std::cerr << "Bad Cast: " <<  e.what() << std::endl;
        return std::make_unique<CTorch::CTensor>(std::vector<std::any>{}, dtype);
    }
}

std::unique_ptr<CTorch::CTensor> CTorch::CTensor::sigmoid(bool deriv) const {
    assert(size() == 1 && "Expected size of 1");
    std::vector<std::any> temp;
    for (int i = 0; i < arr.size(); i++) {
        if (dtype == CTorch::Int32) {
            if (deriv) {
                temp.push_back(std::any_cast<std::int32_t>(arr[i]->getValue()) * (1 - std::any_cast<std::int32_t>(arr[i]->getValue())));
            } else {
                temp.push_back(1 / (1 + std::exp(std::any_cast<std::int32_t>(arr[i]->getValue()))));
            }
        } else if (dtype == CTorch::Int64) {
            if (deriv) {
                temp.push_back(std::any_cast<std::int64_t>(arr[i]->getValue()) * (1 - std::any_cast<std::int64_t>(arr[i]->getValue())));
            } else {
                temp.push_back(1 / (1 + std::exp(std::any_cast<std::int64_t>(arr[i]->getValue()))));
            }
        } else if (dtype == CTorch::Float32) {
            if (deriv) {
                temp.push_back(std::any_cast<float>(arr[i]->getValue()) * (1 - std::any_cast<float>(arr[i]->getValue())));
            } else {
                temp.push_back(1 / (1 + std::exp(std::any_cast<float>(arr[i]->getValue()))));
            }
        } else if (dtype == CTorch::Float64) {
            if (deriv) {
                temp.push_back(std::any_cast<double>(arr[i]->getValue()) * (1 - std::any_cast<double>(arr[i]->getValue())));
            } else {
                temp.push_back(1 / (1 + std::exp(std::any_cast<double>(arr[i]->getValue()))));
            }
        } else {
            throw std::logic_error("invalid data type");
        }
    }
    return std::make_unique<CTorch::CTensor>(temp, dtype);
}

std::unique_ptr<CTorch::CTensor> CTorch::CTensor::sigmoidDerivative() const {
    std::unique_ptr<CTorch::CTensor> derivative = this->sigmoid(true);
    std::vector<std::any> temp;
    for (int i = 0; i < arr.size(); i++) {
        if (dtype == CTorch::Int32) {
            temp.push_back(std::any_cast<std::int32_t>(std::any_cast<std::vector<std::any>>((*derivative)[i]->getValue())[0]) * (1 - std::any_cast<std::int32_t>(std::any_cast<std::vector<std::any>>((*derivative)[i]->getValue())[0])));
        } else if (dtype == CTorch::Int64) {
            temp.push_back(std::any_cast<std::int64_t>(std::any_cast<std::vector<std::any>>((*derivative)[i]->getValue())[0]) * (1 - std::any_cast<std::int64_t>(std::any_cast<std::vector<std::any>>((*derivative)[i]->getValue())[0])));
        } else if (dtype == CTorch::Float32) {
            temp.push_back(std::any_cast<float>(std::any_cast<std::vector<std::any>>((*derivative)[i]->getValue())[0]) * (1 - std::any_cast<float>(std::any_cast<std::vector<std::any>>((*derivative)[i]->getValue())[0])));
        } else if (dtype == CTorch::Float64) {
            temp.push_back(std::any_cast<double>(std::any_cast<double>(std::any_cast<std::vector<std::any>>((*derivative)[i]->getValue())[0])) * (1 - std::any_cast<double>(std::any_cast<std::vector<std::any>>((*derivative)[i]->getValue())[0])));
        } else {
            throw std::logic_error("invalid data type");
        }
    }
    return std::make_unique<CTorch::CTensor>(temp, dtype);
}

std::unique_ptr<CTorch::CTensor> CTorch::CTensor::loss(std::unique_ptr<CTorch::CTensor>& target) {
    std::vector<std::any> temp;
    for (int i = 0; i < arr.size(); i++) {
        if (dtype == CTorch::Int32) {
            temp.push_back(std::pow(std::any_cast<std::int32_t>(arr[i]->getValue()) - std::any_cast<std::int32_t>(target->arr[i]->getValue()), 2));
        } else if (dtype == CTorch::Int64) {
            temp.push_back(std::pow(std::any_cast<std::int64_t>(arr[i]->getValue()) - std::any_cast<std::int64_t>(target->arr[i]->getValue()), 2));
        } else if (dtype == CTorch::Float32) {
            temp.push_back(std::pow(std::any_cast<float>(arr[i]->getValue()) - std::any_cast<float>(target->arr[i]->getValue()), 2));
        } else if (dtype == CTorch::Float64) {
            temp.push_back(std::pow(std::any_cast<double>(arr[i]->getValue()) - std::any_cast<double>(target->arr[i]->getValue()), 2));
        } else {
            throw std::logic_error("invalid data type");
        }
    }
    return std::make_unique<CTorch::CTensor>(temp, dtype);
}

std::unique_ptr<CTorch::CTensor> CTorch::CTensor::lossDerivative(std::unique_ptr<CTorch::CTensor>& target) {
    assert(arr.size() == target->arr.size());
    std::vector<std::any> temp;
    for (int i = 0; i < arr.size(); i++) {
        if (dtype == CTorch::Int32) {
            temp.push_back(2 * (std::any_cast<std::int32_t>(arr[i]->getValue()) - std::any_cast<std::int32_t>(target->arr[i]->getValue())));
        } else if (dtype == CTorch::Int64) {
            temp.push_back(2 * (std::any_cast<std::int64_t>(arr[i]->getValue()) - std::any_cast<std::int64_t>(target->arr[i]->getValue())));
        } else if (dtype == CTorch::Float32) {
            temp.push_back(2 * (std::any_cast<float>(arr[i]->getValue()) - std::any_cast<float>(target->arr[i]->getValue())));
        } else if (dtype == CTorch::Float64) {
            temp.push_back(2 * (std::any_cast<double>(arr[i]->getValue()) - std::any_cast<double>(target->arr[i]->getValue())));
        } else {
            throw std::logic_error("invalid data type");
        }
    }
    return std::make_unique<CTorch::CTensor>(temp, dtype);
}

std::unique_ptr<CTorch::CTensor> CTorch::CTensor::sumLin() const {
    assert(size() == 2 && "Expected size of 2");
    std::vector<std::vector<std::any>> temp;
    for (int i = 0; i < arr.size(); i++) {
        temp.push_back(std::any_cast<std::vector<std::any>>(arr[i].get()->getValue()));
    }
    std::vector<std::any> returned;
    for (int i = 0; i < temp[0].size(); i++) {
        if (dtype == CTorch::Int32) {
            std::int32_t x = 0;
            for (int j = 0; j < arr.size(); j++) {
                x += std::any_cast<std::int32_t>(temp[i][j]);
            }
            returned.push_back(x);
        } else if (dtype == CTorch::Int32) {
            std::int64_t x = 0;
            for (int j = 0; j < arr.size(); j++) {
                x += std::any_cast<std::int64_t>(temp[i][j]);
            }
            returned.push_back(x);
        } else if (dtype == CTorch::Int32) {
            float x = 0;
            for (int j = 0; j < arr.size(); j++) {
                x += std::any_cast<float>(temp[i][j]);
            }
            returned.push_back(x);
        } else if (dtype == CTorch::Int32) {
            double x = 0;
            for (int j = 0; j < arr.size(); j++) {
                x += std::any_cast<double>(temp[i][j]);
            }
            returned.push_back(x);
        } else {
            throw std::logic_error("invalid data type");
        }
    }
    return std::make_unique<CTorch::CTensor>(returned, dtype).get()->t();
}

void CTorch::CTensor::append(std::unique_ptr<CTorch::CTensor>& ct) {
    arr.push_back(std::move(ct));
}

std::unique_ptr<CTorch::CTensor> CTorch::CTensor::pop() {
    std::unique_ptr<DType> dtype = std::move(arr.front());
    std::unique_ptr<CTorch::CTensor> ct(static_cast<CTorch::CTensor*>(dtype.release()));
    arr.erase(arr.begin());
    return ct;
}

std::unique_ptr<CTorch::CTensor> CTorch::CTensor::t() const {
    try {
        std::vector<std::any> temp = std::any_cast<std::vector<std::any>>(getValue());
        std::vector<std::vector<std::any>> temp_arr;
        for (int i = 0; i < temp.size(); i++) {
            temp_arr.push_back(std::any_cast<std::vector<std::any>>(temp[i]));
        }
        std::vector<std::any> new_arr;
        for (int i = 0; i < temp_arr[0].size(); i++) {
            std::vector<std::any> row;
            for (int j = 0; j < temp_arr.size(); j++) {
                row.push_back(temp_arr[j][i]);
            }
            new_arr.push_back(row);
        }
        return std::make_unique<CTorch::CTensor>(new_arr, dtype);
    } catch (const std::logic_error &e) {
        std::cerr << "ValueError: " << e.what() << std::endl;
        return std::make_unique<CTorch::CTensor>(std::vector<std::any>{}, dtype);
    } catch (const std::bad_any_cast &e) {
        try {
            std::vector<std::any> temp;
            temp.push_back(std::any_cast<std::vector<std::any>>(getValue()));
            return std::make_unique<CTorch::CTensor>(temp, dtype);
        } catch (const std::bad_any_cast &e) {
            return std::make_unique<CTorch::CTensor>(std::any_cast<std::vector<std::any>>(getValue()), dtype);
        }
    }
}

CTorch::CTensor::Size::Size(std::vector<int>& size) : shape{size} {}

CTorch::CTensor::Size::~Size() {}

std::unique_ptr<CTorch::CTensor::Size> CTorch::CTensor::shape() const {
    std::vector<int> shape;
    shape.push_back(arr.size());
    if (arr.size() == 0) {
        return std::make_unique<CTorch::CTensor::Size>(shape);
    }
    auto* ct = dynamic_cast<CTorch::CTensor*>(arr[0].get());
    while (ct != nullptr) {
        shape.push_back(ct->arr.size());
        if (ct->arr.size() == 0) { break; }
        ct = dynamic_cast<CTorch::CTensor*>(ct->arr[0].get());
    }
    return std::make_unique<CTorch::CTensor::Size>(shape);
}
int CTorch::CTensor::size() const {
    int x = 0;
    std::any temp = getValue();
    while (true) {
        try {
            std::vector<std::any> t = std::any_cast<std::vector<std::any>>(temp);
            if (t.empty()) { return x; }
            x++;
            temp = t[0];
        } catch (const std::bad_any_cast &e) {
            break;
        }
    }
    return x;
}

std::ostream& operator<<(std::ostream& os, const CTorch::CTensor::Size& s) {
    os << "torch.Size";
    os << "[";
    for (int i = 0; i < s.shape.size(); i++) {
        os << s.shape[i];
        if (i < s.shape.size() - 1) {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

int CTorch::CTensor::Size::operator[](std::size_t dim) const {
    try {
        if (dim < 0 || dim >= shape.size()) {
            throw std::logic_error("index " + std::to_string(dim) + " is out of bounds");
        }
        return shape[dim];
    } catch (const std::logic_error &e) {
        std::cerr << "IndexError: " << e.what() << std::endl;
        return 0;
    }
}

std::default_random_engine CTorch::generator;
std::normal_distribution<float> CTorch::distribution(0.0f, 1.0f);
