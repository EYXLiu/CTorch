#include "ctorch.h"

std::ostream& operator<<(std::ostream& os, const std::any &rt) {
    if (auto* v = std::any_cast<std::int32_t>(&rt)) {
        os << *v;
    } else if (auto* v = std::any_cast<std::int64_t>(&rt)) {
        os << *v;
    } else if (auto* v = std::any_cast<float>(&rt)) {
        os << *v;
    } else if (auto* v = std::any_cast<double>(&rt)) {
        os << *v;
    } else if (auto* v = std::any_cast<std::vector<std::any>>(&rt)) {
        os << "[";
        for (int i = 0; i < v->size(); i++) {
            os << (*v)[i];
            if (i < v->size() - 1) {
                os << ", ";
            }
        }
        os << "]";
    }
    return os;
}

std::any CTorch::CTensor::getValue() const {
    std::vector<std::any> new_arr;
    for (auto& a : arr) {
        std::any temp = a->getValue();
        try {
            new_arr.push_back(std::any_cast<std::vector<std::any>>(temp));
        } catch (const std::bad_any_cast &e) {
            new_arr.push_back(a->getValue());
        }
    }
    return new_arr;
}

std::ostream& operator<<(std::ostream& os, const CTorch::DType& dt) {
    os << dt.getValue();
    return os;
}

CTorch::CInt32::CInt32(std::int32_t v) : v(v) {}
CTorch::CInt32::~CInt32() {}
std::any CTorch::CInt32::getValue() const { return v; }


CTorch::CInt64::CInt64(std::int64_t v) : v(v) {}
CTorch::CInt64::~CInt64() {}
std::any CTorch::CInt64::getValue() const { return v; }

CTorch::CFloat32::CFloat32(float v) : v(v) {}
CTorch::CFloat32::~CFloat32() {}
std::any CTorch::CFloat32::getValue() const { return v; }

CTorch::CFloat64::CFloat64(double v) : v(v) {}
CTorch::CFloat64::~CFloat64() {}
std::any CTorch::CFloat64::getValue() const { return v; }
