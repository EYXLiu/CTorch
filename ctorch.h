#ifndef CTORCH_H
#define CTORCH_H

#include <iostream>
#include <any>
#include <cstdint>
#include <vector>
#include <variant>
#include <typeinfo>
#include <random>
#include <memory>

class CTorch {
    public:
        enum ScalarType {
            Int32,
            Int64,
            Float32,
            Float64,
            Int = Int32,
            Long = Int64,
            Float = Float32,
            Double = Float64,
        };
        class DType {
            public:
                virtual ~DType() = default;
                friend std::ostream& operator<<(std::ostream& os, const DType& dt);
                using ReturnType = std::variant<std::int32_t, std::int64_t, float, double, std::vector<std::any>>;
                virtual std::any getValue() const = 0;
        };
        class CInt32 : public DType {
            std::int32_t v;
            public:
                CInt32(std::int32_t v);
                ~CInt32() override;
                std::any getValue() const override;
        };
        class CInt64 : public DType {
            std::int64_t v;
            public:
                CInt64(std::int64_t v);
                ~CInt64() override;
                std::any getValue() const override;
        };
        class CFloat32 : public DType {
            float v;
            public:
                CFloat32(float v);
                ~CFloat32() override;
                std::any getValue() const override;
        };
        class CFloat64 : public DType {
            double v;
            public:
                CFloat64(double v);
                ~CFloat64() override;
                std::any getValue() const override;
        };
        class CTensor : public DType {
            std::vector<std::unique_ptr<DType>> arr;
            ScalarType dtype;
            std::vector<std::any> unsqueezeHelp(int dim, std::vector<std::any> arr) const;
            public:
                CTensor(std::vector<std::any> arr, ScalarType dtype = CTorch::Float);
                // template in header file as so have inline implementation
                template <typename T>
                CTensor(std::initializer_list<T> list, ScalarType dtype = CTorch::Float) : dtype{dtype} {
                    for (const auto& a : list) {
                        if (dtype == CTorch::Int32) {
                            arr.push_back(std::make_unique<CInt32>(a));
                        } else if (dtype == CTorch::Int64) {
                            arr.push_back(std::make_unique<CInt64>(a));
                        } else if (dtype == CTorch::Float32) {
                            arr.push_back(std::make_unique<CFloat32>(a));
                        } else if (dtype == CTorch::Float64) {
                            arr.push_back(std::make_unique<CFloat64>(a));
                        }
                    }
                }
                template <typename T>
                CTensor(std::initializer_list<std::initializer_list<T>> list, ScalarType dtype = CTorch::Float) : dtype{dtype} {
                    for (const auto& sublist : list) {
                        arr.push_back(std::make_unique<CTensor>(sublist, dtype));
                    }
                }
                template <typename T>
                CTensor(std::initializer_list<std::initializer_list<std::initializer_list<T>>> list, ScalarType dtype = CTorch::Float) : dtype{dtype} {
                    for (const auto& sublist : list) {
                        arr.push_back(std::make_unique<CTensor>(sublist, dtype));
                    }
                }
                CTensor(const CTensor& other);
                ~CTensor() override;
                std::any getValue() const override;
                int getType() const;
                std::unique_ptr<CTensor> add(const CTensor& other) const;
                std::unique_ptr<CTensor> matmul(const CTensor& other) const;
                std::unique_ptr<CTensor> mul(float i) const;
                std::unique_ptr<DType> dot(const CTensor& other) const;
                std::unique_ptr<CTensor> unsqueeze(int dim) const;
                std::unique_ptr<CTensor> operator[](std::size_t dim) const;
                //purely for backprop for linear layer
                std::unique_ptr<CTensor> sigmoid(bool deriv=false) const;
                std::unique_ptr<CTensor> sumLin() const;
                void append(std::unique_ptr<CTensor>& ct);
                std::unique_ptr<CTensor> pop();
                // transposes (0, 1) im not entirely sure how to swap higher dimension tensors but its not that useful so :D
                std::unique_ptr<CTensor> t() const;
                class Size {
                    std::vector<int> shape;
                    public:
                        Size(std::vector<int>& size);
                        friend std::ostream& operator<<(std::ostream& os, const Size& s);
                        int operator[](std::size_t dim) const;
                        ~Size();
                };
                std::unique_ptr<Size> shape() const;
                int size() const;
        };
        // template in header file as so have inline implementation
        template <typename T, typename... Args>
        static std::unique_ptr<CTensor> zeros(T first, Args... args) {
            ScalarType t = CTorch::Float;
            std::vector<std::any> arr;
            if constexpr (sizeof...(args) > 0) {
                for (int i = 0; i < first; i++) {
                    arr.push_back(std::any(CTorch::zeros(args...)->getValue()));
                }
            } else {
                arr = std::vector<std::any>(first, std::any(0.0f));
            }
            return std::make_unique<CTorch::CTensor>(arr, t);
        }
        template<typename T, typename... Args>
        static std::unique_ptr<CTensor> ones(T first, Args... args) {
            ScalarType t = CTorch::Float;
            std::vector<std::any> arr;
            if constexpr (sizeof...(args) == 0) {
                arr = std::vector<std::any>(first, std::any(1.0f));
            } else {
                for (int i = 0; i < first; i++) {
                    arr.push_back(std::any(CTorch::ones(args...)->getValue()));
                }
            }
            return std::make_unique<CTorch::CTensor>(arr, t);
        }
        static std::default_random_engine generator;
        static std::normal_distribution<float> distribution;

        template<typename T, typename... Args>
        static std::unique_ptr<CTensor> randn(T first, Args... args) {
            ScalarType t = CTorch::Float;
            std::vector<std::any> arr;
            if constexpr (sizeof...(args) == 0) {
                for (int i = 0; i < first; i++) {
                    float value = distribution(generator);
                    arr.push_back(std::any(value));
                }
            } else {
                for (int i = 0; i < first; i++) {
                    arr.push_back(std::any(CTorch::randn(args...)->getValue()));
                }
            }
            return std::make_unique<CTorch::CTensor>(arr, t);
        }

};

#endif // CTORCH_H
