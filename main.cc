#include "ctorch.h"
#include "ctorch.cnn.h"
#include "ctorch.cnn.functional.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <tuple>

std::vector<std::vector<std::any>> readCSV(std::string filename, int cols) {
    std::fstream fin;
    fin.open(filename, std::ios::in);

    if (!fin) {
        std::cerr << "Error: Could not open the file IRIS.csv\n";
        return std::vector<std::vector<std::any>>{};
    }
    std::string line;
    std::vector<std::vector<std::any>> data;
    std::getline(fin, line);

    while (std::getline(fin, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<std::any> row;

        for (int i = 0; i < cols; i++) {
            std::getline(ss, value, ',');
            row.push_back(std::stof(value));
        }
        std::getline(ss, value, ',');
        row.push_back(value);
        data.push_back(row);
    }

    return data;
}

std::vector<int> getY(std::vector<std::vector<std::any>>* x) {
    enum Values {
        Setosa, 
        Versicolor, 
        Virginica,
    };
    std::vector<int> y;
    for (int i = 0; i < x->size(); i++) {
        std::string value = std::any_cast<std::string>((*x)[i][(*x)[i].size() - 1]);
        value.erase(value.size() - 1);
        (*x)[i].pop_back();
        if (value == "Iris-setosa") {
            y.push_back(Values::Setosa);
        } else if (value == "Iris-versicolor") {
            y.push_back(Values::Versicolor);
        } else if (value == "Iris-virginica") {
            y.push_back(Values::Virginica);
        }
    }
    return y;
}

int predict(std::unique_ptr<CTorch::CTensor>& predicted) {
    int max_index = 0;
    float max_value = std::any_cast<float>((*predicted)[0]->getValue());

    for (int i = 0; i < predicted->size(); i++) {
        if (std::any_cast<float>((*predicted)[i]->getValue()) > max_value) {
            max_index = i;
            max_value = std::any_cast<float>((*predicted)[i]->getValue());
        }
    }
    return max_index;
}

int main() {
    //data cleaning and appropriating into requirements
    std::vector<std::vector<std::any>> x = readCSV("IRIS.csv", 4);
    std::mt19937 g;
    std::shuffle(x.begin(), x.end(), g);

    std::vector<int> yvalues = getY(&x);
    std::cout << x.size() << std::endl;
    std::cout << yvalues.size() << std::endl;

    std::vector<std::vector<std::any>> y;
    for (int i = 0; i < yvalues.size(); i++) {
        if (yvalues[i] == 0) {
            y.push_back(std::vector<std::any>{1, 0, 0});
        } else if (yvalues[i] == 1) {
            y.push_back(std::vector<std::any>{0, 1, 0});
        } else {
            y.push_back(std::vector<std::any>{0, 0, 1});
        }
    }
    int split = x.size() * 0.8;
    std::vector<std::vector<std::any>> xtrain(x.begin(), x.begin() + split);
    std::vector<std::vector<std::any>> ytrain(y.begin(), y.begin() + split);
    std::vector<std::vector<std::any>> xtest(x.begin() + split, x.end());
    std::vector<std::vector<std::any>> ytest(y.begin() + split, y.end());


    CTorch ctorch;
    std::vector<std::unique_ptr<Cnn::CModule>> ml;
    ml.push_back(std::make_unique<Cnn::CLinear>(4, 16));
    ml.push_back(std::make_unique<Cnn::CLinear>(16, 3));
    Cnn::CSequential* nn = new Cnn::CSequential(ml);
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < xtrain.size(); j++) {
            std::unique_ptr<CTorch::CTensor> temp = std::make_unique<CTorch::CTensor>(xtrain[j]);
            std::unique_ptr<CTorch::CTensor> result = nn->forward(temp);
            softmax(result);
            std::unique_ptr<CTorch::CTensor> expected = std::make_unique<CTorch::CTensor>(ytrain[j]);
            std::unique_ptr<CTorch::CTensor> grad;
            std::tuple<std::unique_ptr<CTorch::CTensor>, std::unique_ptr<CTorch::CTensor>> gradient = nn->backgrad(grad, expected);
            nn->backpass(std::get<0>(gradient));
        }
        std::cout << "Epoch " << i << std::endl;
    }
    std::unique_ptr<CTorch::CTensor> ypredict = std::make_unique<CTorch::CTensor>(xtest[0]);
    std::cout << nn->forward(ypredict) << std::endl;

}