#include <iostream>
#include "data.hpp"

int main() {
    std::cout << "Starting neural network" << std::endl;
    Data d;
    vector<vector<double>> training = d.formatData("/Users/balli/Coding/CourseProjects/ML/perceptron-fisher-lda/datasets/dataset_1.csv");
}