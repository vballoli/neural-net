#include <iostream>
#include <vector>
#include "data.hpp"
#include "nn.hpp"

using namespace std;
typedef vector<Neuron> Layer;

// typdef vector<Neuron> Layer;

int main() {
    std::cout << "Starting neural network" << std::endl;
    Data d;
    vector<vector<double>> training = d.formatData("/Users/balli/Coding/CourseProjects/ML/perceptron-fisher-lda/datasets/dataset_1.csv");   

    vector<int> arch;
    arch.push_back(24);
    arch.push_back(12);
    arch.push_back(6);
    arch.push_back(3);
    arch.push_back(1);

    vector<double> training_data;
    vector<double> training_labels;
    vector<double> test_data;
    
    NN neuralNet(arch, 0.1); 
    // neuralNet.feedForward(training_data);
    // neuralNet.backProp(training_labels);
    // neuralNet.predict(test_Data);
}