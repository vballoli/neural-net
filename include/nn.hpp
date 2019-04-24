#include <iostream>
#include <vector>
#include "neuron.hpp"

using namespace std;
typedef vector<Neuron> Layer;

class NN
{
private:
    vector<Layer> nn_layers;
    
public:
    NN(vector<int> &arch, double learning_rate);
    void feedForward(vector<double> &input_data);
    void backProp(vector<double> &target_values);
    void getOutput(vector<double> &results);
    ~NN();
};