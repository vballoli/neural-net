#include <iostream>
#include <vector>
#include "nn.hpp"

using namespace std;
typedef vector<Neuron> Layer;

NN::NN(vector<int> &arch, double learning_rate)
{
    int layers_count = arch.size();
    for (int layer_num=0; layer_num < layers_count; ++layer_num) {
        // Add a layer
        nn_layers.push_back(Layer());

        int output_count = 0;
        if (layer_num != layers_count - 1) int output_count = arch[layer_num + 1];

        // Add corresponding number of neurons to that layer
        for (int neuron_num=0; neuron_num < arch[layer_num]; ++neuron_num) {
            nn_layers.back().push_back(Neuron(learning_rate, output_count, neuron_num));
        }

        nn_layers.back().back().setOutputValue(1);
    }
}

void NN::feedForward (vector<double> &input_data)
{
    // Check for the input condition
    //assert(input_data.size() == nn_layers[0] - 1); // REmove the bias neuron for the check condition

    for (int i=0; i < input_data.size(); ++i) {
        nn_layers[0][i].setOutputValue(input_data[i]);
    }

    for (int layer_index = 1; layer_index < nn_layers.size(); layer_index++) {
        for (int i = 0; i < nn_layers[layer_index].size(); ++i) {
            Layer l = nn_layers[layer_index - 1];
            nn_layers[layer_index][i].feedForward(l);
        }
    }
}

void NN::backProp(vector<double> &target_values)
{
    double error = 0;

    Layer &last_layer = nn_layers.back();

    for (int i=0; i<nn_layers.back().size() - 1; ++i) {
        error += pow(target_values[i] - last_layer[i].getOutputValue(), 2);
    }

    error /= last_layer.size() - 1;
    error = pow(error, 0.5);

    for (int i=0; i < last_layer.size(); ++i) {
        last_layer[i].outputGrad(target_values[i]);
    }

    for (int i=nn_layers.size()-2 ; i >0; --i ) {
        Layer &current_hidden_layer = nn_layers[i];
        Layer &next_layer = nn_layers[i+1];

        for (int n=0; n < current_hidden_layer.size(); ++n) {
            current_hidden_layer[n].hiddenGrad(next_layer);
        }
    }


    for (int layer_index=nn_layers.size()-1; layer_index>0; --layer_index) {
        Layer &current_layer = nn_layers[layer_index];
        Layer &previous_layer = nn_layers[layer_index-1];

        for(int i=0; i<current_layer.size(); ++i) {
            current_layer[i].updateWeight(previous_layer);
        }
    }
}

void NN::getOutput(vector<double> &results)
{
    results.clear();

    for (int i=0; i<nn_layers.back().size()-1; ++i) {
        results.push_back(nn_layers.back()[i].getOutputValue());
    }
}

NN::~NN()
{
}
