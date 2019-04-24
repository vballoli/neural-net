#include <iostream>
#include <vector>
#include <math.h>
#include <string.h>
#include <random>
#include <functional>
#include <algorithm>
#include <cstdlib>
#include "neuron.hpp"

using namespace std;


Neuron::Neuron(double learning_rate, int output_count, int neuron_position)
{
    alpha = learning_rate;

    for (int i=0; i <= output_count; ++i) {
        output_weights.push_back(Link());
        output_weights.back().weight = rand()/RAND_MAX ;
    }

    neuron_index = neuron_position;
    
    // if (find(Neuron::SUPPORTED_ACTIVATIONS.begin(), Neuron::SUPPORTED_ACTIVATIONS.end(), activation_function) 
    //         != Neuron::SUPPORTED_ACTIVATIONS.end) {

    // } else {
    //     cout<<"Wrong activation function"<<endl;
    // }
}

Neuron::~Neuron()
{
}

double Neuron::linear(double x)
{
    return x;
}

double Neuron::sigmoid(double x) 
{
    return 1/(1 + exp(-1 * x));
}

double Neuron::tanh(double x) 
{
    try {
        return tanh(x);
    } catch (exception e) {
        cout<<e.what()<<endl;
        return 0;
    }
}

double Neuron::derivativeLinear(double x)
{
    return 1;
}

double Neuron::derivativeSigmoid(double x) 
{
    return x * (1-x);
}

double Neuron::derivativeTanh(double x) 
{
    try {
        return 1 - x * x;
    } catch (exception e) {
        cout<<e.what()<<endl;
        return 0;
    }
}

void Neuron::hiddenGrad (Layer &next_layer)
{
    double sum = 0;

    for (int i=0; i<next_layer.size(); ++i) {
        sum += output_weights[i].weight * next_layer[i].grad;
    }

    grad = sum * derivativeTanh(output_value);

}

void Neuron::outputGrad (double target) 
{
    double del = target - output_value;
    grad = del * derivativeTanh(output_value);
}

void Neuron::updateWeight(Layer &previous_layer)
{
    for (int i=0; i<previous_layer.size(); ++i) {
        Neuron &neuron = previous_layer[i];
        
        double delta = alpha * neuron.getOutputValue() * grad;
        neuron.output_weights[neuron_index].weight += delta;
        neuron.output_weights[neuron_index].deltaWeight = delta;
    }
}

void Neuron::feedForward(Layer &previous_layer) 
{
    double sum = 0;

    for (int i=0; i<previous_layer.size(); ++i) {
        sum += previous_layer[i].getOutputValue() * previous_layer[i].output_weights[neuron_index].weight;
    }

    setOutputValue(tanh(sum));
}
