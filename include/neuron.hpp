#include <iostream>
#include <vector>
#include <math.h>
#include <string.h>
#include <random>
#include <functional>
#include <algorithm>
#include <cstdlib>

using namespace std;

struct Link
{
    double weight;
    double deltaWeight;
};

class Neuron
{
    typedef vector<Neuron> Layer;
private:
    string activation;
    double alpha;
    static double linear(double x);
    static double sigmoid(double x);
    static double tanh(double x);
    static double derivativeLinear(double x);
    static double derivativeSigmoid(double x);
    static double derivativeTanh(double x);
    int neuron_index;
    vector<Link> output_weights;
    double output_value;
    double grad;


protected:
    vector<double> w;

public:
    Neuron(double learning_rate, int output_count, int neuron_index);
    ~Neuron();
    void feedForward(Layer &previous_layer);
    void setOutputValue(double output) { output_value = output; }
    double getOutputValue() { return output_value; }
    void outputGrad (double target);
    void hiddenGrad(Layer &next_layer);
    void updateWeight(Layer &previous_layer);
    const static vector<string> SUPPORTED_ACTIVATIONS;
    const static string LINEAR;
    const static string SIGMOID;
    const static string TANH;
};

const string Neuron::LINEAR = string("linear");
const string Neuron::SIGMOID = string("sigmoid");
const string Neuron::TANH = string("tanh");
const vector<string> Neuron::SUPPORTED_ACTIVATIONS = {Neuron::LINEAR, Neuron::SIGMOID, Neuron::TANH};