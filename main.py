import numpy as np
import neural_network as nn

number_of_vectors = 5

# Для регрессии: input = output
# Для классификации: input = 2, output = 1
number_of_input_elements = 2
number_of_output_elements = 2

number_of_neurons = 3
number_of_hidden_layers = 1

# relu, sigmoid, tanh, leaky relu, softmax, l2
function_types = ['leaky relu', 'softmax']

# regression, classification
problem_type = 'classification'

neural_network = nn.NeuralNetwork(number_of_vectors, number_of_input_elements, number_of_output_elements,
                                  number_of_neurons, number_of_hidden_layers, problem_type, function_types)

number_of_epochs = 50
learning_rate = 0.01
neural_network.train(number_of_epochs, learning_rate)
