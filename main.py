import numpy as np
import neural_network as nn

number_of_vectors = 5

# Для регрессии: input = output
# Для классификации: input = 2, output = 2
number_of_input_elements = 5
number_of_output_elements = 5

number_of_neurons = 5
number_of_hidden_layers = 1

# relu, sigmoid, tanh, leaky relu, softmax, l2
function_types = ['leaky relu', 'l2']

# regression, classification
problem_type = 'regression'

neural_network = nn.NeuralNetwork(number_of_vectors, number_of_input_elements, number_of_output_elements,
                                  number_of_neurons, number_of_hidden_layers, problem_type, function_types, 1)

number_of_epochs = 300
learning_rate = 0.00001
neural_network.train(number_of_epochs, learning_rate)
