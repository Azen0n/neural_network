import numpy as np
import neural_network as nn

number_of_vectors = 1
number_of_input_elements = 4
number_of_output_elements = 3
number_of_neurons = 4
number_of_hidden_layers = 1
function_type = 'relu'

neural_network = nn.NeuralNetwork(number_of_vectors, number_of_input_elements, number_of_output_elements,
                                  number_of_neurons, number_of_hidden_layers, function_type)

output = neural_network.compute()
