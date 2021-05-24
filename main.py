import numpy as np
import neural_network as nn

number_of_vectors = 1
# input и output должны быть равны
number_of_input_elements = 4
number_of_output_elements = 4

number_of_neurons = 5
number_of_hidden_layers = 1
# relu, sigmoid, tanh, leaky relu
# softmax, l2
function_types = ['sigmoid', 'sigmoid', 'sigmoid', 'sigmoid']
# regression, classification
problem_type = 'regression'

neural_network = nn.NeuralNetwork(number_of_vectors, number_of_input_elements, number_of_output_elements,
                                  number_of_neurons, number_of_hidden_layers, problem_type,
                                  function_types)

output = neural_network.compute_forward()

backward_stuff = neural_network.compute_backward()
