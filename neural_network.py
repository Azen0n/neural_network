import numpy as np
import layers as lrs


class NeuralNetwork:
    def __init__(self, number_of_vectors, number_of_input_elements,
                 number_of_output_elements, number_of_neurons, number_of_hidden_layers, function_type):
        self.number_of_vectors = number_of_vectors
        self.number_of_input_elements = number_of_input_elements
        self.number_of_output_elements = number_of_output_elements
        self.number_of_neurons = number_of_neurons
        self.number_of_hidden_layers = number_of_hidden_layers
        self.function_type = function_type

        self.input_layer = lrs.InputLayer(number_of_vectors, number_of_input_elements)
        lrs.InputLayer.generate(self.input_layer)

        self.output_layer = lrs.OutputLayer(number_of_vectors, number_of_output_elements)

        self.activation_function = lrs.ActivationFunction(number_of_vectors, number_of_input_elements,
                                                      number_of_neurons, function_type)
        self.hidden_layers = list()
        for _ in range(number_of_hidden_layers):
            self.hidden_layers.append(lrs.HiddenLayer(number_of_vectors, number_of_input_elements, number_of_neurons))

    def compute(self):
        elements = self.input_layer.array
        elements_count = self.number_of_input_elements
        neurons_count = self.number_of_neurons

        i = 1
        for hidden_layer in self.hidden_layers:
            weights = np.random.randn(elements_count, neurons_count)
            hidden_layer.array = elements.dot(weights)
            print('\nHidden layer %s:' % i)
            print(hidden_layer.array)

            elements = lrs.ActivationFunction.activate(self.activation_function, hidden_layer.array, self.function_type)

            print('\nResult after activation function:')
            print(elements)
            elements_count = self.number_of_neurons
            i += 1

        weights = np.random.randn(elements_count, self.number_of_output_elements)

        self.output_layer.array = elements.dot(weights)
        print('\nOutput:')
        print(self.output_layer.array)
        return self.output_layer.array
