import numpy as np


# Базовый класс слоя
class Layer:
    def __init__(self, number_of_vectors, number_of_elements):
        self.number_of_vectors = number_of_vectors
        self.number_of_elements = number_of_elements
        self.array = np.zeros((number_of_vectors, number_of_elements))

    # Кто?
    def backward(self):
        pass

    # Что?
    def forward(self):
        pass


# Слой входных данных
class InputLayer(Layer):
    def __init__(self, number_of_vectors, number_of_elements):
        super().__init__(number_of_vectors, number_of_elements)
        # self.layer = np.zeros(number_of_vectors, number_of_elements)

    np.random.seed(1)

    def generate(self):
        self.array = np.random.randn(self.number_of_vectors, self.number_of_elements)


# Слой выходных данных
class OutputLayer(Layer):
    def __init__(self, number_of_vectors, number_of_elements):
        super().__init__(number_of_vectors, number_of_elements)

    def generate(self):
        return np.random.randn(self.number_of_vectors, self.number_of_elements)


# Скрытый слой
class HiddenLayer(Layer):
    def __init__(self, number_of_vectors, number_of_elements, number_of_neurons):
        super().__init__(number_of_vectors, number_of_elements)
        self.number_of_neurons = number_of_neurons


# Функция активации
class ActivationFunction(Layer):
    def __init__(self, number_of_vectors, number_of_elements, number_of_neurons, function_type):
        super().__init__(number_of_vectors, number_of_elements)
        self.number_of_neurons = number_of_neurons
        self.function_type = function_type

    def unvectorized_relu(self, sum):
        return max(0.0, sum)

    def unvectorized_sigmoid(self, sum):
        return 1.0 / (1.0 + np.exp(-sum))

    def unvectorized_tanh(self, sum):
        return (2.0 / (1.0 + np.exp(-2 * sum))) - 1.0

    def unvectorized_leaky_relu(self, sum):
        return max(0.1 * sum, sum)

    def activate(self, array, function_type):
        if function_type == 'relu':
            vectorized_function = np.vectorize(self.unvectorized_relu)
        elif function_type == 'sigmoid':
            vectorized_function = np.vectorize(self.unvectorized_sigmoid)
        elif function_type == 'tanh':
            vectorized_function = np.vectorize(self.unvectorized_tanh)
        elif function_type == 'leaky relu':
            vectorized_function = np.vectorize(self.unvectorized_leaky_relu)
        else:
            vectorized_function = np.vectorize(self.unvectorized_relu)

        return vectorized_function(array)