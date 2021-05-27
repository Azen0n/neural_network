import numpy as np
from matplotlib import pyplot as plt
from activation_functions import get_function


# Базовый класс слоя
class Layer:
    def __init__(self, number_of_vectors, number_of_elements):
        self.number_of_vectors = number_of_vectors
        self.number_of_elements = number_of_elements
        self.array = np.zeros((number_of_vectors, number_of_elements))
        self.saved_array = None

    def backward(self, upstream_gradient, output_array=None):
        pass

    def forward(self, elements):
        pass

    # Сохранение данных для backward
    def save(self, saved_array):
        self.saved_array = saved_array

    # Загрузка данных для backward
    def load(self):
        return self.saved_array


# Слой входных данных
class InputLayer(Layer):
    def __init__(self, number_of_vectors, number_of_elements, seed):
        super().__init__(number_of_vectors, number_of_elements)
        np.random.seed(seed)

    # Вектор входных данных (x)
    def generate_regression(self):
        self.array = np.random.randn(self.number_of_vectors, self.number_of_elements)

    # Вектор входных данных для задачи классификации (x, y)
    def generate_classification(self):
        array = []
        for _ in range(self.number_of_vectors):
            x = np.random.randn()
            delta = np.random.uniform(-1, 1)
            array.append([x, np.sin(x) + delta])
        self.array = np.array(array)

    def forward(self, elements):
        print('input layer):')
        print(self.array)
        return self.array


# Слой выходных данных
class OutputLayer(Layer):
    def __init__(self, number_of_vectors, number_of_elements):
        super().__init__(number_of_vectors, number_of_elements)
        self.weights = None
        self.predicted_array = None
        self.saved_array = None

    # Генерация весов
    # number_of_neurons — количество нейронов ПОСЛЕДНЕГО СЛОЯ
    def generate_weights(self, number_of_neurons):
        self.weights = np.array(
            [[np.random.uniform(0, 1) for _ in range(self.number_of_elements)] for _ in
             range(number_of_neurons)])

    def forward(self, elements):
        self.saved_array = elements
        self.predicted_array = elements.dot(self.weights)
        print('output layer):')
        print(self.predicted_array)
        return self.predicted_array

    def backward(self, upstream_gradient, output_array=None):
        # Умножение upstream и local градиентов
        grad_weights = self.saved_array.T.dot(upstream_gradient)
        grad_elements = upstream_gradient.dot(self.weights.T)
        return grad_elements, grad_weights

    # Вектор выходных данных (y)
    # Функция — sin(x)
    def generate_regression(self, input_layer):
        sin_input_layer = np.sin(input_layer.array)
        array = []
        for vector in sin_input_layer:
            array_vector = []
            for element in vector:
                delta = np.random.uniform(-1, 1)
                array_vector.append(element + delta)
            array.append(array_vector)
        self.array = np.array(array)

    # Вектор с метками класса
    def generate_classification(self, input_layer):
        self.array = np.array([[1, 0] if element[1] >= 0 else [0, 1] for element in input_layer.array])


# Скрытый слой
class HiddenLayer(Layer):
    def __init__(self, number_of_vectors, number_of_elements, number_of_neurons, layer_index):
        super().__init__(number_of_vectors, number_of_elements)
        self.number_of_neurons = number_of_neurons
        self.layer_index = layer_index
        # TODO: Если для каждого скрытого слоя задавать свое количество нейронов, то первый параметр должен быть
        #  self.prev.number_of_neurons, то есть придется добавить отдельное поле для хранения предыдущего слоя
        self.weights = np.array(
            [[np.random.uniform(0, 1) for _ in range(self.number_of_neurons)] for _ in
             range(self.number_of_neurons)])

    def forward(self, elements):
        # Сохранение элементов для backwards
        self.saved_array = elements
        # Умножение элементов на веса
        self.array = elements.dot(self.weights)
        print('hidden layer %s):' % (self.layer_index + 1))
        print(self.array)
        return self.array

    def backward(self, upstream_gradient, output_array=None):
        grad_weights = self.saved_array.T.dot(upstream_gradient)
        grad_elements = upstream_gradient.dot(self.weights.T)
        return grad_elements, grad_weights


# Функция активации
class ActivationFunction(Layer):
    def __init__(self, number_of_vectors, number_of_elements, number_of_neurons, function_type):
        super().__init__(number_of_vectors, number_of_elements)
        self.number_of_neurons = number_of_neurons
        self.function_type = function_type
        # Сразу в конструкторе определяем функцию и ее производную
        self.function, self.derivative = get_function(function_type)

    def forward(self, elements, output_array=None):
        print('activation function layer — %s):' % self.function_type)

        self.saved_array = elements

        if output_array is not None:
            self.array = self.function(elements, output_array)
        else:
            self.array = self.function(elements)

        print(self.array)
        return self.array

    def backward(self, upstream_gradient, output_array=None):

        if output_array is not None:
            return self.derivative(self.saved_array, output_array) * upstream_gradient, None
        else:
            return self.derivative(self.saved_array) * upstream_gradient, None


if __name__ == "__main__":
    print('Wrong script!')
