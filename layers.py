import numpy as np
from matplotlib import pyplot as plt


# Базовый класс слоя
class Layer:
    def __init__(self, number_of_vectors, number_of_elements):
        self.number_of_vectors = number_of_vectors
        self.number_of_elements = number_of_elements
        self.array = np.zeros((number_of_vectors, number_of_elements))
        self.x = None
        self.y = None

    def backward(self, upstream_gradient, output_array=None):
        pass

    def forward(self, elements, weights):
        pass

    # Сохранение данных для backward
    def save(self, x, y=None):
        self.x = x
        self.y = y

    # Загрузка данных для backward
    def load(self):
        return self.x, self.y


# Слой входных данных
class InputLayer(Layer):
    def __init__(self, number_of_vectors, number_of_elements):
        super().__init__(number_of_vectors, number_of_elements)

    np.random.seed(1)

    # Вектор входных данных (x)
    def generate(self):
        self.array = np.random.randn(self.number_of_vectors, self.number_of_elements)

    # Вектор входных данных для задачи классификации (x, y)
    def generate_classification(self):
        array = []
        for _ in range(self.number_of_vectors):
            x = np.random.randn()
            delta = np.random.uniform(-1, 1)
            array.append([x, np.sin(x) + delta])
        self.array = np.array(array)

    def forward(self, elements, weights):
        print('input layer):')
        print(self.array)
        return self.array


# Слой выходных данных
class OutputLayer(Layer):
    def __init__(self, number_of_vectors, number_of_elements):
        super().__init__(number_of_vectors, number_of_elements)

    # Вектор выходных данных (y)
    # Функция — sin(x)
    def generate(self, input_layer):
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
        self.array = np.array([[1] if element[1] >= 0 else [0] for element in input_layer.array])


# Скрытый слой
class HiddenLayer(Layer):
    def __init__(self, number_of_vectors, number_of_elements, number_of_neurons, layer_index):
        super().__init__(number_of_vectors, number_of_elements)
        self.number_of_neurons = number_of_neurons
        self.layer_index = layer_index

    def forward(self, elements, weights):
        # Сохранение элементов и весов для backwards
        self.save(elements, weights)
        # Умножение элементов на веса
        self.array = elements.dot(weights)
        print('hidden layer %s):' % (self.layer_index + 1))
        print(self.array)
        return self.array

    def backward(self, upstream_gradient, output_array=None):
        # Загрузка элементов и весов
        elements, weights = self.load()

        # Умножение upstream и local градиентов
        grad_elements = weights * upstream_gradient     # dz/dx * dL/dz
        grad_weights = elements * upstream_gradient     # dz/dy * dL/dz
        # (x и y кстати местами поменяны, это важно)
        # Тяжело воспринимать производные в коде, поэтому можно запутаться

        return grad_elements, grad_weights


# Функция активации
class ActivationFunction(Layer):
    def __init__(self, number_of_vectors, number_of_elements, number_of_neurons, function_type):
        super().__init__(number_of_vectors, number_of_elements)
        self.number_of_neurons = number_of_neurons
        self.function_type = function_type
        # Теперь сразу в конструкторе определяем функцию, которая будет храниться в function
        # TODO: Может, отдельный класс для каждой функции?..
        if function_type == 'relu':
            self.function = np.vectorize(unvectorized_relu)
        elif function_type == 'sigmoid':
            self.function = np.vectorize(unvectorized_sigmoid)
        elif function_type == 'tanh':
            self.function = np.vectorize(unvectorized_tanh)
        elif function_type == 'leaky relu':
            self.function = np.vectorize(unvectorized_leaky_relu)
        elif function_type == 'softmax':
            self.function = softmax
        elif function_type == 'l2':
            self.function = l2
        # ReLU по умолчанию
        else:
            self.function = np.vectorize(unvectorized_relu)

    def forward(self, elements, weights):
        print('activation function layer — %s):' % self.function_type)
        self.array = self.function(elements)

        # Сохранение элементов для backwards
        self.save(self.array)

        print(self.array)
        return self.array
        # L2 не работает, нужно как-то передавать output_layer.array, чтобы метод при этом не выглядел как говно,
        # потому что этот массив нужен только для одной функции из 6
        # TODO: Как вариант, вернуться к output_array=None

    def load(self):
        return self.x

    # Производные честно загуглены
    def backward(self, upstream_gradient, output_array=None):
        # Загрузка элементов
        array = self.load()

        # Домножаю на upstream_gradient
        if self.function_type == 'relu':
            return 1.0 if array >= 0.0 else 0.0
        elif self.function_type == 'sigmoid':
            return (1.0 - array).dot(array) * upstream_gradient
        elif self.function_type == 'tanh':
            return 1.0 - np.power(self.function(array), 2)
        elif self.function_type == 'leaky relu':
            return 1.0 if array >= 0.0 else 0.1

        # https://themaverickmeerkat.com/2019-10-23-Softmax/
        elif self.function_type == 'softmax':
            m, n = array.shape
            p = softmax(array)
            tensor1 = np.einsum('ij,ik->ijk', p, p)
            tensor2 = np.einsum('ij,jk->ijk', p, np.eye(n, n))
            d_softmax = tensor2 - tensor1
            return np.einsum('ijk,ik->ij', d_softmax, upstream_gradient)

        elif self.function_type == 'l2':
            return 2.0 * np.subtract(array, output_array)
        # ReLU по умолчанию
        else:
            return 1.0 if array >= 0.0 else 0.0


def unvectorized_relu(x):
    return max(0.0, x)


def unvectorized_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def unvectorized_tanh(x):
    return (2.0 / (1.0 + np.exp(-2 * x))) - 1.0


def unvectorized_leaky_relu(x):
    return max(0.1 * x, x)


def softmax(x):
    # Без "[:, None]" появляется ошибка ValueError: operands could not be broadcast together with shapes
    # https://howtothink.readthedocs.io/en/latest/PvL_06.html
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=len(e_x.shape) - 1)[:, None]


def l2(x, y):
    return np.power(np.subtract(x, y), 2)


# Тесты приколов (с графиком)
if __name__ == "__main__":
    test = np.array(
        [[2.0, 1.0, 0.1],
         [1.0, 2.0, 0.3],
         [2.0, 1.0, 0.1],
         [2.0, 1.0, 0.1]])
    arr2 = softmax(test)
    print(arr2)

    input_layer_classification = InputLayer(20, 100)
    output_layer_classification = OutputLayer(20, 100)
    input_layer_classification.generate_classification()
    output_layer_classification.generate_classification(input_layer_classification)

    input_layer_regression = InputLayer(20, 100)
    output_layer_regression = OutputLayer(20, 100)
    input_layer_regression.generate()
    output_layer_regression.generate(input_layer_regression)

    plt.plot(input_layer_regression.array, output_layer_regression.array, 'o', color='black', markersize=1)
    plt.show()
