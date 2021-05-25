import numpy as np


@np.vectorize
def relu(x):
    return max(0.0, x)


@np.vectorize
def relu_derivative(x):
    return 1.0 if x >= 0.0 else 0.0


@np.vectorize
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@np.vectorize
def sigmoid_derivative(x):
    return (1.0 - x) * x


@np.vectorize
def tanh(x):
    return (2.0 / (1.0 + np.exp(-2 * x))) - 1.0


@np.vectorize
def tanh_derivative(x):
    return 1.0 - np.power(tanh(x), 2)


@np.vectorize
def leaky_relu(x):
    return max(0.1 * x, x)


@np.vectorize
def leaky_relu_derivative(x):
    return 1.0 if x >= 0.0 else 0.1


def softmax(x):
    # Без "[:, None]" появляется ошибка ValueError: operands could not be broadcast together with shapes
    # https://howtothink.readthedocs.io/en/latest/PvL_06.html
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=len(e_x.shape) - 1)[:, None]


# reshape softmax to 2d so np.dot gives matrix multiplication
def softmax_derivative(x):
    s = x.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)



def l2(x, y):
    return np.power(np.subtract(x, y), 2)


def l2_derivative(x, y):
    return 2.0 * np.subtract(x, y)


# Возвращает функцию и ее производную
def get_function(function_type):
    if function_type == 'relu':
        return relu, relu_derivative
    elif function_type == 'sigmoid':
        return sigmoid, sigmoid_derivative
    elif function_type == 'tanh':
        return tanh, tanh_derivative
    elif function_type == 'leaky relu':
        return leaky_relu, leaky_relu_derivative
    elif function_type == 'softmax':
        return softmax, softmax_derivative
    elif function_type == 'l2':
        return l2, l2_derivative
    # ReLU по умолчанию
    else:
        return relu, relu_derivative
