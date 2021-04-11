import numpy as np

N, D, C = 5, 3, 2
H = 4

np.random.seed(1)

x = np.random.randn(N, D)
y = np.random.randn(N, C)
print(y)
# y = np.array([-1.07296862, 0.86540763])
# print(y)

# w1 = np.random.randn(D, H)
w2 = np.random.randn(H, C)


# ReLU
def activation_function(sum):
    return max(0.0, sum)


# Sigmoid
def sigmoid(sum):
    return 1.0 / (1.0 + np.exp(-sum))


# vectorize для применения функции к ndarray
vectorized_activation_function = np.vectorize(activation_function)
vectorized_sigmoid = np.vectorize(sigmoid)


number_of_hidden_layers = 5
for i in range(number_of_hidden_layers):
    wn = np.random.randn(D, H)
    hidden_layer = x.dot(wn)
    print('\nHidden layer %s:' % (i + 1))
    print(hidden_layer)
    x = vectorized_activation_function(hidden_layer)
    #x = vectorized_sigmoid(hidden_layer)
    print('\nResult after activation function:')
    print(x)
    D = 4


# Абстрактный слой, от котроого наследуются все другие слои,
# Слои -- input hidden, функции активации можно тоже наследовать от абстрактного слоя
# Реализовать функции backward forward без реализации (пасс?)


y = x.dot(w2)
print('\nOutput:')
print(y)
