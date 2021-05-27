import numpy as np
import neural_network as nn

# Функции активации:
# relu, sigmoid, tanh, leaky relu, softmax, l2

# Типы задач:
# regression, classification

# Модель для задачи регрессии
neural_network_regression = nn.NeuralNetwork(
    number_of_vectors=20,
    number_of_input_elements=4,
    number_of_output_elements=4,
    number_of_neurons=3,
    number_of_hidden_layers=2,
    problem_type='regression',
    function_types=['leaky relu', 'leaky relu', 'l2'],
    seed=1)

# neural_network_regression.train(number_of_epochs=100, learning_rate=0.0001)
# neural_network_regression.test(number_of_vectors=10)


# Модель для задачи классификации
neural_network_classification = nn.NeuralNetwork(
    number_of_vectors=400,
    number_of_input_elements=2,
    number_of_output_elements=2,
    number_of_neurons=5,
    number_of_hidden_layers=1,
    problem_type='classification',
    function_types=['leaky relu', 'softmax'],
    seed=1)

neural_network_classification.train(number_of_epochs=300, learning_rate=0.00000001)
neural_network_classification.test(number_of_vectors=400)
