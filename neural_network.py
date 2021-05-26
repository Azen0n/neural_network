import numpy as np
import layers as lrs
import activation_functions
from matplotlib import pyplot as plt


class NeuralNetwork:
    def __init__(self, number_of_vectors, number_of_input_elements, number_of_output_elements, number_of_neurons,
                 number_of_hidden_layers, problem_type, function_types, seed):
        self.number_of_vectors = number_of_vectors
        self.number_of_input_elements = number_of_input_elements
        self.number_of_output_elements = number_of_output_elements
        self.number_of_neurons = number_of_neurons
        self.number_of_hidden_layers = number_of_hidden_layers
        self.function_types = function_types
        self.problem_type = problem_type

        # Генерация входных и выходных данных при задаче классификации
        if problem_type == 'classification':
            self.input_layer = lrs.InputLayer(number_of_vectors, number_of_input_elements, seed)
            lrs.InputLayer.generate_classification(self.input_layer)

            self.output_layer = lrs.OutputLayer(number_of_vectors, number_of_output_elements)
            lrs.OutputLayer.generate_classification(self.output_layer, self.input_layer)

        # Генерация входных и выходных данных при задаче регрессии
        else:
            self.input_layer = lrs.InputLayer(number_of_vectors, number_of_input_elements, seed)
            lrs.InputLayer.generate_regression(self.input_layer)

            self.output_layer = lrs.OutputLayer(number_of_vectors, number_of_output_elements)
            lrs.OutputLayer.generate_regression(self.output_layer, self.input_layer)

        # Список всех слоев, включая слои входных, выходных данных и слои функций активации
        self.layers = list()
        self.layers.append(self.input_layer)
        for i in range(number_of_hidden_layers):
            self.layers.append(lrs.HiddenLayer(number_of_vectors, number_of_input_elements, number_of_neurons, i))
            self.layers.append(lrs.ActivationFunction(number_of_vectors, number_of_input_elements, number_of_neurons,
                                                      function_types[i]))

        # Выходной слой
        self.layers.append(self.output_layer)
        # Функция потерь
        self.layers.append(lrs.ActivationFunction(number_of_vectors, number_of_input_elements, number_of_neurons,
                                                  function_types[-1]))
        # Веса выходного слоя генерируются здесь
        lrs.OutputLayer.generate_weights(self.output_layer, self.layers[-4].number_of_neurons)

    # Проход вперед, возвращает предсказанные данные
    def forward(self):
        print('\nLayer 1 (input layer):')
        elements = self.input_layer.array
        print(elements)

        # Если используем L2, вызываем метод forward() с дополнительным параметром
        n = None
        if self.function_types[-1] == 'l2':
            n = -1

        for index, layer in enumerate(self.layers[1:n]):
            print('\nLayer %s (' % (index + 2), end="")
            elements = layer.forward(elements)

        if n:
            self.layers[-1].forward(elements, self.output_layer.array)

        print('\nActual output:')
        print(self.layers[-2].array)

        return self.layers[-2].predicted_array

    # Обучение весов
    def train(self, number_of_epochs, learning_rate):
        # Веса для первого скрытого слоя
        self.layers[1].weights = np.random.randn(self.number_of_input_elements, self.number_of_neurons)

        losses = []
        epochs = []

        for epoch in range(number_of_epochs):
            # Проход вперед
            predicted_array = self.forward()
            # Считаем ошибку для графика
            loss = activation_functions.l2(self.output_layer.predicted_array, self.output_layer.array).sum()
            upstream_gradient = 1.0

            # Если используем L2, вызываем метод backward() с дополнительным параметром
            n = None
            if self.function_types[-1] == 'l2':
                # В цикле по слоям, соответственно, пропускаем слой L2
                n = -1
                upstream_gradient = self.layers[-1].backward(upstream_gradient, self.output_layer.array)

            for index, layer in enumerate(reversed(self.layers[1:n])):
                upstream_gradient = layer.backward(upstream_gradient)

                # Если у слоя есть веса (скрытый и выходной), то мы меняем их на месте
                if hasattr(layer, 'weights'):
                    self.layers[len(self.layers) - index - 2].weights -= learning_rate * upstream_gradient

            epochs.append(epoch)
            losses.append(loss)

        plt.plot(epochs, losses)
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.show()
