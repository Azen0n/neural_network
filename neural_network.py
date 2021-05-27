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
    def forward(self, layers):
        print('\nLayer 1 (input layer):')
        elements = layers[0].array
        print(elements)

        # Если используем L2, вызываем метод forward() с дополнительным параметром
        n = None
        if self.function_types[-1] == 'l2':
            n = -1

        for index, layer in enumerate(layers[1:n]):
            print('\nLayer %s (' % (index + 2), end="")
            elements = layer.forward(elements)

        if n:
            layers[-1].forward(elements, layers[-2].array)

        print('\nActual output:')
        print(layers[-2].array)

        return layers[-2].predicted_array

    # Обучение весов
    def train(self, number_of_epochs, learning_rate):
        # Веса для первого скрытого слоя
        self.layers[1].weights = np.array(
            [[np.random.uniform(0, 1) for _ in range(self.number_of_neurons)] for _ in
             range(self.number_of_input_elements)])

        losses = []
        epochs = []

        for epoch in range(number_of_epochs):
            # Проход вперед
            predicted_array = self.forward(self.layers)
            # Считаем ошибку для графика
            loss = activation_functions.l2(self.output_layer.predicted_array, self.output_layer.array).sum()
            upstream_gradient = 1.0

            # Если используем L2, вызываем метод backward() с дополнительным параметром
            n = None
            if self.function_types[-1] == 'l2':
                # В цикле по слоям, соответственно, пропускаем слой L2
                n = -1
                upstream_gradient, weights = self.layers[-1].backward(upstream_gradient, self.output_layer.array)

            for index, layer in enumerate(reversed(self.layers[1:n])):
                upstream_gradient, weights = layer.backward(upstream_gradient)

                # Если backward вернул веса, меняем их на месте
                if weights is not None:
                    self.layers[len(self.layers) - index - (2 if n else 1)].weights -= learning_rate * weights

            epochs.append(epoch)
            losses.append(loss)

        if self.function_types[-1] == 'l2':
            plt.plot(epochs, losses)
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.show()

    # Проверка точности
    def test(self, number_of_vectors):
        # Генерация входных и выходных данных при задаче классификации
        if self.problem_type == 'classification':
            test_input_layer = lrs.InputLayer(number_of_vectors, self.number_of_input_elements, 2)
            lrs.InputLayer.generate_classification(test_input_layer)

            test_output_layer = lrs.OutputLayer(number_of_vectors, self.number_of_output_elements)
            lrs.OutputLayer.generate_classification(test_output_layer, test_input_layer)

        # Генерация входных и выходных данных при задаче регрессии
        else:
            test_input_layer = lrs.InputLayer(number_of_vectors, self.number_of_input_elements, 2)
            lrs.InputLayer.generate_regression(test_input_layer)

            test_output_layer = lrs.OutputLayer(number_of_vectors, self.number_of_output_elements)
            lrs.OutputLayer.generate_regression(test_output_layer, test_input_layer)

        test_output_layer.weights = self.output_layer.weights

        test_layers = [test_input_layer]
        for i in range(len(self.layers) - 3):
            test_layers.append(self.layers[i + 1])
            test_layers[i + 1].number_of_vectors = number_of_vectors
        test_layers.append(test_output_layer)

        test_layers.append(
            lrs.ActivationFunction(number_of_vectors, self.number_of_input_elements, self.number_of_neurons,
                                   self.function_types[-1]))

        prediction = self.forward(test_layers)
        classes = test_layers[-1].array

        print('\nPredicted array (test): ')
        print(prediction)

        if self.function_types[-1] == 'l2':
            squared_error = np.power((test_layers[-2].array - test_layers[-2].predicted_array), 2).sum()
            mean_squared_error = squared_error / (number_of_vectors * self.number_of_input_elements)
            print('\nMean squared error: ')
            print(mean_squared_error)
        else:
            correct = 0.0
            for i in range(number_of_vectors):
                prediction_rounded = np.round(classes, 0)
                if prediction_rounded[i][0] == test_layers[-2].array[i][0]:
                    correct += 1

            accuracy = correct * 100.0 / number_of_vectors

            print('\nAccuracy: %.2f%%' % accuracy)
