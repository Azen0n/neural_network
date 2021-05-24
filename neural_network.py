import numpy as np
import layers as lrs


class NeuralNetwork:
    def __init__(self, number_of_vectors, number_of_input_elements, number_of_output_elements, number_of_neurons,
                 number_of_hidden_layers, problem_type, function_types):
        self.number_of_vectors = number_of_vectors
        self.number_of_input_elements = number_of_input_elements
        self.number_of_output_elements = number_of_output_elements
        self.number_of_neurons = number_of_neurons
        self.number_of_hidden_layers = number_of_hidden_layers
        self.function_types = function_types
        self.problem_type = problem_type

        # Генерация входных и выходных данных при задаче классификации
        if problem_type == 'classification':
            self.input_layer = lrs.InputLayer(number_of_vectors, number_of_input_elements)
            lrs.InputLayer.generate_classification(self.input_layer)

            self.output_layer = lrs.OutputLayer(number_of_vectors, number_of_output_elements)
            lrs.OutputLayer.generate_classification(self.output_layer, self.input_layer)

        # Генерация входных и выходных данных при задаче регрессии
        else:
            self.input_layer = lrs.InputLayer(number_of_vectors, number_of_input_elements)
            lrs.InputLayer.generate(self.input_layer)

            self.output_layer = lrs.OutputLayer(number_of_vectors, number_of_output_elements)
            lrs.OutputLayer.generate(self.output_layer, self.input_layer)

        # Список всех слоев, включая слои входных, выходных данных и слои функций активации
        self.layers = list()
        self.layers.append(self.input_layer)
        for i in range(number_of_hidden_layers):
            self.layers.append(lrs.HiddenLayer(number_of_vectors, number_of_input_elements, number_of_neurons, i))
            self.layers.append(lrs.ActivationFunction(number_of_vectors, number_of_input_elements, number_of_neurons,
                                                      function_types[i]))
        self.layers.append(self.output_layer)

    def compute_forward(self):
        print('\nLayer 1 (input layer):')
        elements = self.input_layer.array
        print(elements)
        weights = np.random.randn(self.number_of_input_elements, self.number_of_neurons)

        for index, layer in enumerate(self.layers[1:-1]):
            print('\nLayer %s (' % (index + 2), end="")
            elements = layer.forward(elements, weights)
            # TODO: Если для каждого скрытого слоя задавать свое количество нейронов, то первый переметр должен быть
            #  self.prev.number_of_neurons, то есть придется добавить отдельное поле для хранения предыдущего слоя
            weights = np.random.randn(self.number_of_neurons, self.number_of_neurons)

        print('\nOutput:')
        print(self.layers[-2].array)
        print('\nActual output:')
        print(self.layers[-1].array)

        return self.layers[-2].array

    def compute_backward(self):
        upstream_gradient = 1.0
        # Если L2, то проходим вне цикла и начинаем с предпредпоследнего слоя,
        # иначе с предпоследнего
        n = -1
        if self.function_types[-1] == 'l2':
            n = -2
            upstream_gradient = self.layers[-2].backward(upstream_gradient, self.output_layer.array)

        for index, layer in reversed(list(enumerate(self.layers[1:n]))):
            upstream_gradient = layer.backward(upstream_gradient)

    # Первая версия
    def compute_v1(self):
        elements = self.input_layer.array
        elements_count = self.number_of_input_elements
        neurons_count = self.number_of_neurons

        for index, hidden_layer in enumerate(self.hidden_layers):
            weights = np.random.randn(elements_count, neurons_count)
            hidden_layer.array = elements.dot(weights)
            print('\nHidden layer %s:' % (index + 1))
            print(hidden_layer.array)

            elements = lrs.ActivationFunction.activate(self.activation_function, hidden_layer.array,
                                                       self.function_type)

            print('Result after activation function:')
            print(elements)
            elements_count = self.number_of_neurons

        weights = np.random.randn(elements_count, self.number_of_output_elements)

        self.output_layer.array = elements.dot(weights)
        print('\nOutput:')
        print(self.output_layer.array)
        return self.output_layer.array

    # Вторая версия
    def compute_v2(self):
        # Входные данные
        elements = self.input_layer.array
        # Случайные веса
        weights = np.random.randn(self.number_of_input_elements, self.number_of_neurons)

        # За каждый скрытый слой умножаем матрицы элементов и весов и применяем функцию активации,
        # результат записывается в ту же переменную elements, которая будет использоваться для следующего слоя
        # (либо будет являться конечным результатом)
        for index, hidden_layer in enumerate(self.hidden_layers):
            print('\nHidden layer %s:' % (index + 1))
            elements = hidden_layer.forward(elements, weights, self.activation_function, self.output_layer.array)
            # Снова генерируются случайные веса, в будущем здесь должно быть их обучение...
            weights = np.random.randn(self.number_of_input_elements, self.number_of_neurons)

        print('\nOutput:')
        print(self.hidden_layers[-1].array)
        return self.hidden_layers[-1].array
