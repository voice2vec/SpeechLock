import theano.tensor as T

from lasagne.layers import InputLayer, DenseLayer, \
    ReshapeLayer, Conv1DLayer, MaxPool1DLayer, \
    GlobalPoolLayer, get_output, get_all_params

from lasagne.nonlinearities import very_leaky_rectify, sigmoid

from lasagne.updates import adagrad

import theano


class NeuralNetwork:
    def __init__(self):
        num_units = 150  # выходной вектор будет в этом пространстве

        # Входной вектор (Кол-во батчей, кол-во записей, время, частота)
        input_X = T.tensor4("X")

        # Входной слой
        input_layer = InputLayer(shape=(None, None, 20, 100), input_var=input_X.swapaxes(2, 3))

        # Сворачиваем все записи друг за другом
        input_reshape = ReshapeLayer(input_layer, (-1, 20, 100))

        convolution = Conv1DLayer(input_reshape, num_filters=200, filter_size=5,
                                  nonlinearity=very_leaky_rectify)

        pooling = MaxPool1DLayer(convolution, 2)

        global_pooling = GlobalPoolLayer(pooling)

        dense = DenseLayer(global_pooling, num_units=300)

        output_dense = DenseLayer(dense, num_units=num_units, nonlinearity=sigmoid,
                                  name='output')

        train_output = ReshapeLayer(output_dense, (-1, 3, num_units))

        to_vector = ReshapeLayer(nn, (-1, 1, num_units))

        # предсказание нейронки (theano-преобразование)
        train_predicted = get_output(train_output)  # for loss

        vector = get_output(to_vector)  # for use

        # все веса нейронки (shared-переменные)
        all_weights = get_all_params(output_dense, trainable=True)

        # функция ошибки
        loss = (T.maximum(T.sum(T.sqr(train_predicted[:, 0] - train_predicted[:, 1])) + 1e-2, 0)
                - T.maximum(T.sum(T.sqr(train_predicted[:, 0] - train_predicted[:, 2])) + 1e-2, 0))

        # сразу посчитать словарь обновлённых значений с шагом по градиенту, как раньше
        updates_sgd = adagrad(loss, all_weights, learning_rate=0.001)

        # функция, которая обучает сеть на 1 шаг и возвращащет значение функции потерь и точности
        self.train = theano.function([input_X], [loss], updates=updates_sgd)

        # функция, которая считает точность
        self.predict = theano.function([input_X], vector)
