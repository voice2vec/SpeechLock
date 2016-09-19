import theano
import theano.tensor as T

from lasagne.layers import InputLayer, DenseLayer, ReshapeLayer, Conv1DLayer, MaxPool1DLayer, GlobalPoolLayer, \
    get_output, get_all_params, get_all_param_values, set_all_param_values

from lasagne.nonlinearities import very_leaky_rectify, tanh

from lasagne.updates import adagrad

from lasagne.layers import Layer


class Speech2VecLayer(Layer):
    def __init__(self, incoming, sound_shape, num_units, **kwargs):
        """
        :param incoming: the layer feeding into this layer, or the expected input shape.
        :param sound_shape: shape of freq x time
        :param num_units: output vector dimension
        """

        super().__init__(incoming, **kwargs)

        input_reshape = ReshapeLayer(incoming, (-1,) + sound_shape)  # Сворачиваем все записи друг за другом
        convolution = Conv1DLayer(input_reshape, num_filters=100, filter_size=5, nonlinearity=very_leaky_rectify)
        pooling = MaxPool1DLayer(convolution, 2)
        global_pooling = GlobalPoolLayer(pooling)
        dense = DenseLayer(global_pooling, num_units=300)
        output_dense = DenseLayer(dense, num_units=num_units, nonlinearity=tanh, name='output')
        all_vectors_output = ReshapeLayer(output_dense, (-1, 3, num_units))

        self.layers = (input_reshape, convolution, pooling, global_pooling, dense, output_dense, all_vectors_output)

        self.get_output_for = all_vectors_output.get_output_for
        self.get_output_shape_for = all_vectors_output.get_output_shape_for
        self.get_params = all_vectors_output.get_params


class BaseNeuralNetwork:
    def __init__(self, sound_shape, num_units, main_layer_class, loss_func, updates_func):
        # входной тензор (кол-во батчей, кол-во записей, время, частота)
        input_X = T.tensor4("X")

        # сеть
        input_layer = InputLayer(shape=(None, 3) + sound_shape, input_var=input_X.swapaxes(2, 3))
        all_output = main_layer_class(input_layer, sound_shape, num_units)  # for loss
        vector_output = ReshapeLayer(all_output, (-1, 1, num_units))  # for use

        # предсказание нейронки
        all_predicted = get_output(all_output)  # for loss
        vector_predicted = get_output(vector_output)  # for use

        # функция ошибки
        loss = loss_func(all_predicted)

        # посчитать обновлённые веса с шагом по градиенту
        trainable_weights = get_all_params(all_output, trainable=True)
        updates_sgd = updates_func(loss, trainable_weights)

        # функция, которая обучает сеть на 1 шаг и возвращащет значение функции потерь
        self.fit = theano.function([input_X], loss, updates=updates_sgd)

        # функция, которая возвращает вектор голоса
        self.predict = theano.function([input_X], vector_predicted)

        self.all_output = all_output
        self.vector_output = vector_output
        self.all_predicted = all_predicted
        self.vector_predicted = vector_predicted

    def get_params(self, **tags):
        return get_all_params(self.all_output, **tags)

    def get_params_values(self, **tags):
        return get_all_param_values(self.all_output, **tags)

    def set_params_values(self, values, **tags):
        set_all_param_values(self.all_output, values, **tags)


class NeuralNetwork(BaseNeuralNetwork):
    def __init__(self, num_units=2):
        """
        :param num_units: output vector dimension
        """

        def loss_func(all_predicted):
            def distance_sq(x1, x2):
                return T.sum(T.sqr(x1 - x2))

            d1 = distance_sq(all_predicted[:, 0], all_predicted[:, 1])
            d2 = distance_sq(all_predicted[:, 0], all_predicted[:, 2])
            alpha = 1e-2

            return T.maximum(d1 + alpha, 0) - T.maximum(d2 + alpha, 0)

        super().__init__(
            sound_shape=(20, 100),
            num_units=num_units,
            main_layer_class=Speech2VecLayer,
            loss_func=loss_func,
            updates_func=lambda loss, weights: adagrad(loss, weights, learning_rate=0.001)
        )
