from neural.model import NeuralNetwork
from data_utils.data import Users

# Загрузим модель
our_network = NeuralNetwork()

# Загрузим данные
train_data = Users().get_train()

# Обучимся
our_network.train(train_data)