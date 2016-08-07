import dill
from numpy.random import choice
from librosa import load, logamplitude
from librosa.feature import melspectrogram
import numpy as np


def get_spectgorgamm(path):
    '''Строим спектограмму из wav файла'''
    y, sr = load(path)
    S = melspectrogram(y, sr=sr, n_mels=100)
    log_S = logamplitude(S, ref_power=np.max)
    return log_S


class Users():
    def __init__(self):
        # Путь до dill модели
        self.path = 'users.dl'
        # Defaultdict информации
        self.base = dill.load(open(self.path, 'rb'))

    def __getitem__(self, item):
        return self.base[item]

    def save(self):
        ''' Поскольку это не стандартная DB тут нужна функция сохранения. '''
        dill.dump(self.base, open(self.path, 'wb'))

    def get_train(self):
        ''' Делаем из данных train выборку '''
        train = []
        for key in self.base.keys():
            if len(self.base[key]) >= 2:
                for _ in range(0, len(self.base[key]), 2):
                    values = choice(list(self.base[key]), 2)
                    other = choice(list(self.base), 1)[0]
                    other_value = choice(list(self.base[other]), 1)[0]
                    train.append((key, self.base[key][values[0]],
                                  other, self.base[other][other_value], key, self.base[key][values[1]]))

        return train

    def add_wav(self, user, word):
        '''Сохраняем спектограму для пользователя'''
        self.base[user][word] = get_spectgorgamm('/tmp/file.wav')

    def get_train_for_user(self, user, shape = (100,20)):
        ''' Трейн данные для одного пользователя '''
        train = []
        if len(self.base[user]) >= 2:
            for _ in range(0, len(self.base[user]), 2):
                values = choice(list(self.base[user]), 2)
                
                value_first = np.asarray(self.base[user][values[0]])
                value_second = np.asarray(self.base[user][values[1]]
                
                other = choice(list(self.base), 1)[0]
                other_value = np.asarray(self.base[other][choice(list(self.base[other]), 1)[0]])
                
                np.resize(other_value, shape)
                np.resize(value_first, shape)
                np.resize(value_second, shape)
                                          
                train.append((user, value_first,
                              other, other_value, user, value_second))
        else:
            raise ValueError
