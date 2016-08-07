from dill import dump, load
from numpy.random import choice


class Users():
    def __init__(self):
        # Путь до dill модели
        self.path = 'users.dl'
        # Defaultdict информации
        self.base = load(open(self.path, 'rb'))

    def __getitem__(self, item):
        return self.base[item]

    def save(self):
        ''' Поскольку это не стандартная DB тут нужна функция сохранения. '''
        dump(self.base, open(self.path, 'wb'))

    def get_train(self):
        '''Делаем из данных train выборку'''
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
