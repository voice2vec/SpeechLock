from numpy.random import choice

def GetTrain(data):
    '''Делаем из данных train выборку'''
    train = []
    for key in data.keys():
        for _ in range(0, len(data[key]), 2):
            values = choice(data[key], 2)
            other = choice(data.keys(), 1)[0]
            other_value = choice(data[other], 1)[0]
            train.append((key, values[0], other, other_value, key, values[1]))

    return train