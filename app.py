from neural.model import NeuralNetwork
from data_utils.data import Users
from flask import Flask, render_template

# Загрузим модель
our_network = NeuralNetwork()

# Загрузим данные
data = Users()

# Обучимся
our_network.train(data.get_train())

vectors_x = []
vectors_y = []
names = []
labels = []
for user in data.base.keys():
    names.append(user)
    x, y = [], []
    label = []
    for word in data.base[user].keys():
        label.append(word)
        voice = data[user][word]
        a = voice.copy()
        a.resize(100, 20)

        predicted = our_network.predict([[a]])[0][0]

        x.append(predicted[0])
        y.append(predicted[1])

    labels.append(label)

    vectors_x.append(x)
    vectors_y.append(y)

new_data = '['

for i in range(len(names)):
    new_data += 'l' + str(i) +','

new_data += ']'

app = Flask(__name__)


@app.route("/")
def main():
    return render_template('index.html', vectors_x=vectors_x,
                           vectors_y=vectors_y, names=names, labels=labels,
                           data=new_data)


if __name__ == "__main__":
    app.run()
