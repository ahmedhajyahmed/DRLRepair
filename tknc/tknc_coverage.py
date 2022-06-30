from tensorflow.keras import backend as K
import numpy as np
from tensorflow import keras


# helper function
def get_layer_i_output(model, layer, data):
    layer_model = K.function([model.layers[0].input], [layer.output])
    ret = layer_model([data])[0]
    num = data.shape[0]
    ret = np.reshape(ret, (num, -1))
    return ret


class Coverage:
    def __init__(self, model=None, x_train=None, y_train=None, top_k=2):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.pattern_set = set()
        self.top_k = top_k

    def TKNC(self, data, layer):

        layer_output = get_layer_i_output(self.model, layer, data)
        topk = np.argpartition(layer_output, -self.top_k, axis=1)[:, -self.top_k:]
        topk = np.sort(topk, axis=1)
        for j in range(topk.shape[0]):
            for z in range(self.top_k):
                a = str(layer.name)
                b = str(topk[j][z])
                self.pattern_set.add(str(layer.name) + "+" + str(topk[j][z]))

    def calculate_metrics(self, data, batch=1024):
        data_num = data.shape[0]
        for i in self.model.layers:
            begin, end = 0, batch
            while begin < data_num:
                self.TKNC(data[begin:end], i)
                begin += batch
                end += batch


def main():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = np.reshape(x_train, x_train.shape + (1,))
    x_test = np.reshape(x_test, x_test.shape + (1,))

    model = keras.Sequential()
    # model.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='relu'))
    # model.add(keras.layers.Flatten())
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    model.add(keras.layers.Dense(24, activation='relu'))
    model.add(keras.layers.Dense(24, activation='relu'))
    model.add(keras.layers.Dense(10, activation='linear'))
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    for i in range(6):
        model.fit(x_train, y_train, epochs=1, verbose=2)

    coverage = Coverage(model, x_train=x_train, y_train=y_train)
    coverage.calculate_metrics(x_test)
    a = coverage.pattern_set


# if __name__ == '__main__':
#     # main()
#     pass




