import numpy as np
from .NeuralNetwork import NeuralNetwork
from .CostFunction import CostFunction

class Train:
    def __init__(self, nn:NeuralNetwork, cf:CostFunction, lr:int) -> None:
        self.nn = nn
        self.lr = lr
        self.cf = cf

    def train_labeled_batch(self, Xs, Ys, update_period:int=5):
        assert len(Xs) == len(Ys)
        k = 20
        while k > 0:
            for i in range(len(Xs)):
                dC_dA = self.cf.derv(self.nn.forward(Xs[i]), Ys[i])
                self.nn.backward(dC_dA)
            self.nn.update(self.lr)
            k -= 1

    def train(self, data, batch_size, epochs:int=1, update_period:int=5, test_data_percentage:float=0.1):
        test_data = data[:int(test_data_percentage*len(data))]
        data      = data[int(test_data_percentage*len(data)):]
        extra = len(data) - (len(data) // batch_size) * batch_size
        if extra > 0: extra = batch_size - extra
        for i in np.random.randint(0, len(data), extra):
            data.append(data[i])

        for _ in range(epochs):
            np.random.shuffle(data)
            for i in range(0, len(data), batch_size):
                batch = data[i:i+batch_size]
                Xs, Ys = zip(*batch)
                Xs = np.array(Xs)
                Ys = np.array(Ys)
                self.train_labeled_batch(Xs, Ys, update_period)
                self.nn.update(self.lr)

            Xs, Ys = zip(*test_data)
            Xs = np.array(Xs)
            Ys = np.array(Ys)
            cost = 0.0
            for j in range(len(Xs)):
                cost += self.cf(self.nn.forward(Xs[j]), Ys[j])
            cost /= len(Xs)
            print(f"# Epoch {_+1} done {cost}")

        