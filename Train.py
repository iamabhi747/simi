from .NeuralNetwork import NeuralNetwork
from . import cost
import numpy as np

class Train:
    def __init__(self, nn: NeuralNetwork, lr=0.1, cf=None) -> None:
        self.nn = nn
        self.lr = lr
        self.cf = cf if cf is not None and isinstance(cf, cost.CostFunction) else cost.MSE

    def train_batch(self, Xs, Ys, threshold=0.5):
        assert len(Xs) == len(Ys)
        i = 0
        while True and i < 1000:
            c = sum([self.cf(Y, self.nn.forward(X)) for X, Y in zip(Xs, Ys)])

            print(f"\rCost[{i}]: ", c, end="")
            if c < threshold:
                break

            for X, Y in zip(reversed(Xs), reversed(Ys)):
                self.nn.backward(X, Y, self.cf)

            self.nn.update(self.lr)
            i += 1
        print()

    def train(self, data, batch_size, epochs:int=1, test_data_percentage:float=0.1):
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
                self.train_batch(Xs, Ys)

            Xs, Ys = zip(*test_data)
            Xs = np.array(Xs)
            Ys = np.array(Ys)
            c = 0.0
            for j in range(len(Xs)):
                c += self.cf(self.nn.forward(Xs[j]), Ys[j])
            c /= len(Xs)
            print(f"# Epoch {_+1} done {c}")
            self.nn.flush_cache()