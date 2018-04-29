import numpy as np
import pandas as pd


class GHA:
    df = pd.read_fwf("training data path")        # try np.genfromtxt() if this didn't work well with you
    df2 = pd.read_fwf("testing data path")
    x = df.as_matrix() #training data
    test = df2.as_matrix() #tesring data

    np.random.seed(0) # To keep the random numbers the same every run
    desired_pca_features = 100
    w = np.random.randn(desired_pca_features, x.shape[1]) * 0.01

    def forward (self, w, data):
        z = np.dot(w, data.T)
        return z

    def update_w(self, net, lr):
        new_w = ((np.dot(net, self.x)) - (np.dot(net, np.dot(self.w.T, net).T))) * lr
        return new_w

    def train(self, lr, epoch):
        for i in range(epoch):
            net = self.forward(self.w, self.x)
            new_w = self.update_w(net, lr)
            self.w += new_w * lr


if __name__ == '__main__':
    H = GHA()
    H.train(.0000001, 400)
    pca_training = H.forward(H.x, H.w)
    pca_testing = H.forward(H.test, H.w)

