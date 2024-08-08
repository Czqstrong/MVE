import numpy as np
import matplotlib.pyplot as plt
import time

np.random.seed(1)

class ToySystem(object):
    def __init__(self, sigma=np.sqrt(2) * np.identity(2)):
        self.dim = 2
        self.sigma = sigma
        self.range = [-2, 2]

    def get_f(self, X):

        x1 = X[:, 0]
        x2 = X[:, 1]

        F1 = -4*x1*(x1**2-1)-10*(x1-np.mean(x1))
        F2 = -2*x2-10*(x2-np.mean(x2))

        return np.stack([F1, F2], axis=-1)


Toy = ToySystem()


def get_uniform_data(N):
    data = np.zeros(dtype=np.float64, shape=(N, 2))
    for i in range(Toy.dim):
        data[:, i] = np.random.uniform(Toy.range[0], Toy.range[1], N)
    return data


def sample_equili_data(N, dt=1e-2, T=10, sigma=Toy.sigma, m0=0, m=100, k=1):
    X = []
    x = get_uniform_data(N)
    for i in range(np.int_(T / dt * k)):
        if i >= m0 * k and i % (m * k) == 0:
            X.append(x + 0.)
            if i % m0 == 0:
                print(i)

        x = x + Toy.get_f(x) * dt / k + np.sqrt(dt / k) * (np.random.normal(0, 1, x.shape) @ np.transpose(sigma))

        if (i + 1) % np.int_(T / dt * k / 10) == 0:
            print((i + 1) / np.int_(T / dt * k / 10))
            fig1, ax = plt.subplots(1, 3, figsize=(9, 3))
            ax[0].hist(x[:, 0], bins=100, density=True)
            ax[1].hist(x[:, 1], bins=100, density=True)
            ax[2].scatter(x[:,0],x[:,1],s=0.2)
            plt.tight_layout()
            name = np.int_((i + 1) / np.int_(T / dt * k / 10))
            plt.savefig(str(name) + ".pdf")
            plt.show()

        if i == m0 * k:
            print('========================================')
            print(i)
    return np.reshape(X, (-1, Toy.dim))


tt = time.time()
equili_data = sample_equili_data(N=1000, dt=1e-3, T=100000, m0=100000, m=1000)
print(time.time() - tt)
np.save('dw_true.npy', equili_data)
