import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import pdist, squareform, euclidean
from scipy.sparse.linalg import eigsh
from tqdm import tqdm
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, linear_kernel
from sklearn.preprocessing import KernelCenterer
import cvxpy as cp


class Kernel(object):
    """
    This is a base Kernel class (acting as MixIn).
    It is not supposed to be directly initialized, but should be inherited from.

    """

    def __init__(self, kernel_type="linear", degree=2, gamma=None, coef0=1):

        self.kernel_type = kernel_type
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0

        self.centerer = KernelCenterer()

    def c_(self, X):
        """
        Center the gram matrix
        """
        return self.centerer.fit_transform(X)

    def apply_kernel(self, X):
        kernel_handler = {"rbf": self._apply_rbf,
                          "linear": self._apply_linear,
                          "poly": self. _apply_poly}
        return self.c_(kernel_handler[self.kernel_type](X))

    def _apply_linear(self, X):
        return linear_kernel(X)

    def _apply_poly(self, X):
        return polynomial_kernel(X, degree=self.degree, coef0=self.coef0, gamma=self.gamma)

    def _apply_rbf(self, X):
        return rbf_kernel(X, gamma=self.gamma)


class PCA(Kernel):
    """
    Kernelized version of principal compnent analysis.
    If kernel is linear, equivalent to classic PCA.
    """

    def __init__(self, n_components, kernel_type="linear", degree=2, gamma=1.5, coef0=0):
        super(PCA, self).__init__(kernel_type, degree, gamma, coef0)
        self.n_components = n_components

    def fit_transform(self, X):

        self.K = self.apply_kernel(X)

        self.lambdas_, self.alphas_ = eigsh(self.K,
                                            self.n_components,
                                            which="LA")

        indices = self.lambdas_.argsort()[::-1]
        self.lambdas_ = self.lambdas_[indices]
        self.alphas_ = self.alphas_[:, indices]

        X_transformed = self.alphas_ * np.sqrt(self.lambdas_)
        return X_transformed


class KMeans(Kernel):
    """
    Kernelized version of KMeans clustering.
    """

    def __init__(self, n_clusters, kernel_type="linear", degree=2, gamma=1.5, coef0=0):
        super(KMeans, self).__init__(kernel_type, degree, gamma, coef0)
        self.n_clusters = n_clusters

    def fit(self, X, num_iter=100):

        self.X = self.apply_kernel(X)
        self.centroids = self.X[np.random.randint(
            0, self.X.shape[0], size=self.n_clusters), :]
        self.clusters = np.random.randint(0, 3, size=self.X.shape[0])

        for _ in range(num_iter):
            for i in range(self.X.shape[0]):

                self.clusters[i] = np.argmin(
                    np.linalg.norm(self.X[i, :] - self.centroids, axis=1))

            for j in range(self.centroids.shape[0]):
                self.centroids[j] = np.mean(
                    self.X[self.clusters == j, :], axis=0)
        return self

    def transform(self):
        if not hasattr(self, "centroids"):
            raise ValueError("The classifier is not fitted")
        return self.clusters, self.centroids

    def fit_transform(self, X, num_iter=100):

        self.fit(X, num_iter)
        return self.transform()

    def plot(self):
        colors = ['r', 'g', 'b', 'y', 'c', 'm']

        if self.X.shape[1] > 2:
            raise ValueError("Cannot plot data  with dimentions more than 2")

        for i in range(self.n_clusters):
            plt.scatter(self.X[self.clusters == i, 0],
                        self.X[self.clusters == i, 1], c=colors[i])

        plt.show()


class LogisticRegression(Kernel):
    """
    Kernelized logistic regression.
    """

    def __init__(self, lr=0.01, fit_bias=True, kernel_type='linear', degree=2, gamma=1.5, coef0=1):
        super(LogisticRegression, self).__init__(
            kernel_type, degree, gamma, coef0)
        self.lr = lr
        self.fit_bias = fit_bias

    def _init_bias(self):
        bias = np.ones((self.X_.shape[0], 1))
        self.X_ = np.concatenate((bias, self.X_), axis=1)

    def fit(self, X, y, num_iter=300):
        self.X_ = self.apply_kernel(X)
        self.y = y
        if self.fit_bias:
            self._init_bias()
        self.W = np.zeros((self.X_.shape[1]))

        for _ in range(num_iter):

            dw = (self.y - self.predict()).dot(self.X_) * (1/self.y.shape[0])

            self.W += self.lr*dw
        return self

    @property
    def cost(self):
        y_hat = self.predict()
        cost = -self.y.T.dot(np.log(y_hat)) - (1-self.y).T.dot(np.log(1-y_hat))
        return cost.mean()

    @property
    def accuracy(self):
        y_hat = self.predict()
        y_hat[y_hat >= 0.5] = 1
        y_hat[y_hat < 0.5] = 0
        return accuracy_score(self.y, y_hat)

    def predict_proba(self):
        Z = self.X_.dot(self.W)
        return self.sigmoid(Z)

    def predict(self):
        y_hat = self.predict_proba()
        y_hat[y_hat >= 0.5] = 1
        y_hat[y_hat < 0.5] = 0
        return y_hat

    def sigmoid(self, Z):
        return (1.0/(1 + np.exp(-Z)))

    def plot(self):
        if not hasattr(self, "W"):
            raise ValueError("Classifier is not fitted")
        plt.scatter(self.X_[self.y == 0, 1], self.X_[self.y == 0, 2], c="r")
        plt.scatter(self.X_[self.y == 1, 1], self.X_[self.y == 1, 2], c="b")
        xx = np.linspace(-4, 4)
        a = -self.W[1] / self.W[2]
        yy = a * xx - self.W[0] / self.W[2]
        plt.plot(xx, yy)
        plt.show()


class SVDD(Kernel):
    """
    Kernelized Support Vector Data Descriptor. 
    Some different formulations of this algorithm also known as One Class SVM.
    """

    def __init__(self, C=None, kernel_type='linear', degree=2, gamma=1.5, coef0=1):
        super(SVDD, self).__init__(kernel_type=kernel_type,
                                   degree=degree, gamma=gamma, coef0=coef0)
        self.C = C

    def fit(self, X, y=None):
        self.X = X
        self.G = self.apply_kernel(X)  # get a gramm matrix
        n = X.shape[0]
        self.alpha = cp.Variable(n)
        G = cp.Parameter(shape=self.G.shape, value=self.G, PSD=True)
        e = np.ones(n)
        self.C = cp.Parameter(nonneg=True, value=self.C)

        left_part = cp.quad_form(self.alpha, G)
        right_part = self.alpha.T@cp.diag(G)
        loss = left_part - right_part

        constraints = [e@self.alpha == 1, 0 <=
                       self.alpha, self.alpha <= self.C]

        problem = cp.Problem(cp.Minimize(loss), constraints)

        problem.solve(verbose=True)

        return self

    @property
    def radius(self):
        k = 0
        xkxk = self.support_vectors[k, :].dot(self.support_vectors[k, :])

        def sum_one(
            i): return self.alpha.value[i]*(self.G[i, :].dot(self.support_vectors[k, :]))
        sum_one = 2 * \
            np.sum(sum_one([i for i in range(self.alpha.value.shape[0])]))
        sum_two = self.alpha.value.dot(self.alpha.value@self.G)
        return xkxk - sum_one + sum_two

    @property
    def support_vectors(self):
        if not hasattr(self, "alpha"):
            raise AttributeError("No alpha, run fit first")
        return self.G[self.alpha.value < self.C.value, :]

    @property
    def center(self):
        return self.alpha.value.reshape(1, -1)@self.G

    def predict(self, X):
        G = self.apply_kernel(X)

        dist = np.zeros(G.shape[0])
        for i in range(G.shape[0]):
            dist[i] = euclidean(G[i, :].reshape(1, -1), self.center)

        print(dist)
        print(self.radius**2)
        return dist >= self.radius**2
