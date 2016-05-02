import random
import numpy as np
import math
from numpy import linalg as LA

class DualLRclassifier(object):


    def __init__(self, C, num_feats, num_samples):
        self.C = C
        self.num_feats = num_feats
        self.num_samples = num_samples
        self.gradient = np.zeros(self.num_feats)
        self.accumulated_gradients = np.zeros(self.num_feats)


    def init_lambdas(self):
        self.lambdas = []
        for i in range(0, self.num_samples):
            eps1 = random.uniform(0.001, 0.025)
            eps2 = random.uniform(0.001, 0.025)
            self.lambdas.append(min(eps1 * self.C, eps2))
        self.lambdas = np.asarray(self.lambdas)

    def init_weights(self, X, y):
        ret = (X.T * self.lambdas * y).T
        self.weights = np.sum(ret, axis = 0)

    def start(self, X, y):
        self.init_lambdas()
        self.init_weights(X, y)

        self.prime_lambdas = self.C - self.lambdas
        self.Q = np.zeros((X.shape[0], X.shape[0]))
        for i in range(X.shape[0]):
            self.Q[i][i] = np.dot(X[i].T, X[i])

    def construct_subproblem(self, x, y, i):
        c1 = self.lambdas[i]
        c2 = self.prime_lambdas[i]
        a = self.Q[i][i]
        b = y * np.dot(self.weights.T, x)
        return a, b, c1, c2

    def g_prime(self, Zt, a, b, ct, s):
        return math.log(Zt/(s - Zt)) + a * (Zt - ct) + b

    def g_double_prime(self, Zt, a, b, ct, s):
        return a + s / (Zt * (s - Zt))

    def modified_newton(self, a, b, c1, c2, n_iter=10, eps=0.025):
        # Algorithm 4
        c = [c1, c2]
        bt = [b, -b]
        s = c1 + c2
        zm = (c2 - c1)/2.0
        if zm >= -b/a:
            t = 0
        else:
            t = 1
        Z = np.random.uniform(0, s, size=2)
        for k in range(n_iter):
            if self.g_prime(Z[t], a, b, c[t], s) == 0:
                break
            d = - self.g_prime(Z[t], a, bt[t], c[t], s) / self.g_double_prime(Z[t], a, bt[t], c[t], s)
            # update Zt
            if Z[t] + d <= 0:
                Z[t] *= eps
            else:
                Z[t] += d
        if t == 0:
            Z[1] = s - Z[0]
        else:
            Z[0] = s - Z[1]
        return Z

    def update_rule(self, x, y, i, Z):
        self.weights += (Z[0] - self.lambdas[i]) * y * x
        self.lambdas[i] = Z[0]
        self.prime_lambdas[i] = Z[1]

    # truncate big numbers
    def exp(self, x):
        if x > 100:
            return math.exp(100)
        else:
            return math.exp(x)

    def logistic_function(self, t):
        return 1.0 / (1 + self.exp(t))

    def reset_accum_gradient(self):
        self.accumulated_gradients = np.zeros(self.num_feats)

    def train(self, Xtrain, Ytrain):
        self.reset_accum_gradient()
        for i, (x, y) in enumerate(zip(Xtrain, Ytrain)):
            a, b, c1, c2 = self.construct_subproblem(x, y, i)
            Z = self.modified_newton(a, b, c1, c2, n_iter=100)
            self.update_rule(x, y, i, Z)
            # compute gradient
            self.gradient =  self.logistic_function(y * np.dot(x, self.weights)) * (-1.0 * y) * x
            # print self.gradient
            self.accumulated_gradients -= self.gradient
            # print LA.norm(self.accumulated_gradients, 2)


    def predict(self, samples):
        # backwards cause it is the dual and we need to maximize
        ret = []
        for x in samples:
            t = np.dot(x, self.weights)
            prob_p = self.logistic_function(t)
            if prob_p > .5:
                ret.append(-1)
            else:
                ret.append(1)
        return np.asarray(ret)
