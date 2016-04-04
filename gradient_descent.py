import numpy as np
import math

''' Gradient Descent for the primal version of logistic regression '''

class GradientDescent:

    def __init__(self, n_iter, step_size, lmbd):
        self.n_iter = n_iter
        self.step_size = step_size
        self.lmbd = lmbd

    # truncate big numbers
    def exp(self, x):
        if x > 45:
            return math.exp(45)
        else:
            return math.exp(x)

    def logistic_function(self, t):
        return 1.0 / (1 + self.exp(-1.0 * t))

    def update_gradients(self, x, y):
        # compute gradient
        t = np.dot(x, self.weights)
        gradient =  -1.0 * self.logistic_function(-1.0 * y * t) * y * x
        # take a step down
        self.gradients -= self.step_size * gradient

    def l2_update(self):
        # regularization
        self.gradients += self.lmbd * self.weights

    def update_weights(self):
        # update rule
        self.weights += (1.0/self.num_samples) * self.gradients

    def objective_function(self, samples, labels):
        nll = self.neg_log_likelihood(samples, labels)
        return nll + 1/2 * self.lmbd * np.dot(self.weights, self.weights)

    def neg_log_likelihood(self, samples, labels):
        neg_log_likelihood = 0.0
        for x, y in range(samples, labels):
            t = np.dot(x, self.weights)
            neg_log_likelihood -= math.log(1 + self.exp(-1.0 * y * t))
        return neg_log_likelihood

    def fit(self, samples, labels):
        self.num_feats = samples.shape[1]
        self.num_samples = samples.shape[0]

        # initialize weights: w <- 0
        self.weights = np.zeros(self.num_feats)

        for n in xrange(self.n_iter):
            self.gradients = np.zeros(self.num_feats)

            for x, y in zip(samples, labels):
                self.update_gradients(x, y)

            self.l2_update()
            self.update_weights()
            #print "Iter:", n, "objective function:", self.objective_function(samples, labels)

    def predict(self, samples):
        ret = []
        for x in samples:
            t = np.dot(x, self.weights)
            prob_p = self.logistic_function(t)
            if prob_p > .5:
                ret.append(1)
            else:
                ret.append(-1)
        return np.asarray(ret)
