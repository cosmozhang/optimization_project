import numpy as np
import math

''' Gradient Descent for the primal version of logistic regression '''

class LRclassifier(object):

    def __init__(self, step_size, lmbd, num_feats, num_samples):
        self.step_size = step_size
        self.lmbd = lmbd
        self.num_feats = num_feats
        self.num_samples = num_samples

        # initialize weights: w <- 0
        self.weights = np.zeros(self.num_feats)
        self.gradients = np.zeros(self.num_feats)

    # truncate big numbers
    def exp(self, x):
        if x > 100:
            return math.exp(100)
        else:
            return math.exp(x)

    def logistic_function(self, t):
        return 1.0 / (1 + math.exp(t))

    def update_gradients(self, Xtrain, Ytrain):
        for x, y in zip(Xtrain, Ytrain):
            # compute gradient
            gradient =  self.logistic_function(y * np.dot(x, self.weights)) * (-1.0 * y) * x
            # take a step down
            self.gradients -= self.step_size * gradient

    def l2_update(self):
        # regularization
        # print "weights:", np.linalg.norm(self.weights)
        # print "gradients", np.linalg.norm(self.gradients)
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

    def reset_gradient(self):
        self.gradients = np.zeros(self.num_feats)


    def predict(self, samples):
        ret = []
        for x in samples:
            prob_p = self.logistic_function(-np.dot(x, self.weights))
            if prob_p > .5:
                ret.append(1)
            else:
                ret.append(-1)
        return np.asarray(ret)

    def train(self, X, Y):
        self.update_gradients(X, Y)
        self.l2_update()
        self.update_weights()
        self.reset_gradient()
