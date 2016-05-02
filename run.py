import csv
import numpy as np
from gradient_descent import LRclassifier
from dual_coordinate_descent import DualLRclassifier
from sklearn import metrics
import matplotlib.pyplot as pl
import sys
from numpy import linalg as LA

def plot_error(epls, cdls, gdls):
    pl.figure(1)
    cd_line, = pl.plot(epls, cdls, color="green", marker='.', ls="--", ms=3)
    gd_line, = pl.plot(epls, gdls, color="red", marker='.', ls="--", ms=3)

    pl.xlim(0.0, epls[-1]+1.0)
    pl.ylim(min([min(cdls), min(gdls)])-0.05, max([max(cdls), max(gdls)])+0.05)
    pl.xlabel(r"Epoch")
    pl.ylabel(r"Error Rate")
    pl.title(r"Test Error")
    pl.legend([cd_line, gd_line], ['Coordinate Descent', 'Gradient Descent'])
    filename="error"+".png"
    pl.savefig(filename, format='png')
    print "error plotting ok"

def plot_gra_norm(epls, cdls, gdls):
    pl.figure(2)
    cd_line, = pl.plot(epls, cdls, color="green", marker='.', ls="--", ms=3)
    gd_line, = pl.plot(epls, gdls, color="red", marker='.', ls="--", ms=3)

    pl.xlim(0.0, epls[-1]+1.0)
    pl.ylim(min([min(cdls), min(gdls)])-0.05, max([max(cdls), max(gdls)])+0.05)
    pl.xlabel(r"Epoch")
    pl.ylabel(r"$||\nabla P(w)||$")
    pl.title(r"Gradient Norm")
    pl.legend([cd_line, gd_line], ['Coordinate Descent', 'Gradient Descent'])
    filename="grad_norm"+".png"
    pl.savefig(filename, format='png')
    print "gnorm plotting ok"

def read_data(filename):
    with open(filename) as txtfile:
        Xset = []
        Yset = []
        for row in txtfile:
            data_row = np.zeros(123)
            splits = row.split()
            label = int(splits[0])
            for i in range(1, len(splits) - 1):
                (index, value) = splits[i].split(":")
                data_row[int(index)] = float(value)
            Xset.append(data_row)
            Yset.append(label)
    return (np.asarray(Xset), np.asarray(Yset))


def primal_gradient_descent(Xtrain, Ytrain, Xtest, Ytest, lr=0.25, lmbd=1, epoches=100):
    num_feats = Xtrain.shape[1]
    num_samples = Xtrain.shape[0]
    classifier = LRclassifier(lr, lmbd, num_feats, num_samples)
    grad_norm_ls, test_error_ls =[], []

    for epoch in xrange(1, epoches+1):
        print "\n==============Running epoch: %d=================\n" % epoch

        classifier.train(Xtrain, Ytrain)

        # prediction
        Predtrain = classifier.predict(Xtrain)
        error_on_train = 1 - metrics.accuracy_score(Ytrain, Predtrain)
        # train_error_ls.append(error_on_train)

        Predtest = classifier.predict(Xtest)
        error_on_test = 1 - metrics.accuracy_score(Ytest, Predtest)
        test_error_ls.append(error_on_test)
        grad_norm_ls.append(LA.norm(classifier.accumulated_gradients, 2))

        print "\nError on train: %0.4f, Error on test: %0.4f\n" %(error_on_train, error_on_test)

    #plot_error(range(1, epoches+1), train_error_ls, test_error_ls)
    return (grad_norm_ls, test_error_ls)


def dual_coordinate_descent(Xtrain, Ytrain, Xtest, Ytest, epoches=100):
    num_feats = Xtrain.shape[1]
    num_samples = Xtrain.shape[0]
    classifier = DualLRclassifier(1, num_feats, num_samples)
    classifier.start(Xtrain, Ytrain)
    grad_norm_ls, test_error_ls =[], []

    for epoch in xrange(1, epoches+1):
        print "\n==============Running epoch: %d=================\n" % epoch

        classifier.train(Xtrain, Ytrain)

        # prediction
        Predtrain = classifier.predict(Xtrain)
        error_on_train = 1 - metrics.accuracy_score(Ytrain, Predtrain)
        # train_error_ls.append(error_on_train)

        Predtest = classifier.predict(Xtest)
        error_on_test = 1 - metrics.accuracy_score(Ytest, Predtest)
        test_error_ls.append(error_on_test)
        grad_norm_ls.append(LA.norm(classifier.accumulated_gradients, 2))

        print "\nError on train: %0.4f, Error on test: %0.4f\n" %(error_on_train, error_on_test)

    #plot_error(range(1, epoches+1), train_error_ls, test_error_ls)
    return (grad_norm_ls, test_error_ls)


def main():

    Xtrain, Ytrain = read_data("./data/a9a.train")
    Xtest, Ytest = read_data("./data/a9a.test")
    # problem = sys.argv[1]
    EPOCHES = 100

    #if problem == 'primal':
    gd_norm_ls, gd_error_ls = primal_gradient_descent(Xtrain, Ytrain, Xtest, Ytest, lr=0.25, lmbd=1, epoches=EPOCHES)
    #else:
    cd_norm_ls, cd_error_ls = dual_coordinate_descent(Xtrain, Ytrain, Xtest, Ytest, epoches=EPOCHES)

    plot_error(range(1, EPOCHES+1), cd_error_ls, gd_error_ls)

    # print cd_norm_ls, gd_norm_ls

    plot_gra_norm(range(1, EPOCHES+1), cd_norm_ls, gd_norm_ls)


if __name__ == "__main__":
    main()

