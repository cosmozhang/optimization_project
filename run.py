import csv
import numpy as np
from gradient_descent import LRclassifier
from dual_coordinate_descent import DualLRclassifier
from sklearn import metrics
import matplotlib.pyplot as pl
import sys


def plot_func(epls, trls, tels, model):
    pl.figure(1)
    train_line, = pl.plot(epls, trls, color="green", marker='.', ls="--", ms=3)
    test_line, = pl.plot(epls, tels, color="red", marker='.', ls="--", ms=3)

    pl.xlim(0.0, epls[-1]+1.0)
    pl.ylim(min([min(trls), min(tels)])-0.05, max([max(trls), max(tels)])+0.05)
    pl.xlabel("Epoch")
    pl.ylabel("Error Rate")
    pl.title("learning curve")
    pl.legend([train_line, test_line], ['Training', 'Testing'])
    filename="convergence_"+model+".png"
    pl.savefig(filename, format='png')
    print "plotting ok"

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
    train_error_ls, test_error_ls =[], []

    for epoch in xrange(1, epoches+1):
        print "\n==============Running epoch: %d=================\n" % epoch

        classifier.train(Xtrain, Ytrain)

        # prediction
        Predtrain = classifier.predict(Xtrain)
        error_on_train = 1 - metrics.accuracy_score(Ytrain, Predtrain)
        train_error_ls.append(error_on_train)

        Predtest = classifier.predict(Xtest)
        error_on_test = 1 - metrics.accuracy_score(Ytest, Predtest)
        test_error_ls.append(error_on_test)

        print "\nError on train: %0.4f, Error on test: %0.4f\n" %(error_on_train, error_on_test)

    plot_func(range(1, epoches+1), train_error_ls, test_error_ls, "primal")


def dual_coordinate_descent(Xtrain, Ytrain, Xtest, Ytest, epoches=100):
    num_feats = Xtrain.shape[1]
    num_samples = Xtrain.shape[0]
    classifier = DualLRclassifier(1, num_feats, num_samples)
    classifier.start(Xtrain, Ytrain)
    train_error_ls, test_error_ls =[], []

    for epoch in xrange(1, epoches+1):
        print "\n==============Running epoch: %d=================\n" % epoch

        classifier.train(Xtrain, Ytrain)
        # prediction
        Predtrain = classifier.predict(Xtrain)
        error_on_train = 1 - metrics.accuracy_score(Ytrain, Predtrain)
        train_error_ls.append(error_on_train)

        Predtest = classifier.predict(Xtest)
        error_on_test = 1 - metrics.accuracy_score(Ytest, Predtest)
        test_error_ls.append(error_on_test)

        print "\nError on train: %0.4f, Error on test: %0.4f\n" %(error_on_train, error_on_test)

    plot_func(range(1, epoches+1), train_error_ls, test_error_ls, "dual")


def main():

    Xtrain, Ytrain = read_data("./data/a9a.train")
    Xtest, Ytest = read_data("./data/a9a.test")
    problem = sys.argv[1]

    if problem == 'primal':
        primal_gradient_descent(Xtrain, Ytrain, Xtest, Ytest, lr=0.25, lmbd=1, epoches=100)
    else:
        dual_coordinate_descent(Xtrain, Ytrain, Xtest, Ytest, epoches=100)


if __name__ == "__main__":
    main()

