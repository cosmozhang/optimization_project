import csv
import numpy as np
from gradient_descent import LRclassifier
from sklearn import metrics
import matplotlib.pyplot as pl


def plot_func(epls, trls, tels):
    pl.figure(1)
    train_line, = pl.plot(epls, trls, color="green", marker='.', ls="--", ms=3)
    test_line, = pl.plot(epls, tels, color="red", marker='.', ls="--", ms=3)

    pl.xlim(0.0, epls[-1]+1.0)
    pl.ylim(min([min(trls), min(tels)])-0.05, max([max(trls), max(tels)])+0.05)
    pl.xlabel("Epoch")
    pl.ylabel("Error Rate")
    pl.title("learning curve")
    pl.legend([train_line, test_line], ['Training', 'Testing'])
    filename="convergence"+".png"
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


if __name__ == "__main__":


    Xtrain, Ytrain = read_data("./data/a9a.train")
    Xtest, Ytest = read_data("./data/a9a.test")

    lr = 0.25
    lmbd = 1
    num_feats = Xtrain.shape[1]
    num_samples = Xtrain.shape[0]

    classifier = LRclassifier(lr, lmbd, num_feats, num_samples)
    epoches = 1000

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

    plot_func(range(1, epoches+1), train_error_ls, test_error_ls)






