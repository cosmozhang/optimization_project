import csv
import numpy as np
from gradient_descent import GradientDescent
from sklearn import metrics

with open("./data/a9a.train") as txtfile:
    dataset = []
    labels = []
    for row in txtfile:
        data_row = np.zeros(123)
        splits = row.split()
        label = int(splits[0])
        for i in range(1, len(splits) - 1):
            (index, value) = splits[i].split(":")
            data_row[int(index)] = float(value)
        dataset.append(data_row)
        labels.append(label)

dataset = np.asarray(dataset)
labels = np.asarray(labels)

lr = GradientDescent(100, 0.25, 1)
lr.fit(dataset, labels)
ret = lr.predict(dataset)

print metrics.classification_report(labels, ret)
