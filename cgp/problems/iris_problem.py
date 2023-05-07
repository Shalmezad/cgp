import csv
import random
import numpy as np
from .problem_base import ProblemBase

class IrisProblem(ProblemBase):

    def __init__(self) -> None:
        super().__init__()
        self._data = []
        # Load the dataset:
        with open('data/iris_flower_classification/iris.data', newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in csvreader:
                if len(row) == 5:
                    input = row[:4]
                    input = [float(i) for i in input]
                    output = row[4]
                    data_item = (input, output)
                    self._data.append(data_item)

        self._classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        self._class_map = {}
        for idx in range(len(self._classes)):
            c = self._classes[idx]
            self._class_map[c] = idx
        # Now the fun part:
        # We need to split between training and validation:
        # 150 data items:
        random.shuffle(self._data)
        train_data = self._data[:100]
        validation_data = self._data[100:]
        # Now to map it better:
        train_data_in = [td[0] for td in train_data]
        train_data_in = np.asarray(train_data_in)
        train_data_out = [self._class_map[td[1]] for td in train_data]
        train_data_out = np.asarray(train_data_out)
        # train_data_out = self._one_hot(train_data_out, 3)

        self._training_data = (train_data_in, train_data_out)
        validation_data_in = [vd[0] for vd in validation_data]
        validation_data_in = np.asarray(validation_data_in)
        validation_data_out = [self._class_map[vd[1]] for vd in validation_data]
        validation_data_out = np.asarray(validation_data_out)
        # validation_data_out = self._one_hot(validation_data_out, 3)
        self._validation_data = (validation_data_in, validation_data_out)

    def numInputs(self):
        return 4
    
    def numOutputs(self):
        # Going with softmax:
        return 3
    
    def trainingSet(self):
        return self._training_data

    def validationSet(self):
        return self._validation_data
    
    def _one_hot(self, a, num_classes):
        return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

    def measureFitness(self, expected_output, actual_output):
        exps = np.exp(actual_output)
        exp_sum = np.sum(exps, axis=1)
        softmax = exps / exp_sum[:,None]
        # print(expected_output)
        # print(expected_output.shape)

        # m = expected_output.shape[0]
        # log_likelihood = -np.log(softmax[range(m),expected_output])
        # loss = np.sum(log_likelihood) / m
        true_class_logits = actual_output[np.arange(len(actual_output)), expected_output]
        cross_entropy = - true_class_logits + np.log(np.sum(np.exp(actual_output), axis=-1))
        # return cross_entropy
        return np.mean(cross_entropy)