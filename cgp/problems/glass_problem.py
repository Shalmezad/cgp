import csv
import random
import numpy as np
from .problem_base import ProblemBase

class GlassProblem(ProblemBase):
    """
    UCI Glass Identification Data Set
    https://archive.ics.uci.edu/ml/datasets/glass+identification
    """

    def __init__(self) -> None:
        super().__init__()
        self._data = []
        # Load the dataset:
        with open('data/uci_glass/glass.data', newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in csvreader:
                if len(row) == 11:
                    input = row[1:10]
                    input = [float(i) for i in input]
                    # Have to subtract one as classes start at 1, and our indexes start at 0
                    output = int(row[10]) - 1
                    data_item = (input, output)
                    self._data.append(data_item)

        # Now the fun part:
        # We need to split between training and validation:
        # 150 data items:
        random.shuffle(self._data)
        # train_data = self._data[:150]
        # validation_data = self._data[150:]
        train_data = self._data
        validation_data = self._data
        # Now to map it better:
        train_data_in = [td[0] for td in train_data]
        train_data_in = np.asarray(train_data_in)
        train_data_out = [td[1] for td in train_data]
        train_data_out = np.asarray(train_data_out)

        self._training_data = (train_data_in, train_data_out)
        validation_data_in = [vd[0] for vd in validation_data]
        validation_data_in = np.asarray(validation_data_in)
        validation_data_out = [vd[1] for vd in validation_data]
        validation_data_out = np.asarray(validation_data_out)
        self._validation_data = (validation_data_in, validation_data_out)

    def numInputs(self):
        return 9
    
    def numOutputs(self):
        # Going with softmax:
        return 7
    
    def trainingSet(self):
        return self._training_data

    def validationSet(self):
        return self._validation_data
    
    def _one_hot(self, a, num_classes):
        return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

    def measureFitness(self, expected_output, actual_output):
        omax = np.max(actual_output)
        actual_output = actual_output / omax
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