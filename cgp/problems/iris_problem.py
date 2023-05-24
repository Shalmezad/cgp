import csv
import random

import numpy as np
import numpy.typing as npt

from .problem_base import ProblemBase


class IrisProblem(ProblemBase):

    def __init__(self, normalizeInputs: bool = False) -> None:
        super().__init__()

        self._classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        self._class_map = {}
        for idx in range(len(self._classes)):
            c = self._classes[idx]
            self._class_map[c] = idx

        self._data = []
        # Load the dataset:
        datafile = 'data/iris_flower_classification/iris.data'
        with open(datafile, newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in csvreader:
                if len(row) == 5:
                    input = [float(i) for i in row[:4]]
                    output = self._class_map[row[4]]
                    data_item = (input, output)
                    self._data.append(data_item)

        if normalizeInputs:
            # Need to normalize:
            max_input = [0, 0, 0, 0]
            for i, o in self._data:
                for idx in range(4):
                    if i[idx] > max_input[idx]:
                        max_input[idx] = i[idx]

            for idx in range(len(self._data)):
                i, o = self._data[idx]
                i = np.asarray(i) / np.asarray(max_input)
                self._data[idx] = (i, o)

        # Now the fun part:
        # We need to split between training and validation:
        # 150 data items:
        random.shuffle(self._data)
        train_data = self._data[:100]
        validation_data = self._data[100:]
        # Now to map it better:
        train_data_in = np.asarray([td[0] for td in train_data])
        train_data_out = np.asarray(
            [td[1] for td in train_data])

        self._training_data = (train_data_in, train_data_out)
        validation_data_in = np.asarray([vd[0] for vd in validation_data])
        validation_data_out = np.asarray(
            [vd[1] for vd in validation_data])
        self._validation_data = (validation_data_in, validation_data_out)

    def numInputs(self) -> int:
        return 4

    def numOutputs(self) -> int:
        # Going with softmax:
        return 3

    def trainingSet(self) -> tuple[
            npt.NDArray[np.float64],
            npt.NDArray[np.float64]]:
        return self._training_data

    def validationSet(self) -> tuple[
            npt.NDArray[np.float64],
            npt.NDArray[np.float64]]:
        return self._validation_data

    def measureFitness(self,
                       expected_output: npt.NDArray[np.float64],
                       actual_output: npt.NDArray[np.float64]
                       ) -> float:
        true_class_logits = actual_output[
            np.arange(len(actual_output)), expected_output]
        cross_entropy = - true_class_logits + np.log(
            np.sum(np.exp(actual_output), axis=-1))
        # return cross_entropy
        return np.mean(cross_entropy)  # type: ignore
