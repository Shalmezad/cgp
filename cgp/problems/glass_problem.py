import csv
import random

import numpy as np
import numpy.typing as npt

from .problem_base import ProblemBase


class GlassProblem(ProblemBase):
    """
    UCI Glass Identification Data Set
    https://archive.ics.uci.edu/ml/datasets/glass+identification
    """

    def __init__(self, normalizeInputs: bool = False) -> None:
        super().__init__()
        self._data = []
        # Load the dataset:
        with open('data/uci_glass/glass.data', newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in csvreader:
                if len(row) == 11:
                    input = [float(i) for i in row[1:10]]
                    # Have to subtract one as classes start at 1,
                    # and our indexes start at 0
                    output = int(row[10]) - 1
                    data_item = (input, output)
                    self._data.append(data_item)

        # Need to normalize:
        if normalizeInputs:
            max_input = [0] * 9
            for i, o in self._data:
                for idx in range(9):
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
        # train_data = self._data[:150]
        # validation_data = self._data[150:]
        train_data = self._data
        validation_data = self._data
        # Now to map it better:
        train_data_in = np.asarray([td[0] for td in train_data])
        train_data_out = np.asarray([td[1] for td in train_data])

        self._training_data = (train_data_in, train_data_out)
        validation_data_in = np.asarray([vd[0] for vd in validation_data])
        validation_data_out = np.asarray([vd[1] for vd in validation_data])
        self._validation_data = (validation_data_in, validation_data_out)

    def numInputs(self) -> int:
        return 9

    def numOutputs(self) -> int:
        # Going with softmax:
        return 7

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
                       actual_output:  npt.NDArray[np.float64]) -> float:
        with np.errstate(all='raise'):
            try:
                omax = np.max(actual_output)
                if omax != 0:
                    actual_output = actual_output / omax

                true_class_logits = actual_output[
                    np.arange(len(actual_output)),
                    expected_output]
                cross_entropy = - true_class_logits + np.log(
                    np.sum(np.exp(actual_output), axis=-1))
                # return cross_entropy
                return np.mean(cross_entropy)  # type: ignore
            except FloatingPointError:
                print("Max: {}".format(np.max(actual_output)))
                print("Actual output: {}".format(actual_output))
                print("Expected output: {}".format(expected_output))
                raise
