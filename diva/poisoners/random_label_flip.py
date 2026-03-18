import numpy as np
from poisoners.base_poisoner import BasePoisoner

class RandomLabelFlipPoisoner(BasePoisoner):
    def __init__(self, poisoning_rate):
        super().__init__(poisoning_rate)

    def poison(self, dataset):
        poisoned_dataset = dataset.copy()
        y = poisoned_dataset["y"].values

        y_unique = np.unique(y)
        n_flip = int(len(y) * self.poisoning_rate)

        flip_indices = np.random.choice(len(y), size=n_flip, replace=False)
        y[flip_indices] = np.random.choice(y_unique, size=n_flip, replace=True)

        poisoned_dataset["y"] = y

        return poisoned_dataset