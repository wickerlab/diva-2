import numpy as np
from poisoners.base_poisoner import BasePoisoner
from sklearn.svm import SVC

class FALFAPoisoner(BasePoisoner):
    def __init__(self, poisoning_rate):
        super().__init__(poisoning_rate)

    def poison(self, dataset):
        poisoned_dataset = dataset.copy()
        X = poisoned_dataset.drop(columns=["y"]).values
        y = poisoned_dataset["y"].values.copy()

        n_samples = len(y)
        n_flip = int(n_samples * self.poisoning_rate)

        # FALFA (Feature Attack with Label Flip Approximation) poisoning:
        # We flip labels of samples that are close to the decision boundary,
        # approximating the impact of feature manipulation by targeting uncertain samples.

        clf = SVC(kernel="rbf", probability=True)
        clf.fit(X, y)
        probabilities = clf.predict_proba(X)
        uncertainties = 1 - np.max(probabilities, axis=1)

        flip_indices = np.argsort(uncertainties)[-n_flip:]
        unique_labels = np.unique(y)

        for idx in flip_indices:
            current_label = y[idx]

            candidate_labels = unique_labels[unique_labels != current_label]
            if len(candidate_labels) == 0:
                continue

            new_label = np.random.choice(candidate_labels)
            y[idx] = new_label

        poisoned_dataset["y"] = y
        return poisoned_dataset
