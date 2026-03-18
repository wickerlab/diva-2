import numpy as np
from poisoners.base_poisoner import BasePoisoner
from sklearn.svm import SVC

class ALFAPoisoner(BasePoisoner):
    def __init__(self, poisoning_rate):
        super().__init__(poisoning_rate)

    def poison(self, dataset):
        poisoned_dataset = dataset.copy()
        X = poisoned_dataset.drop(columns=["y"]).values
        y = poisoned_dataset["y"].values.copy()

        y_unique = np.unique(y)
        n_classes = len(y_unique)
        n_samples = len(y)
        n_flip = int(n_samples * self.poisoning_rate)

        # ALFA (Adversarial Label Flip Attack) poisoning:
        # Flip labels to maximize the loss of a trained classifier.
        # We iteratively select the sample whose label flip causes the
        # greatest increase in loss (approximated via decision function distance).

        flipped = np.zeros(n_samples, dtype=bool)

        for _ in range(n_flip):
            # Train a classifier on the current (partially poisoned) labels
            clf = SVC(kernel="rbf")
            clf.fit(X, y)

            # Compute decision function values for all non-flipped samples
            decision_values = clf.decision_function(X)

            best_idx = None
            best_score = -np.inf

            for i in range(n_samples):
                if flipped[i]:
                    continue

                original_label = y[i]
                # Choose the most adversarial label (different from current)
                candidate_labels = y_unique[y_unique != original_label]

                if len(candidate_labels) == 0:
                    continue

                # For each candidate, the adversarial impact is approximated
                # by how confidently the classifier predicts the current label
                # (flipping confident correct predictions hurts the model most)
                if n_classes == 2:
                    score = abs(decision_values[i])
                else:
                    # Multi-class: use the margin for the current class
                    class_idx = list(y_unique).index(original_label)
                    score = decision_values[i][class_idx]

                if score > best_score:
                    best_score = score
                    best_idx = i

            if best_idx is not None:
                original_label = y[best_idx]
                candidate_labels = y_unique[y_unique != original_label]
                if n_classes == 2:
                    # Flip to the other class
                    y[best_idx] = candidate_labels[0]
                else:
                    # Flip to the class that would cause the most damage:
                    # the class with the lowest decision function value
                    class_indices = [list(y_unique).index(c) for c in candidate_labels]
                    worst_class_idx = class_indices[
                        np.argmin(decision_values[best_idx][class_indices])
                    ]
                    y[best_idx] = y_unique[worst_class_idx]
                flipped[best_idx] = True

        poisoned_dataset["y"] = y

        return poisoned_dataset