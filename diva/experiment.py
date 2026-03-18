import warnings

from poisoners.alfa import ALFAPoisoner
from helpers.dataset import generate_datasets, extract_complexity_measures
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

class Experiment:
    def __init__(self, num_datasets):
        self.num_datasets = num_datasets
        self.datasets = []
        self.complexity_measures = []
        self.poisoned_datasets = []

    def run(self):
        self.datasets = generate_datasets(num_datasets=self.num_datasets)

        for dataset in self.datasets:
            complexity = extract_complexity_measures(dataset)
            self.complexity_measures.append(complexity)

        for dataset in self.datasets:
            poisoner = ALFAPoisoner(poisoning_rate=0.4)
            poisoned_dataset = poisoner.poison(dataset)
            self.poisoned_datasets.append(poisoned_dataset)

        for i, poisoned_dataset in enumerate(self.poisoned_datasets):
            original = self.datasets[i]

            X_poisoned = poisoned_dataset.drop('y', axis=1)
            y_poisoned = poisoned_dataset['y']
            X_train, X_test, y_train, y_test = train_test_split(
                X_poisoned, y_poisoned, test_size=0.2, random_state=42
            )

            X_clean = original.drop('y', axis=1)
            y_clean = original['y']

            X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(
                X_clean, y_clean, test_size=0.2, random_state=42
            )

            poisoned_model = RandomForestClassifier(random_state=42)
            poisoned_model.fit(X_train, y_train)

            clean_model = RandomForestClassifier(random_state=42)
            clean_model.fit(X_train_clean, y_train_clean)

            poisoned_preds = poisoned_model.predict(X_test)
            poisoned_preds_on_clean = poisoned_model.predict(X_test_clean)
            clean_preds = clean_model.predict(X_test_clean)

            poisoned_acc = accuracy_score(y_test, poisoned_preds)
            poisoned_on_clean_acc = accuracy_score(y_test_clean, poisoned_preds_on_clean)
            clean_acc = accuracy_score(y_test_clean, clean_preds)

            print(f"Dataset {i+1}:")
            print(f"  Complexity Measures: {self.complexity_measures[i]}")
            print(f"  Poisoned Model Accuracy on Poisoned Data: {poisoned_acc:.4f}")
            print(f"  Poisoned Model Accuracy on Clean Data: {poisoned_on_clean_acc:.4f}")
            print(f"  Clean Model Accuracy on Clean Data: {clean_acc:.4f}")


if __name__ == "__main__":
    experiment = Experiment(num_datasets=1)
    experiment.run()