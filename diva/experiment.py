from poisoners.random_label_flip import RandomLabelFlipPoisoner
from helpers.dataset import generate_datasets, extract_complexity_measures

import warnings

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
            poisoner = RandomLabelFlipPoisoner(spacing=0.05, poisoning_rate=0.1)
            poisoned_dataset = poisoner.poison(dataset)
            self.poisoned_datasets.append(poisoned_dataset)

        print("Original dataset:")
        print(self.datasets[0].head())
        print("\nPoisoned dataset:")
        print(self.poisoned_datasets[0].head())

if __name__ == "__main__":
    experiment = Experiment(num_datasets=1)
    experiment.run()