# Meta-Poison-Detector: Detecting Data Poisoning via Complexity Measures
## 📌 Overview
In adversarial machine learning, detecting if a training set has been poisoned is notoriously difficult. This project introduces a novel, lightweight detection mechanism based on the premise that Complexity Measures (C-Measures) are all you need to predict the theoretical clean accuracy of a dataset.

By leveraging a verified clean test set, we can estimate exactly how well a classifier should perform. If a model trained on an unverified training set drastically underperforms this theoretical baseline, we can reliably flag the training data as poisoned.

## 🧠 The Core Concept
The detection pipeline relies on the gap between Predicted Clean Accuracy and Empirical Accuracy:

- **The Ground Truth**: For any given classification task, the intrinsic complexity (C-Measures) of a test set is enough to approximate a standard classifier's accuracy.

- **The Meta-Learner**: We train a Meta-Learner to understand this relationship. It takes the C-Measures of a clean test set as input and outputs the Expected Clean Accuracy.

- **The Detection Trigger**: When evaluating a new model, if the difference between the Meta-Learner's predicted accuracy and the model's actual empirical accuracy is statistically significant, it indicates the model learned from corrupted (poisoned) training data.

## 🚀 Pipeline Architecture
The project is split into two distinct phases:

### Phase 1: Training the Meta-Learner
To teach the Meta-Learner how to predict accuracy, we generate a massive "Meta-Database" of synthetic and real-world datasets:

Generate/Load Data: Create diverse datasets with varying features, classes, overlapping clusters, and noise.

Extract C-Measures: Calculate the complexity metrics purely on the clean test set.

Train Baseline Classifiers: Train standard models (e.g., SVR, SVC) on the clean training splits to establish the true empirical accuracy.

Meta-Training: Train a regressor (the Meta-Learner) to predict the baseline accuracy based solely on the test set's C-Measures.

### Phase 2: Real-World Testing & Poison Detection
We deploy the trained Meta-Learner against real-world datasets (e.g., MNIST, Enron, Spambase) to detect adversarial attacks:

Extract the C-Measures from a clean, trusted test set.

Ask the Meta-Learner to predict the Expected Clean Accuracy.

Train your standard classifier on a potentially poisoned training set.

Evaluate your classifier on the clean test set to get its Empirical Accuracy.

Compare: If Expected Accuracy - Empirical Accuracy > Threshold, flag the training set as poisoned.

## 📊 Results & Validation
Extensive testing on real-world datasets validates this approach. Our Meta-Learner successfully maps complexity to performance with remarkable precision:

Baseline Error: The difference between the Meta-Learner's predicted clean accuracy and the true empirical clean accuracy on real-world datasets is less than 3%.

Reliable Triggering: Because the Meta-Learner's natural margin of error is so tight, empirical accuracy drops caused by data poisoning (e.g., Label Flipping, Biggio Attacks) easily pierce the confidence threshold, allowing for robust, automated flagging of poisoned datasets without needing to inspect the training data itself.