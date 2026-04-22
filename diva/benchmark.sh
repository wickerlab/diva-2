python ./test.py --dataset enron --step 0.1 --metalearner universal_meta_classifier.joblib --method poissvm --max_sample 2000
python ./test.py --dataset enron --step 0.1 --metalearner universal_meta_classifier.joblib --method feature_noise --max_sample 2000
python ./test.py --dataset enron --step 0.1 --metalearner universal_meta_classifier.joblib --method random_flip --max_sample 2000
python ./test.py --dataset enron --step 0.1 --metalearner universal_meta_classifier.joblib --method art --max_sample 500
