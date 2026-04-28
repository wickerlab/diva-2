python ./test.py --dataset synthetic --step 0.1 --method feature_noise --max_sample 2000
python ./test.py --dataset synthetic --step 0.1 --method random_flip --max_sample 2000
python ./test.py --dataset synthetic --step 0.1 --method alfa --max_sample 2000
python ./test.py --dataset synthetic --step 0.1 --method biggio --max_sample 500

python ./test.py --dataset enron --step 0.1 --method feature_noise --max_sample 2000
python ./test.py --dataset enron --step 0.1 --method random_flip --max_sample 2000
python ./test.py --dataset enron --step 0.1 --method alfa --max_sample 2000
python ./test.py --dataset enron --step 0.1 --method biggio --max_sample 1000

python ./test.py --dataset breast_cancer --step 0.1 --method feature_noise --max_sample 2000
python ./test.py --dataset breast_cancer --step 0.1 --method random_flip --max_sample 2000
python ./test.py --dataset breast_cancer --step 0.1 --method alfa --max_sample 2000
python ./test.py --dataset breast_cancer --step 0.1 --method biggio --max_sample 1000

python ./test.py --dataset spambase --step 0.1 --method feature_noise --max_sample 2000
python ./test.py --dataset spambase --step 0.1 --method random_flip --max_sample 2000
python ./test.py --dataset spambase --step 0.1 --method alfa --max_sample 2000
python ./test.py --dataset spambase --step 0.1 --method biggio --max_sample 1000

python ./test.py --dataset diabetes --step 0.1 --method feature_noise --max_sample 2000
python ./test.py --dataset diabetes --step 0.1 --method random_flip --max_sample 2000
python ./test.py --dataset diabetes --step 0.1 --method alfa --max_sample 2000
python ./test.py --dataset diabetes --step 0.1 --method biggio --max_sample 1000
