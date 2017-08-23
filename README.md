Instructions for running the project

Dependencies: listed in requirements.txt
In particular, the following python packages are needed:
scipy (0.19.1)
numpy (1.13.1)
scikit-learn (0.19.0)
imblearn (0.30.0) - install: pip3 -U imbalanced-learn
tensorflow (1.2.1)
pandas (0.20.2)

Random Forest:
Code for the Random Forest is in random_forest.py

This can be run with: python3 `random_forest.py`

It expects the data set to be contained in a file 'creditcard.csv' in the same directory.
Running this file will first train and report the results for the random forest on unaltered data,
then on data preprocessed using SMOTE.

MLP Neural Net:
Code for the baseline run on the MLP neural net is in neural_net.py. Code to optimize the various aspects 
of the neural net are present in files started with the word "neural".

Same as Random Forest, it expects the data set to be contained in a file named creditcard.csv located in the same folder. Additionally it can be run with python3 `neural_net.py`.


Results for both algorithms will be printed out and also stored in csv files in the local directory.
