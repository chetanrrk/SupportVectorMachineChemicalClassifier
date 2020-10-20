# SupportVectorMachineChemicalClassifier
This code was used for building the support vector machine based chemical classifier model for the published Collaborative Modeling Project for Androgen Receptor Activity (CoMPARA)
https://ehp.niehs.nih.gov/doi/full/10.1289/EHP5580

Users can pass in the training file, testing file, and prediction (to predit the labels for) in csv format containing 'N' data and 'M' features

# To run the code: 
python svm_chemical_classifier.py train.csv test.csv predict.csv

where,

train.csv: contains training data containg 'm' chemicals and 'n' features and activity class label (0/1)

test.csv: contains testing data containg 'm' chemicals and 'n' features and activity class label (0/1)

predict.csv: contains prediction data containg 'm' chemicals and 'n' features and no label (will be predicted) 

# General Description
Applies undersampling of the highly populated class to balance the data set

Scales the features using minmax scaler and applies feature selection using PCA and L2 regularization 

Runs a 10 fold cross validation and find the best classifier 

Applies rbf kernel for the svm model

Applies leverage statistics to compute the model domain and flag compounds outside the domain

Outputs the uncertinity in prediction using the vairiance of the models from the cross validation

# Requirement.txt includes
1) sciket-learn (available in anaconda)
2) pandas (available in anaconda)
3) numpy (available in anaconda)
4) matplotlib (available in anaconda)
5) Multiprocessing (available in python)

