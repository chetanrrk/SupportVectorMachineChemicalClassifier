# SupportVectorMachineChemicalClassifier
This code was used for building the support vector machine based chemical classifier model for the published Collaborative Modeling Project for Androgen Receptor Activity (CoMPARA)
https://ehp.niehs.nih.gov/doi/full/10.1289/EHP5580

Users can pass in the training file, testing file, and prediction (to predit the labels for) in csv format containing 'N' data and 'M' features

# To run the code: 
python svm_chemical_classifier.py train.csv test.csv predict.csv

# General Description
The code applies undersampling of the highly populated class to balance the data set
The code runs the 10 fold cross validation and find the best classifier 
The code applies rbf kernel for the svm model
The code uses leverage statistics to compute the model domain and flag compounds outside the domain
The code outputs the uncertinity in prediction using the vairiance of the models from the cross validation

# Requirement.txt includes
1) sciket-learn (available in anaconda)
2) pandas (available in anaconda)
3) numpy (available in anaconda)
4) matplotlib (available in anaconda)
5) Multiprocessing (available in python)

