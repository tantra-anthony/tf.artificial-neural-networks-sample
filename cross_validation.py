# Artificial Neural Network

# 1. Randomly initialise weights to small numbers close to 0
# 2. Input first observation of dataset in input layer, one feature in one input node
# 3. Forward-Propagation from left to right in diagram, propagate the activations
#    until getting the predicted value of y
# 4. Compare predicted result to the actual result. Measure generated error/loss
# 5. Back-Propagation: from right to left in diagram, error is back-propagated
#    update the weights according to how much they are responsible for the error
#    Learning rate decides by how much we update the weights.
# 6. Repeat steps 1 to 5 and update weights for every observation (Reinforcement Learning)
#    Or Repeat steps 1 to 5 but update the weights only after a batch of observations
#    (Batch Learning)
# 7. When the whole training set is passed through the ANN, it makes an epoch.
#    Redo more epochs

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# upgrade all
# conda upgrade --all

# start data preprocessing
# this is a classification problem
# predict binary outcome whether customers leave or stay at the bank
# artificial neural networks can detect the most influential independent variable
# then match the appropriate weights to them


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# we need to encode "Categorical Data"
# Encoding Categorial Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# create first one for country
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1]) # returns an encoded array

# create first one for gender
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_1.fit_transform(X[:, 2]) # returns an encoded array

# categorical data is not ordinal, no relation between one another
# create dummy variables for country only
onehotencoder = OneHotEncoder(categorical_features = [1]) # 0 is the index number of column
X = onehotencoder.fit_transform(X).toarray()

# remove first column to prevent the dummy variable trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

# build the architecture of the ann using a function
# repeat steps for building ANN
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

# next step is we need to wrap the whole thing together
# instead of fitting it with the batch_size and epochs only, we will use the k fold cross val
# on 10 different folds, the only thing that is different is the training part
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)

# cross_val_score function then will get the 10 accuracies of the 10 folds
# estimator is the object to use to fit the data: X_train y_train
# cv is the number of folds, usually put 10, check if low bias low variance
# n_jobs is no of CPU used, -1 uses parallel computing to run at the same time
# if n_jobs -1 not working try using 1 or 2
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = 1)

# then we take the mean of the accuracies as well as the standard deviation
mean = accuracies.mean()
variance = accuracies.std()