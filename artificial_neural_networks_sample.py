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

# Create the artificial neural networks here
# Fit Logistic Regression to Training Set
# create your classifier here

# Import Keras library based on Tensorflow
import keras

# sequential module required to initialize neural network
# dense to create the layers of the neural network
from keras.models import Sequential
from keras.layers import Dense

# use this to prevent overfitting
# overfitting is when the model trains too much on the training set
# causing it to cause some bias in test sets
# overfitting can be detected when there is unusually high variance when using cross validation
# because when it's overfitted, correlations learnt in training set does not apply to test set
# to prevent overfitting we need to use the dropout regularization
# we need to apply dropout disables some connection with the neurons in the ANN
from keras.layers import Dropout

# now we initialize the ANN
# two ways to initialize: by defining the sequence of layers
# or defining a graph
# create object from the Sequential class, we define by defining the sequence of layers
# this object is the model itself (the ANN), since it's a classification problem
# we need a classifier
# activation function for hidden layer is usually the rectifier function
# but for the output layer, it's usually the sigmoid function, because we can obtain
# probabilities of the outcome of whether they will leave or not
# can even rank the customers based on the percentage of leaving or staying
classifier = Sequential()

# add the different layers of the ANN
# adding the input layer and the first hidden layer
# Dense is going to take care of the weight allocation
# initially we need to put
# generally there are several methods to determine hidden layers
# in this code, we make it easy, average of input and output layers: 11 + 1 / 2
# the first part of the adding of layers, we need to specify no. of input layers
# because no layers has been initialized yet
# the following one no need
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(Dropout(p = 0.1))
# add drop out, p is the fraction of neurons we want to drop at EACH iteration
# IF you detect overfitting, try to set 0.1 first for p, then increase 0.2, 0.3, 0.4
# 0.5 is too much and will cause underfitting

# adding the second hidden layer
# no need to input no of input layers since it's already defined
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(p = 0.1))
# just add the drop out after setting the layers

# adding the output layer
# since it's a categorical data, we only need to create 1 output layer
# use sigmoid function
# if there are more than 2 categories, units need to represent no of categories
# then activation needs to change to 'softmax' which is sigmoid function for more
# than 2 categories
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# now we compile the ANN together
# adding stochastic gradient descent
# optimizer is the algorithm that helps to update the weights
# this is such that we can find the best weights
# loss is the loss function used (sum of squared differences)
# there is also logarithmic loss, where it is the loss used to calculate when the
# regression is logistic, here we use logarithmic loss since we use the sigmoid function in the output
# and since our dependent variable produces a binary outcome, then logarithmic function
# is "binary_crossentropy" else >2 is "categorical_crossentropy"
# metrics is to improve the model's performance after the batch is completed
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# epoch is the number of times we're training our ANN on the training set
# now we fit the ANN to the training set
# batch_size is the number of observations after which you want to update your weights
# epoch is a round when the whole training set passed through the ANN
# there is no rule of thumb, we just need to experiment with the batch_size and epochs
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Predict test results
y_pred = classifier.predict(X_test)
# at this point y_pred can use the information here to rank which customers which
# is more likely to leave and target them
# analyse more in depth why they are more likely to leave the bank
# overall it creates more value-add to the bank's operations
# so that future customers can be prevented to leave from the bank

# create a Confusion Matrix
# however we need the results to be only true or false in this matrix
# we need to decide on a treshold, where above the treshold it's 1 and below is 0
# natural one is 50%, if sensitive information, treshold can be higher (e.g. cancer)
y_pred = (y_pred > 0.5)
# this syntax is if (y_pred >0.5) return true

new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
# but this format is not standardised yet, so we have to standardise again
# take the sc object since it's already fitted to the X_train
# then we need to get the final prediction, true or false
# put 0.0 because we need the array to be a member of floats
new_prediction = (new_prediction > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# woohoo first ANN done!
# now try to predict whether this customer will leave the bank!
# geography: France
# credit score: 600
# gender: male
# age: 40
# tenure: 3
# balance: 60000
# no of products: 2
# has credit card: yes
# is active member: yes
# estimated salary: 50000
# refer to line 157 to 162 for answers and execute them

# saving a model
# classifier.save('model1.h5')


# =============================================================================


# at this point, we need to evaluate the model and retrain it
# this is because there is a variance problem, where different test sets
# will produce different accuracies, this is not ideal
# therefore we use k-Fold cross validation, where training set is split into 10 folds
# we train our model on 9 fold, and we test it on the remaining fold
# since with 10 folds, we can come up with combinations of 9 fold and 1 fold,
# we'll get a much better idea of the accuracy by taking the average and standard dev of those
# 10 iterations.

# implementing k-Fold cross-validation
# need to combine sklearn and keras, use keras wrapper to apply the kfcv to the keras model
# import keras wrapper and cross_val function
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

# build the architecture of the ann using a function
# repeat steps for building ANN
def build_classifier ():
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
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)

# then we take the mean of the accuracies as well as the standard deviation
mean = accuracies.mean()
variance = accuracies.std()


# =============================================================================



# after this we can tune the ANN
# parameters can be fixed or can vary
# hyper-parameters are those parameters that stay fixed for example batch_size,
# epochs, optimizer, no of neurons in the layer
# we need to train with fixed values
# however, we need to tune these hyper-parameters to find the most optimized values
# of these hyper-parameters
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

# build the architecture of the ann using a function
# repeat steps for building ANN
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    # optimizer is going to be replaced
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

# since batch_size and epochs are tuned, we need to remove the arguments from the KerasClassifier
classifier = KerasClassifier(build_fn = build_classifier)

# create a dictionary with the hyper-parameters
# it will try with all the values and return the best values
# to test for the parameters inside the Dense function, we need to create parameter
# in the build_classifier function
# rmsprop is also good for gradient descent
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}

# then we execute the cross validation function with the parameters
# scoring metric is to choose what category we use to pick the relevant params
grid_search = GridSearchCV(estimator = classifier, 
                           param_grid = parameters, 
                           scoring = 'accuracy',
                           cv = 10)

# then we fit the grid_search into out training set
grid_search = grid_search.fit(X_train, y_train)

# then we need to find the best selection and the best accuracy
# use the attribute from the GridSearchCV class
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

# then we wait lol, cos this will take bloody long knnccb
# in the end you will get a better result


