#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 16:35:37 2018

@author: charlie
"""

import time
import numpy as np
import copy
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVR, SVC
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
import seaborn as sns

### Get the Height from the A, B, C, D input with the given formula
def getY_h( A, B, C, D):
    Y_h = 0.72 + 0.1614 * A + 0.2619 * B - 0.1980 * C - \
        0.24636 * D + 0.3503 * A**2 + 0.3017 * D**2 -\
        0.2507 * A * C - 0.2479 * A * D + 0.22275 * C * D
    return Y_h

### Get the the training None*15 matrix, 15 denotes the number of parameter
def getX_quad( A, B, C, D ):
    num_samples = A.size
    X_quad_mat = np.concatenate(( np.ones((num_samples,1)), A, B, C, D,
                          A**2, B**2, C**2, D**2,
                          A*B, A*C, A*D, 
                          B*C, B*D, C*D), axis=1)
    return X_quad_mat

### Critical and NOK threshold

## Input data: X_train, Y_train
x_1 = np.array([
        -1, 1, -1, 1,
        0, 0, 0, 0,
       -1, 1, -1, 1,
        0, 0, 0, 0,
       -1, 1, -1, 1,
        0, 0, 0, 0,
        0, 0, 0])
x_2 = np.array([
        -1, -1, 1, 1,
        0, 0, 0, 0,
        0, 0, 0, 0,
        -1, 1, -1, 1,
        0, 0, 0, 0,
        -1, 1, -1, 1,
        0, 0, 0
        ])
x_3 = np.array([
        0, 0, 0, 0,
        -1, 1, -1, 1,
        0, 0, 0, 0,
        -1, -1, 1, 1,
        -1, -1, 1, 1,
        0, 0, 0, 0,
        0, 0, 0
        ])

x_4 = np.array([
        0, 0, 0, 0,
        -1, -1, 1, 1,
        -1, -1, 1, 1,
        0, 0, 0, 0,
        0, 0, 0, 0,
        -1, -1, 1, 1,
        0, 0, 0
        ])

y_height = np.array([
        0.890, 1.031, 1.158, 1.580,
        1.985, 1.092, 0.821, 0.819,
        1.091, 1.808, 1.399, 1.126,
        0.867, 1.084, 0.590, 1.031,
        0.986, 1.952, 0.912, 0.875,
        0.974, 1.738, 0.331, 1.235,
        0.710, 0.673, 0.780
        ])

y_thick = np.array([
        0.2709, 0.2747, 0.2868, 0.3153,
        0.4223, 0.2383, 0.2368, 0.2307,
        0.3055, 0.3272, 0.3060, 0.2977,
        0.2923, 0.2539, 0.2040, 0.2841,
        0.2578, 0.3862, 0.2624, 0.2195,
        0.3073, 0.3176, 0.1661, 0.2871,
        0.2020, 0.1891, 0.2260
        ]) 
'''
plt.figure()
plt.hist( y_height, bins=50 )
plt.title("Histogram of burr_height ")
#plt.show()
plt.savefig("Histogram_height_raw.png", format='png') 
'''    
    
beta_exact = np.array([ 0.720, 0.1614, 0.2619, -0.1980, -0.24636, \
                       0.3503, 0.0, 0.0, 0.3017, \
                       0.0, -0.2507, -0.2479, 0.0, 0.0, 0.22275])

num_para = beta_exact.size
## Input data transpose
num_samples = y_height.size

x_1 = x_1.reshape(num_samples, 1)
x_2 = x_2.reshape(num_samples, 1)
x_3 = x_3.reshape(num_samples, 1)
x_4 = x_4.reshape(num_samples, 1)
Y_height_train = y_height.reshape(num_samples, 1)
Y_thick_train = y_thick.reshape(num_samples, 1)


## Parameter initialization
X_train = getX_quad( x_1, x_2, x_3, x_4 )
'''
X_train = np.concatenate(( np.ones((num_samples,1)), x_1, x_2, x_3, x_4,
                          x_1**2, x_2**2, x_3**2, x_4**2,
                          x_1*x_2, x_1*x_3, x_1*x_4, 
                          x_2*x_3, x_2*x_4, x_3*x_4), axis=1)
'''
X_train_rank = np.linalg.matrix_rank(X_train)

#####################################################################
##          PARAMETER ESTIMATION WITH RAW DATA                     ##
#####################################################################
### Parameter estimation with least square linear matrix solver
beta_height, residual_height, rank, singu_value = np.linalg.lstsq(X_train, Y_height_train, rcond=None)

beta_thick, residual_thick, rank, singu_value = np.linalg.lstsq( X_train, Y_thick_train, rcond=None)

### Parameter estimation with svr
svr_lin = LinearSVR(verbose=True)
svr_lin.fit( X_train, Y_height_train )
beta_height_svr = svr_lin.coef_


#####################################################################
##                  MONTE CARLO SIMULATION                         ##
#####################################################################

### Monte Carlo Simulation
mu_vec = np.zeros((4,1))
sigma_vec = np.ones((4,1))

### Parameters -> number of sample an error percentage
num_samples_mc = 1000000;
error_percentage = 0.2


### Data of input 
A_train_mc = np.random.normal( mu_vec[0], sigma_vec[0], num_samples_mc )
B_train_mc = np.random.normal( mu_vec[1], sigma_vec[1], num_samples_mc )
C_train_mc = np.random.normal( mu_vec[2], sigma_vec[2], num_samples_mc )
D_train_mc = np.random.normal( mu_vec[3], sigma_vec[3], num_samples_mc )


### Data of output : Y. Using the given function in the paper.
Y_h_mc = getY_h( A_train_mc, B_train_mc, C_train_mc, D_train_mc )

print("\n First Element of Burr height: "+str(Y_h_mc[0]))
### Remove the negative burr height and corresponding input data
A_train_mc = A_train_mc[np.nonzero(Y_h_mc>0.)]
B_train_mc = B_train_mc[np.nonzero(Y_h_mc>0.)]
C_train_mc = C_train_mc[np.nonzero(Y_h_mc>0.)]
D_train_mc = D_train_mc[np.nonzero(Y_h_mc>0.)]
Y_h_mc     = Y_h_mc[np.nonzero(Y_h_mc>0.)]
num_samples_mc = Y_h_mc.size

### Reshape input data
A_train_mc = A_train_mc.reshape(num_samples_mc, 1)
B_train_mc = B_train_mc.reshape(num_samples_mc, 1)
C_train_mc = C_train_mc.reshape(num_samples_mc, 1)
D_train_mc = D_train_mc.reshape(num_samples_mc, 1)

### Get the training data matrix
X_h_mc = getX_quad( A_train_mc, B_train_mc, C_train_mc, D_train_mc )

### Feature analysis of Burr height_MC
'''
### Plot the histogram
plt.figure()
plt.hist( Y_h_mc, bins=50 )
plt.title("Histogram of burr_height_mc ")
plt.xlim([0, 7])
#plt.show()
plt.savefig("Histogram_height_mc.png", format='png')
'''
###     Split all data to training and test data 
X_h_mc_train, X_h_mc_test, Y_h_mc_train, Y_h_mc_test = train_test_split( X_h_mc, Y_h_mc, test_size=0.15 )

#####################################################################
##                    MC  REGRESSION                               ##
#####################################################################

### Least square (matrix inverse) solution to linear matrix equation
beta_height_mc_lstsq, res_height_mc_lstsq, rank_mc_lstsq, singu_value_mc_lstsq\
    = np.linalg.lstsq( X_h_mc_train, Y_h_mc_train, rcond=None)

### SVR
svr_lin.fit( X_h_mc_train, Y_h_mc_train )
beta_height_mc_svr = svr_lin.coef_
#beta_height_mc_svr = beta_height_mc_svr + svr_lin.intercept_

### Linear Regression
lin_regr = linear_model.LinearRegression()
lin_regr.fit( X_h_mc_train, Y_h_mc_train )
beta_height_mc_lin_regr = lin_regr.coef_
#beta_height_mc_lin_regr[0][0] = lin_regr.intercept_


### Full connected NN regression
#model = Sequential()
#model.add(Dense())

#####################################################################
##                  MC CLASSIFICATION                              ##
#####################################################################


### Set the critical and NOK threshold
Y_h_mc_sorted = np.sort( Y_h_mc, axis = None )
threshold_crit = Y_h_mc_sorted[int(num_samples_mc*0.5)]
threshold_nok  = Y_h_mc_sorted[int(num_samples_mc*0.7)]

### Split all data to training and test data 
X_h_mc_train_clf, X_h_mc_test_clf, Y_h_mc_train_clf, Y_h_mc_test_clf = train_test_split( X_h_mc, Y_h_mc, test_size=0.15 )

### Transfer raw continuous burr height to categarical label
### 1-> OK, 2->Critical, 3->NOK
Y_h_mc_label_train_clf = np.array([1 if x<threshold_crit else 2 if x<threshold_nok else 3 for x in Y_h_mc_train_clf])
Y_h_mc_label_test_clf = np.array([1 if x<threshold_crit else 2 if x<threshold_nok else 3 for x in Y_h_mc_test_clf])

print("\n#####################################################################\n\
##                  MC CLASSIFICATION                              ##\n\
#####################################################################\n\n")

#####################################################################
##                   Decision Tree  Classifier                     ##
#####################################################################

start_time = time.time()
DecsTreeClf = DecisionTreeClassifier( )
DecsTreeClf.fit(X_h_mc_train_clf, Y_h_mc_label_train_clf)
Y_h_mc_pred_DecsTreeClf = DecsTreeClf.predict(X_h_mc_test_clf)
DecsTreeAccuracy = accuracy_score( Y_h_mc_label_test_clf, Y_h_mc_pred_DecsTreeClf)
print("\nDecision Tree Accuracy: " + str(DecsTreeAccuracy) )
print("Decision Tree --- %s seconds ---\n" % (time.time() - start_time))
#DecsTreeClf.score( X_h_mc_test_clf, Y_h_mc_label_test_clf)


#####################################################################
##                   RANDOM FOREST  CLASSIFIER                     ##
#####################################################################
start_time = time.time()
RandForestClf = RandomForestClassifier(verbose = 1 )
RandForestClf.fit( X_h_mc_train_clf, Y_h_mc_label_train_clf)
Y_h_mc_pred_RandForestClf = RandForestClf.predict( X_h_mc_test_clf )
RandForestAccuracy = accuracy_score( Y_h_mc_label_test_clf, Y_h_mc_pred_RandForestClf )
print("\nRandom Forest Accuracy: " + str(RandForestAccuracy) ) 
print("Random Forest --- %s seconds ---\n" % (time.time() - start_time))


#####################################################################
##                      ADABOOST  CLASSIFIER                       ##
#####################################################################
start_time = time.time()
AdaBoostClf = AdaBoostClassifier()
AdaBoostClf.fit( X_h_mc_train_clf, Y_h_mc_label_train_clf)
Y_h_mc_pred_AdaBoostClf = AdaBoostClf.predict( X_h_mc_test_clf )
AdaBoostAccuracy = accuracy_score( Y_h_mc_label_test_clf, Y_h_mc_pred_AdaBoostClf)
print( "\nAdaBoost Accuracy: " + str(AdaBoostAccuracy) )
print("AdaBoost --- %s seconds ---\n" % (time.time() - start_time))


#####################################################################
##                    SUPPORT VECTOR  CLASSIFIER                   ##
#####################################################################
start_time = time.time()
svc = SVC(kernel='linear', verbose = 1)
svc.fit( X_h_mc_train_clf, Y_h_mc_label_train_clf )
Y_h_mc_pred_SVC = svc.predict( X_h_mc_test_clf )
SVCAccuracy = accuracy_score( Y_h_mc_label_test_clf, Y_h_mc_pred_SVC )
print("\nSVC Accuracy: " + str(SVCAccuracy) )
print("SVC --- %s seconds ---\n" % (time.time() - start_time))


#####################################################################
##              BAGGING SUPPORT VECTOR  CLASSIFIER                 ##
#####################################################################
'''
start_time = time.time()
svc_bagging = OneVsRestClassifier(BaggingClassifier(SVC(kernel='linear', verbose = 1)))
svc_bagging.fit(X_h_mc_train_clf, Y_h_mc_label_train_clf)
Y_h_mc_pred_SVC_bagging = svc_bagging.predict( X_h_mc_test_clf )
SVC_bagging_Accuracy = accuracy_score( Y_h_mc_label_test_clf, Y_h_mc_pred_SVC_bagging )
print("\nSVC Bagging Accuracy: " + str(SVC_bagging_Accuracy) )
print("SVC Bagging --- %s seconds ---\n" % (time.time() - start_time))
'''
#####################################################################
##                  NEURAL NETWORK CLASSIFIER                      ##
#####################################################################
start_time = time.time()
batch_size = 100
epochs = 3
num_classes = 3
Y_h_mc_label_train_nn = np_utils.to_categorical( Y_h_mc_label_train_clf-1, num_classes )

Y_h_mc_label_test_nn = np_utils.to_categorical( Y_h_mc_label_test_clf-1, num_classes )

model = Sequential()
model.add( Dense(num_para*2, input_dim = num_para, activation='relu') ) 
model.add( Dropout(0.2) )
model.add( Dense(num_para*2, activation='relu') )
#model.add( Dropout(0.25) )
model.add( Dense(3, activation='softmax') )

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit( X_h_mc_train_clf, Y_h_mc_label_train_nn, batch_size=batch_size, 
          epochs=epochs, verbose=0)

score = model.evaluate( X_h_mc_test_clf, Y_h_mc_label_test_nn, verbose=1 )
print( 'Neural Network Test score: ', score[0] )
print( 'Neural Network Test accuracy: ', score[1] )
print("Neural Network --- %s seconds ---\n" % (time.time() - start_time))




#####################################################################
##              CLASSIFICATION WITH ERROR IN INPUT DATA            ##
#####################################################################

print("\n#####################################################################\n\
##              CLASSIFICATION WITH ERROR IN INPUT DATA            ##\n\
#####################################################################\n\n")

##              TRAINING DATA AND LABELS                           ##

num_train_smpl_mc = X_h_mc_train_clf.shape[0]
error_num = int(error_percentage * num_train_smpl_mc)
error_idx = np.random.choice( num_train_smpl_mc, size=error_num)

error_mu = threshold_crit/2.0
error_sigma = threshold_crit/4.0

error = np.random.normal(error_mu, error_sigma, error_num )
Y_h_mc_train_clf_err = copy.deepcopy(Y_h_mc_train_clf)
for i in range(error_num):
    Y_h_mc_train_clf_err[error_idx[i]] = Y_h_mc_train_clf[error_idx[i]] + error[i]

'''
Y_h_mc_train_clf[error_idx[0]]
Y_h_mc_train_clf_err[error_idx[0]]
'''
Y_h_mc_label_train_clf_err = np.array([1 if x<threshold_crit else 2 if x<threshold_nok else 3 for x in Y_h_mc_train_clf_err])

#####################################################################
##         Decision Tree  Classifier With Error                    ##
#####################################################################

start_time = time.time()
DecsTreeClf_err = DecisionTreeClassifier( )
DecsTreeClf_err.fit(X_h_mc_train_clf, Y_h_mc_label_train_clf_err)
Y_h_mc_pred_DecsTreeClf_err = DecsTreeClf_err.predict(X_h_mc_test_clf)
DecsTreeAccuracy = accuracy_score( Y_h_mc_label_test_clf, Y_h_mc_pred_DecsTreeClf_err)
print("\nDecision Tree<ERROR> Accuracy: " + str(DecsTreeAccuracy) )
print("Decision Tree<ERROR> --- %s seconds ---\n" % (time.time() - start_time))


#####################################################################
##         RANDOM FOREST  CLASSIFIER WITH ERROR                    ##
#####################################################################
start_time = time.time()
RandForestClf_err = RandomForestClassifier(verbose = 1 )
RandForestClf_err.fit( X_h_mc_train_clf, Y_h_mc_label_train_clf_err)
Y_h_mc_pred_RandForestClf_err = RandForestClf_err.predict( X_h_mc_test_clf )
RandForestAccuracy = accuracy_score( Y_h_mc_label_test_clf, Y_h_mc_pred_RandForestClf_err )
print("\nRandom Forest<ERROR> Accuracy: " + str(RandForestAccuracy) ) 
print("Random Forest<ERROR> --- %s seconds ---\n" % (time.time() - start_time))


#####################################################################
##                    SUPPORT VECTOR  CLASSIFIER                   ##
#####################################################################
'''
start_time = time.time()
svc_err = SVC(kernel='linear', verbose = 1)
svc_err.fit( X_h_mc_train_clf, Y_h_mc_label_train_clf_err )
Y_h_mc_pred_SVC_err = svc_err.predict( X_h_mc_test_clf )
SVCAccuracy = accuracy_score( Y_h_mc_label_test_clf, Y_h_mc_pred_SVC_err )
print("\nSVC<ERROR> Accuracy: " + str(SVCAccuracy) )
print("SVC<ERROR> --- %s seconds ---\n" % (time.time() - start_time))
'''


#####################################################################
##                  NEURAL NETWORK CLASSIFIER                      ##
#####################################################################
start_time = time.time()
batch_size = 100
epochs = 3
num_classes = 3
Y_h_mc_label_train_nn_err = np_utils.to_categorical( Y_h_mc_label_train_clf_err-1, num_classes )

Y_h_mc_label_test_nn = np_utils.to_categorical( Y_h_mc_label_test_clf-1, num_classes )

model_err = Sequential()
model_err.add( Dense(num_para*2, input_dim = num_para, activation='relu') ) 
model_err.add( Dropout(0.2) )
model_err.add( Dense(num_para*2, activation='relu') )
#model_err.add( Dropout(0.25) )
model_err.add( Dense(3, activation='softmax') )

model_err.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model_err.fit( X_h_mc_train_clf, Y_h_mc_label_train_nn_err, batch_size=batch_size, 
          epochs=epochs, verbose=0)

score = model_err.evaluate( X_h_mc_test_clf, Y_h_mc_label_test_nn, verbose=1 )
print( 'Neural Network<ERROR> Test score: ', score[0] )
print( 'Neural Network<ERROR> Test accuracy: ', score[1] )
print("Neural Network<ERROR> --- %s seconds ---\n" % (time.time() - start_time))

