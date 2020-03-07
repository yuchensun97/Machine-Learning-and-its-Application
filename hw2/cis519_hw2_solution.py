import pandas as pd

import numpy as np
from numpy import linalg as LA
from numpy.linalg import *

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import RidgeCV
'''
    Template for polynomial regression
'''

import numpy as np

#-----------------------------------------------------------------
#  Class PolynomialRegression
#-----------------------------------------------------------------

class PolynomialRegression:

    def __init__(self,degree=1,regLambda=0,tuneLambda=False,regLambdaValues=[0]):
        '''
        Constructor
        '''
        #TODO
        self.degree = degree
        self.Lambda = regLambda
        self.Lambdavalue = regLambdaValues
        self.alpha = 0.25
        self.theta = np.matrix(np.zeros((degree+1))).T
        self.tuneLambda = tuneLambda
        self.mean = 0
        self.std = 0


    def polyfeatures(self, X, degree):
        '''
        Expands the given X into an n * d array of polynomial features of
            degree d.

        Returns:
            A n-by-d data frame, with each column comprising of
            X, X * X, X ** 3, ... up to the dth power of X.
            Note that the returned matrix will not include the zero-th power.

        Arguments:
            X is an n-by-1 data frame
            degree is a positive integer
        '''
        #TODO
        X = X.to_numpy()     # convert dataframe to n*1 array
        poly_X = X
        for i in range(2,degree+1):
            X_col = X**i     # compute the new feature
            poly_X = np.c_[poly_X,X_col]     # add the new feature to the previous matrix
        
        return pd.DataFrame(poly_X)
        

    def fit(self, X, y):
        '''
            Trains the model
            Arguments:
                X is a n-by-1 data frame
                y is an n-by-1 data frame
            Returns:
                No return value
            Note:
                You need to apply polynomial expansion and scaling first
        '''
        #TODO
        # preprocessing
        n = len(y)
        y = y.to_numpy()
        fit_X = self.polyfeatures(X,self.degree)    # perform polynomial feature expansion, n-by-d matrix
        fit_X = fit_X.to_numpy()    # convert the dataframe to np.array
        X_mean = np.mean(fit_X, axis=0)    # compute mean value of each feature, types: row
        self.mean = X_mean
        X_std = np.std(fit_X, axis=0)     # compute std value of each feature, types: row
        self.std = X_std
        fit_X = (fit_X-X_mean)/X_std      # standarize the polynomial feature, n-by-d matrix
        
        n,d = fit_X.shape
        y = y.reshape(n,1)
        
        theta = self.theta

        # automatically tune the lambda values
        if self.tuneLambda:
            reg_cv = RidgeCV(alphas=self.Lambdavalue,cv=2).fit(fit_X,y)
            Lambda = reg_cv.alpha_
        else:
            Lambda = self.Lambda
            
        # add ones to the feature
        fit_X = np.c_[np.ones((n,1)),fit_X]   
            
        while True:
            # implementing theta iteration
            yhat = fit_X*theta
            old_theta = theta.copy()
            theta = theta-(fit_X.T*(yhat-y))*(self.alpha/n)-self.alpha*Lambda*theta
            new_theta = theta.copy()
            eps = LA.norm(new_theta-old_theta)
            if eps<1E-4:
                self.theta = new_theta
                break

        return
        
    def predict(self, X):
        '''
        Use the trained model to predict values for each instance in X
        Arguments:
            X is a n-by-1 data frame
        Returns:
            an n-by-1 data frame of the predictions
        '''
        # TODO
        
        # preprocessing
        pre_fit_X = self.polyfeatures(X,self.degree)    # perform polynomial feature expansion, n-by-d matrix
        pre_fit_X = pre_fit_X.to_numpy()
        pre_fit_X = (pre_fit_X-self.mean)/self.std      # standarize the polynomial feature, n-by-d matrix
        
        # make prediction
        pre_n,pre_d = pre_fit_X.shape
        pre_fit_X = np.c_[np.ones((pre_n,1)),pre_fit_X]    # Add a row of ones for the bias term
        pre_value = pd.DataFrame(pre_fit_X*self.theta)
        return pre_value  # make prediction