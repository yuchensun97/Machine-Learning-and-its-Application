import pandas as pd

import numpy as np
from numpy import linalg as LA
from numpy.linalg import *

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold

from collections import Counter

import math
'''
    Template for polynomial regression
'''

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
        self.tune_mean = 0
        self.tune_std = 0
        self.tune_bool = 0
        self.cv = 2


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
        y_df = y.copy()    # for tuning lambda
        y = y.to_numpy()    
        fit_X = self.polyfeatures(X,self.degree)    # perform polynomial feature expansion, n-by-d matrix
        fit_X_df = fit_X.copy()    # for tuning lambda
        fit_X = fit_X.to_numpy()    # convert the dataframe to np.array
        X_mean = np.mean(fit_X, axis=0)    # compute mean value of each feature, types: row
        self.mean = X_mean
        X_std = np.std(fit_X, axis=0)     # compute std value of each feature, types: row
        self.std = X_std
        fit_X = (fit_X-X_mean)/X_std      # standarize the polynomial feature, n-by-d matrix
        
        n,d = fit_X.shape
        y = y.reshape(n,1)
        
        theta_to_pre = self.theta
        theta = self.theta

        # automatically tune the lambda values
        if self.tuneLambda:

            best_lambda_ls = []
            valid_row = math.floor(n/self.cv)

            for i in range(self.cv):
                self.tune_bool=1
                # get the valid data
                X_valid = fit_X_df.iloc[i*valid_row:(i+1)*valid_row,0]
                y_valid = y_df.iloc[i*valid_row:(i+1)*valid_row]

                # get the trainning data
                valid_idx = list(range(i*valid_row,(i+1)*valid_row))
                X_train = fit_X_df.drop(fit_X_df.index[valid_idx],axis=0)
                y_train = y_df.drop(y_df.index[valid_idx],axis=0)

                # pre-processing trainning data
                X_train = X_train.to_numpy()
                self.tune_mean = np.mean(X_train,axis=0)
                self.tune_std = np.std(X_train,axis=0)
                X_train = (X_train-self.tune_mean)/self.tune_std

                # convert to np.array
                y_valid = y_valid.to_numpy()
                y_train = y_train.to_numpy()

                n=len(y_train)
                y_train = y_train.reshape(n,1)
                X_train = np.c_[np.ones((n,1)),X_train]

                judge = []
                for Lambda in self.Lambdavalue:

                    # obtain the theta for each lambda
                    self.theta = self.gradientDescent(X_train,y_train,theta,Lambda)

                    # compare with valid data
                    y_points = self.predict(X_valid)
                    y_points = y_points.to_numpy()
                    y_judge = LA.norm(y_points-y_valid)
                    judge.append(y_judge)
                
                # return the index of most accurate prediction
                min_judge = min(judge)
                lambda_idx = judge.index(min_judge)

                # return the best lambda
                best_lambda = self.Lambdavalue[lambda_idx]
                best_lambda_ls.append(best_lambda)

            #  return the best lambda
            Lambda = min(best_lambda_ls)
            self.tune_bool=0
            
        else:
            Lambda = self.Lambda

        # add ones to the feature
        n = len(y)
        fit_X = np.c_[np.ones((n,1)),fit_X]   
        self.theta = self.gradientDescent(fit_X,y,theta_to_pre,Lambda)
        return
    
    def gradientDescent(self,X,y,theta,Lambda):
        """
        This function used to calculate the theta by iteration
        X: np.array
        y: np.array
        theta: np.array
        Lambda: np.array
        """
        n,d = X.shape
        while True:
            yhat = X*theta
            old_cost = self.computeCost(X,y,theta,Lambda)
            theta = theta-(X.T*(yhat-y))*(self.alpha/n)-self.alpha*Lambda*theta
            new_cost = self.computeCost(X,y,theta,Lambda)
            eps = abs(new_cost-old_cost)
            if eps<1E-4:
                break
        return theta

    def computeCost(self,X,y,theta,Lambda):
        """
        Compute cost function of current theta
        Input:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            theta is a d-dimensional numpy vector
            Lambda is is the regulariztion, n-dimensional numpy vector
        Returns:
            cost of current theta
        """
        n,d = X.shape
        yhat = X*theta
        cost = 1/n*(yhat-y).T*(yhat-y)+Lambda*theta.T*theta    # cost function
        return cost
        
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
        if self.tune_bool==1:
            pre_fit_X = self.polyfeatures(X,self.degree)    # perform polynomial feature expansion, n-by-d matrix
            pre_fit_X = pre_fit_X.to_numpy()
            pre_fit_X = (pre_fit_X-self.tune_mean)/self.tune_std      # standarize the polynomial feature, n-by-d matrix
        else:
            pre_fit_X = self.polyfeatures(X,self.degree)    # perform polynomial feature expansion, n-by-d matrix
            pre_fit_X = pre_fit_X.to_numpy()
            pre_fit_X = (pre_fit_X-self.mean)/self.std      # standarize the polynomial feature, n-by-d matrix
        
        # make prediction
        pre_n,pre_d = pre_fit_X.shape
        pre_fit_X = np.c_[np.ones((pre_n,1)),pre_fit_X]    # Add a row of ones for the bias term
        pre_value = pd.DataFrame(pre_fit_X*self.theta)
        return pre_value  # make prediction

import numpy as np
import matplotlib.pyplot as plt

def test_polyreg_univariate():
    '''
        Test polynomial regression
    '''

    # load the data
    filepath = "http://www.seas.upenn.edu/~cis519/spring2020/data/hw2-polydata.csv"
    df = pd.read_csv(filepath, header=None)

    X = df[df.columns[:-1]]
    y = df[df.columns[-1]]

    # regression with degree = d
    d = 8
    LambdaValues = [0,0.001,0.003,0.006,0.01,0.03,0.06,0.1,0.3,0.6,1,3,10]
    model = PolynomialRegression(degree=8,regLambda=0,tuneLambda=True,regLambdaValues=LambdaValues)
    model.fit(X, y)
    
    # output predictions
    xpoints = pd.DataFrame(np.linspace(np.max(X), np.min(X), 100))
    ypoints = model.predict(xpoints)

    # plot curve
    plt.figure()
    plt.plot(X, y, 'rx')
    plt.title('PolyRegression with d = '+str(d))
    plt.plot(xpoints, ypoints, 'b-')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

if __name__ == "__main__":
    test_polyreg_univariate()