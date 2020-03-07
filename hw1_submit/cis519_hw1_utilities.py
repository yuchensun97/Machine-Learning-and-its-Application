import math
import numpy as np
import random
import pandas as pd
from sklearn import tree


def cross_validated_accuracy(DecisionTreeClassifier, X, y, num_trials, num_folds, random_seed):
    random.seed(random_seed)
    """
     Args:
          DecisionTreeClassifier: An Sklearn DecisionTreeClassifier (e.g., created by "tree.DecisionTreeClassifier(criterion='entropy')")
          X: Input features
          y: Labels
          num_trials: Number of trials to run of cross validation
          num_folds: Number of folds (the "k" in "k-folds")
          random_seed: Seed for uniform execution (Do not change this) 

      Returns:
          cvScore: The mean accuracy of the cross-validation experiment

      Notes:
          1. You may NOT use the cross-validation functions provided by Sklearn
    """
    



  ## TODO ##
  # loof for each trial
    y = pd.DataFrame(y)
    column = list(y.columns)
    y = pd.Series(y[column[0]].tolist())
    accuracy = np.zeros((num_trials,num_folds)) #define a matrix to store each test's 
    rows = X.shape[0] # get the number of rows in the dataframe
    test_row = math.floor(rows/num_folds) # round down the number of test cases
    for k in range(0,num_trials):
        idx = list(range(0,rows)) #get the index of the dataframe
        random.shuffle(idx) #get the shuffle index of the dataframe
        # get the training data after shuffle
        X_backup = X.copy()
        y_backup = y.copy()
        Shuffle_X = X_backup.iloc[idx,:] 
        Shuffle_y = y_backup.iloc[idx] 
        Shuffle_X = Shuffle_X.reset_index(drop=True) # reset index
        Shuffle_y = Shuffle_y.reset_index(drop=True) # reset index
        
        for i in range(0,num_folds):
            # get the test data
            X_test = Shuffle_X.iloc[i*test_row:(i+1)*test_row,:]
            y_test = Shuffle_y.iloc[i*test_row:(i+1)*test_row]
            
            #get the training data
            test_idx = list(range(i*test_row,(i+1)*test_row))
            X_train = Shuffle_X.drop(Shuffle_X.index[test_idx],axis =0)
            y_train = Shuffle_y.drop(Shuffle_y.index[test_idx],axis=0)
            
            #Train the decision tree
            clf = DecisionTreeClassifier 
            clf = clf.fit(X_train,y_train)
            
            # Measure the test error
            y_pred = clf.predict(X_test) # give the predicted y lables
            accuracy[k][i] = (y_test == y_pred).sum() / X_test.shape[0] #store the accuracy in a single test
            
    # calculate the mean accuracy of the cross-validation experiment
    cvScore = np.mean(accuracy)
    
    
    return cvScore

def automatic_dt_pruning(DecisionTreeClassifier, X, y, num_trials, num_folds, random_seed):
    random.seed(random_seed)
    """
    Returns the pruning parameter (i.e., ccp_alpha) with the highest cross-validated accuracy

    Args:
          DecisionTreeClassifier  : An Sklearn DecisionTreeClassifier (e.g., created by "tree.DecisionTreeClassifier(criterion='entropy')")      
          X (Pandas.DataFrame)    : Input Features
          y (Pandas.Series)       : Labels
          num_trials              : Number of trials to run of cross validation
          num_folds               : Number of folds for cross validation (The "k" in "k-folds") 
          random_seed             : Seed for uniform execution (Do not change this)


      Returns:
          ccp_alpha : Tuned pruning paramter with highest cross-validated accuracy

      Notes:
          1. Don't change any other Decision Tree Classifier parameters other than ccp_alpha
          2. Use the cross_validated_accuracy function you implemented to find the cross-validated accuracy

    """


    ## TODO ##
    # get the effective alphas for the training data
    clf = DecisionTreeClassifier
    ccp_alphas = np.arange(0,1,0.001)
    params = {'ccp_alpha':ccp_alphas}
    
    # change ccp_alpha
    clfs = []
    accuracy = []
    idx = 0
    for i,j in params.items():
        for val in j:
            clf_tree = clf.set_params(**{i:val}) # set the ccp_alpha value without changing other parameter
            single_accuracy = cross_validated_accuracy(clf_tree,X,y,num_trials,num_folds,random_seed) # get the cross validated accuracy
            print(single_accuracy)
            accuracy.append(single_accuracy)
            if idx !=0:
                if accuracy[idx]<accuracy[idx-1]:
                    break
            idx+=1
    ccp_alpha = ccp_alphas[idx-1] # the most accurate ccp alpha
        
    return ccp_alpha