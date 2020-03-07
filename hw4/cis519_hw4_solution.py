import pandas as pd
import numpy as np

import math
from sklearn import tree
from sklearn import datasets
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class BoostedDT:

    def __init__(self, numBoostingIters=100, maxTreeDepth=3):
        '''
        Constructor

        Class Fields 
        clfs : List object containing individual DecisionTree classifiers, in order of creation during boosting
        betas : List of beta values, in order of creation during boosting
        '''

        self.clfs = None  # keep the class fields, and be sure to keep them updated during boosting
        self.betas = None 
        
        #TODO
        self.numBoostingIters = numBoostingIters
        self.maxTreeDepth = maxTreeDepth
        self.labels = []
        self.weight = []
        self.classes = []
        self.K = 0



    def fit(self, X, y, random_state=None):
        '''
        Trains the model. 
        Be sure to initialize all individual Decision trees with the provided random_state value if provided.
        
        Arguments:
            X is an n-by-d Pandas Data Frame
            y is an n-by-1 Pandas Data Frame
            random_seed is an optional integer value
        '''
        #TODO
        n,d = X.shape
        X = X.to_numpy()
        y = y.to_numpy().flatten()
        # initialize the weight vectors
        w = np.ones(n)/n
        self.clfs = []
        self.betas = []
        self.labels = list(set(y))
        K = len(np.unique(y))

        def isWrong(y_pre,y_real):
            if y_pre==y_real:
                return 1
            else:
                return -1

        # adaboost-samme
        for i in range(self.numBoostingIters):
            clfs = tree.DecisionTreeClassifier(max_depth=self.maxTreeDepth,random_state=random_state)
            self.clfs.append(clfs)
            weak_model = clfs.fit(X,y,sample_weight=w)    # train the weak model with instance weights 
            yhat = weak_model.predict(X)
            train_error = sum((yhat!=y)*w)    # compute the weighted training error of hypothesis
            beta = 1/2*(np.log((1-train_error)/train_error)+np.log(K-1))    # choose beta
            y_np = y
            self.betas.append(beta)

            # update all instance weights
            for j in range(n):
                w[j] = w[j]*np.exp(-beta*isWrong(yhat[j],y_np[j]))
            w = w/w.sum()    # normalized data
        
        self.classes = np.unique(y, axis=0)
        self.classes = np.array(self.classes).reshape(K,1)
        self.K = K


    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is an n-by-d Pandas Data Frame
        Returns:
            an n-by-1 Pandas Data Frame of the predictions
        '''
        #TODO
        X_np = X.to_numpy()
        n,d = X_np.shape
        prediction = 0
        result = np.zeros(n)
        for i in range(self.numBoostingIters):
            yhat_proba = self.clfs[i].predict_proba(X_np)
            weight_proba = np.multiply(self.betas[i],yhat_proba)
            prediction +=weight_proba

        for j in range(n):
            result[j] = self.labels[np.argmax(prediction[j,:])]

        result = pd.DataFrame(result)


        return result


# def test_boostedDT():

#     # # load the data set
#     # sklearn_dataset = datasets.load_iris()
#     # # convert to pandas df
#     # df = pd.DataFrame(sklearn_dataset.data,columns=sklearn_dataset.feature_names)
#     # df['CLASS'] = pd.Series(sklearn_dataset.target)
#     # df.head()

#     X = np.array([[0, 1], [2, 3], [0, -1], [2, -3], [2, 3]])
#     y = np.array([[1], [1], [-1], [1], [-1]])

#     X = pd.DataFrame(X)
#     y = pd.DataFrame(y)

#     # # split randomly into training/testing
#     # train, test = train_test_split(df, test_size=0.5, random_state=42)
#     # # Split into X,y matrices
#     # X_train = train.drop(['CLASS'], axis=1)
#     # y_train = train['CLASS']
#     # X_test = test.drop(['CLASS'], axis=1)
#     # y_test = test['CLASS']


#     # # train the decision tree
#     # modelDT = DecisionTreeClassifier()
#     # modelDT.fit(X_train, y_train)

#     # train the boosted DT
#     modelBoostedDT = BoostedDT(numBoostingIters=100, maxTreeDepth=2)
#     modelBoostedDT.fit(X, y)

#     # train sklearn's implementation of Adaboost
#     # modelSKBoostedDT = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=100)
#     # modelSKBoostedDT.fit(X_train, y_train)

#     # # output predictions on the test data
#     # ypred_DT = modelDT.predict(X_test)
#     # ypred_BoostedDT = modelBoostedDT.predict(X_test)
#     # ypred_SKBoostedDT = modelSKBoostedDT.predict(X_test)

#     # # compute the training accuracy of the model
#     # accuracy_DT = accuracy_score(y_test, ypred_DT)
#     # accuracy_BoostedDT = accuracy_score(y_test, ypred_BoostedDT)
#     # accuracy_SKBoostedDT = accuracy_score(y_test, ypred_SKBoostedDT)

#     # print("Decision Tree Accuracy = "+str(accuracy_DT))
#     # print("My Boosted Decision Tree Accuracy = "+str(accuracy_BoostedDT))
#     # print("Sklearn's Boosted Decision Tree Accuracy = "+str(accuracy_SKBoostedDT))
#     # print()
#     # print("Note that due to randomization, your boostedDT might not always have the ")
#     # print("exact same accuracy as Sklearn's boostedDT.  But, on repeated runs, they ")
#     # print("should be roughly equivalent and should usually exceed the standard DT.")

# if __name__ == "__main__":
#     test_boostedDT()