# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # CIS 419/519 
# #**Homework 4 : Adaboost and the Challenge**

# %%
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from scipy import stats
# %% [markdown]
# # Data Info

# %%

# load the training data
baseDir = ''
X = pd.read_csv(baseDir+'ChocolatePipes_traindata.csv')

# output debugging info
print('Training Data Info')
print(X.shape)

# load the training label
baseDir_label = ''
y = pd.read_csv(baseDir+'ChocolatePipes_trainLabels-2.csv')

# output debugging info
print('Testing Data Info')
print(y.shape)
X.head()

# %%
print('Percentage of instances with missing features:')
print(X.isnull().sum(axis=0)/X.shape[0])
print()
print('Class information:')
print(y['label'].value_counts())
X.describe()

#%% [markdown]
# # Matching Data

#%%
# find the intersecting id
X = X.set_index('id')
y = y.set_index('id')
y = y.loc[~y.index.duplicated(keep='first')]    # drop the redundant label
correspond_id = X.index.intersection(y.index)

# re-set the dataframe
X = X.loc[correspond_id,:]
y = y.loc[correspond_id,['label']]

# concat the full data
full_data = X.copy()
full_data=full_data.assign(label=y['label'].values)

#%% [markdown]
## Matched Data Info
#%%
print('training data info')
print(full_data.shape)
print()
print('Percentage of instances with missing features:')
print(full_data.isnull().sum(axis=0)/full_data.shape[0])
print()
print('Class information:')
print(full_data['label'].value_counts())

# visualize correlation matrix
import matplotlib.pyplot as plt

# print important feature using correlation function
cor = full_data.corr()
cor_target = abs(cor['label'])
# important = cor_target[cor_target>0.2] # select the highly relevant features
print(cor_target)

plt.matshow(full_data.corr())
plt.show()

#%% [markdown]
## Preprocessing Data

#%%
# preprocessing starts here
full_data = full_data.drop(['Recorded by','Size of chocolate pool','Date of entry','Year constructed'],axis=1)    # drop these two feature because the former has the same value and the later has too many outliers
full_data['Height of pipe'] = full_data['Height of pipe'].replace(0, np.nan)
full_data['management_group']=full_data['management_group'].replace('unknown',np.nan)

# full_data = full_data[['Height of pipe','Region code','Location','Cocoa farm','Oompa loompa management','Type of pump','management','Payment scheme','chocolate_quality','chocolate_quantity','chocolate_source','pipe_type','label']]

# OHE categorical feature
non_numeric = full_data.select_dtypes(include='object')
cols = non_numeric.columns
full_data[cols]=full_data[cols].fillna(full_data.mode().iloc[0])    # impute with mode
full_data = pd.get_dummies(full_data, prefix=cols, columns=cols)

# the remaining categorical feature
categoric = ['Cocoa farm','Oompa loompa management','Type of pump','management','Payment scheme','chocolate_quality','chocolate_quantity','chocolate_source','pipe_type']
full_data[categoric]=full_data[categoric].fillna(full_data.mode().iloc[0])    # impute with mode
full_data = pd.get_dummies(full_data,prefix=categoric,columns=categoric)

# impute nan with mean in numeric columns
full_data = full_data.fillna(full_data.mean())

# remove outliers 
full_data = full_data[(np.abs(stats.zscore(full_data))<20).all(axis=1)]

# print important feature using correlation function
cor = full_data.corr()
cor_target = abs(cor['label'])
important = cor_target[cor_target>0.1] # select the highly relevant features
print(important)

X = full_data.drop(['label'],axis=1)
y = full_data['label']

# %% [markdown]
# # Adaboost-SAMME

# %%
import numpy as np
import math
from sklearn import tree

class BoostedDT:

    def __init__(self, numBoostingIters=100, maxTreeDepth=2):
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
        self.weight = []



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
        # initialize the weight vectors
        w = np.ones(n)/n
        self.clfs = []
        self.betas = []
        self.labels = list(set(y))
        K = len(pd.unique(y))

        def isWrong(y_pre,y_real):
            if y_pre==y_real:
                return 1
            else:
                return -1

        # adaboost-samme
        for i in range(self.numBoostingIters):
            clfs = tree.DecisionTreeClassifier(criterion='entropy',max_depth=self.maxTreeDepth,ccp_alpha=0.0001)
            self.clfs.append(clfs)
            weak_model = clfs.fit(X,y,sample_weight=w)    # train the weak model with instance weights 
            yhat = weak_model.predict(X)
            train_error = sum((yhat!=y)*w)    # compute the weighted training error of hypothesis
            beta = 1/2*(np.log((1-train_error)/train_error)+np.log(K-1))    # choose beta
            y_np = y.to_numpy().flatten()
            self.betas.append(beta)

            # update all instance weights
            for j in range(n):
                w[j] = w[j]*np.exp(-beta*isWrong(yhat[j],y_np[j]))

            w = w/w.sum()    # normalized data

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


        return result

# %% [markdown]
# # Test BoostedDT

# %%
# boosted DT
import numpy as np
from sklearn import datasets
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def test_boostedDT(df):

    # # load the data set
    # sklearn_dataset = datasets.load_breast_cancer()
    # # convert to pandas df
    # df = pd.DataFrame(sklearn_dataset.data,columns=sklearn_dataset.feature_names)
    # df['CLASS'] = pd.Series(sklearn_dataset.target)
    df.head()

    # split randomly into training/testing
    train, test = train_test_split(df, test_size=0.5, random_state=42)
    # Split into X,y matrices
    X_train = train.drop(['label'], axis=1)
    y_train = train['label']
    X_test = test.drop(['label'], axis=1)
    y_test = test['label']


    # train the decision tree
    modelDT = DecisionTreeClassifier()
    modelDT.fit(X_train, y_train)

    # train the boosted DT
    modelBoostedDT = BoostedDT(numBoostingIters=100, maxTreeDepth=10)
    modelBoostedDT.fit(X_train, y_train)

    # train sklearn's implementation of Adaboost
    modelSKBoostedDT = AdaBoostClassifier(DecisionTreeClassifier(criterion='entropy',max_depth=4), n_estimators=100)
    modelSKBoostedDT.fit(X_train, y_train)

    # output predictions on the test data
    ypred_DT = modelDT.predict(X_test)
    ypred_BoostedDT = modelBoostedDT.predict(X_test)
    ypred_SKBoostedDT = modelSKBoostedDT.predict(X_test)
    print(set(ypred_DT))
    print(pd.value_counts(ypred_BoostedDT))
    print(pd.value_counts(ypred_SKBoostedDT))

    # compute the training accuracy of the model
    accuracy_DT = accuracy_score(y_test, ypred_DT)
    accuracy_BoostedDT = accuracy_score(y_test, ypred_BoostedDT)
    accuracy_SKBoostedDT = accuracy_score(y_test, ypred_SKBoostedDT)

    print("Decision Tree Accuracy = "+str(accuracy_DT))
    print("My Boosted Decision Tree Accuracy = "+str(accuracy_BoostedDT))
    print("Sklearn's Boosted Decision Tree Accuracy = "+str(accuracy_SKBoostedDT))
    print()
    print("Note that due to randomization, your boostedDT might not always have the ")
    print("exact same accuracy as Sklearn's boostedDT.  But, on repeated runs, they ")
    print("should be roughly equivalent and should usually exceed the standard DT.")

test_boostedDT(full_data)

#%%
# SKlearn SVM
import numpy as np
from sklearn import datasets
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import svm

def test_SVM(df):

    df.head()

    # split randomly into training/testing
    train, test = train_test_split(df, test_size=0.5, random_state=42)
    # Split into X,y matrices
    X_train = train.drop(['label'], axis=1)
    y_train = train['label']
    X_test = test.drop(['label'], axis=1)
    y_test = test['label']

    clf = svm.SVC(kernel='rbf',gamma='auto')
    clf.fit(X_train,y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)

    print("The accuracy of SVM with rbf kernel is:"+str(accuracy))

test_SVM(full_data)

# %%
# Sklearn Random Forest
import numpy as np
from sklearn import datasets
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def test_RandomForest(df):

    # # load the data set
    # sklearn_dataset = datasets.load_breast_cancer()
    # # convert to pandas df
    # df = pd.DataFrame(sklearn_dataset.data,columns=sklearn_dataset.feature_names)
    # df['CLASS'] = pd.Series(sklearn_dataset.target)
    df.head()

    # split randomly into training/testing
    train, test = train_test_split(df, test_size=0.5, random_state=42)
    # Split into X,y matrices
    X_train = train.drop(['label'], axis=1)
    y_train = train['label']
    X_test = test.drop(['label'], axis=1)
    y_test = test['label']

    clf = RandomForestClassifier(max_depth=20,random_state=42)
    clf.fit(X_train,y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)

    print("The accuracy of random forest is:"+str(accuracy))

test_RandomForest(full_data)


# %%[markdown]
## Training grading data

#%%
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from scipy import stats
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

def gradingData(df):
    # load the grading data
    baseDir = ''
    X = pd.read_csv(baseDir+'ChocolatePipes_gradingTestData.csv')
    X = X.set_index('id')

    # output debugging info
    print('Training Data Info')

    # preprocessing starts here
    X = X.drop(['Recorded by','Size of chocolate pool','Date of entry','Year constructed'],axis=1)    # drop these two feature because the former has the same value and the later has too many outliers
    X['Height of pipe'] = X['Height of pipe'].replace(0, np.nan)
    X['management_group']=X['management_group'].replace('unknown',np.nan)

    # OHE categorical feature
    non_numeric = X.select_dtypes(include='object')
    cols = non_numeric.columns
    X[cols]=X[cols].fillna(X.mode().iloc[0])    # impute with mode
    X = pd.get_dummies(X, prefix=cols, columns=cols)

    # the remaining categorical feature
    categoric = ['Cocoa farm','Oompa loompa management','Type of pump','management','Payment scheme','chocolate_quality','chocolate_quantity','chocolate_source','pipe_type']
    X[categoric]=X[categoric].fillna(X.mode().iloc[0])    # impute with mode
    X = pd.get_dummies(X,prefix=categoric,columns=categoric)

    # impute nan with mean in numeric columns
    X = X.fillna(X.mean())

    # adding new columns to complete the test data
    col = X.columns.tolist()
    col.insert(col.index('Oompa loompa management_11.0')+1,'Oompa loompa management_12.0')
    col.insert(col.index('Type of pump_9')+1,'Type of pump_10')
    col.insert(col.index('pipe_type_5')+1,'pipe_type_6')
    X = X.reindex(columns=col)
    X['Oompa loompa management_12.0']=0
    X['pipe_type_6'] = 0
    X['Type of pump_10']=0

    # Split the traing data into X,y matrices
    X_train = df.drop(['label'], axis=1)
    y_train = df['label']

    # train the random forest
    modelRF = RandomForestClassifier(max_depth=20,random_state=42)
    modelRF.fit(X_train, y_train)

    # train the boosted DT
    modelBoostedDT = BoostedDT(numBoostingIters=100, maxTreeDepth=10)
    modelBoostedDT.fit(X_train, y_train)

    # output predictions on the test data
    ypred_RF = modelRF.predict(X)
    ypred_BoostedDT = modelBoostedDT.predict(X)

    # save the file
    # save the file
    ypred_RF = pd.DataFrame(list(ypred_RF))
    ypred_BoostedDT = pd.DataFrame(list(ypred_BoostedDT))

    ypred_RF.to_csv('predictions-grading-best.csv',index=True)
    ypred_BoostedDT.to_csv('predictions-grading-BoostedDT.csv',index=True)

gradingData(full_data)

#%% [markdown]
## Training leaderborad data

#%%
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from scipy import stats
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

def leaderboardData(df):
    # load the grading data
    baseDir = ''
    X = pd.read_csv(baseDir+'ChocolatePipes_leaderboardTestData.csv')
    X = X.set_index('id')

    # output debugging info
    print('Training Data Info')
    print(X.shape)

    # preprocessing starts here
    X = X.drop(['Recorded by','Size of chocolate pool','Date of entry','Year constructed'],axis=1)    # drop these two feature because the former has the same value and the later has too many outliers
    X['Height of pipe'] = X['Height of pipe'].replace(0, np.nan)
    X['management_group']=X['management_group'].replace('unknown',np.nan)

    # OHE categorical feature
    non_numeric = X.select_dtypes(include='object')
    cols = non_numeric.columns
    X[cols]=X[cols].fillna(X.mode().iloc[0])    # impute with mode
    X = pd.get_dummies(X, prefix=cols, columns=cols)

    # the remaining categorical feature
    categoric = ['Cocoa farm','Oompa loompa management','Type of pump','management','Payment scheme','chocolate_quality','chocolate_quantity','chocolate_source','pipe_type']
    X[categoric]=X[categoric].fillna(X.mode().iloc[0])    # impute with mode
    X = pd.get_dummies(X,prefix=categoric,columns=categoric)

    # impute nan with mean in numeric columns
    X = X.fillna(X.mean())

    # adding new columns to complete the test data
    col = X.columns.tolist()
    col.insert(col.index('Oompa loompa management_11.0')+1,'Oompa loompa management_12.0')
    X = X.reindex(columns=col)
    X['Oompa loompa management_12.0']=0

    # Split the traing data into X,y matrices
    X_train = df.drop(['label'], axis=1)
    y_train = df['label']

    # train the random forest
    modelRF = RandomForestClassifier(max_depth=20,random_state=42)
    modelRF.fit(X_train, y_train)

    # train the boosted DT
    modelBoostedDT = BoostedDT(numBoostingIters=100, maxTreeDepth=10)
    modelBoostedDT.fit(X_train, y_train)

    # output predictions on the test data
    ypred_RF = modelRF.predict(X)
    ypred_BoostedDT = modelBoostedDT.predict(X)

    # save the file
    # save the file
    ypred_RF = pd.DataFrame(list(ypred_RF))
    ypred_BoostedDT = pd.DataFrame(list(ypred_BoostedDT))

    ypred_RF.to_csv('predictions-leaderboard-best.csv',index=True)
    ypred_BoostedDT.to_csv('predictions-leaderboard-BoostedDT.csv',index=True)

leaderboardData(full_data)

# %%
# Train svm
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from scipy import stats
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.preprocessing import StandardScaler

def svmData(df):
    # load the grading data
    baseDir_1 = ''
    X_grad = pd.read_csv(baseDir_1+'ChocolatePipes_gradingTestData.csv')
    X_grad = X_grad.set_index('id')

    baseDir = ''
    X_leader = pd.read_csv(baseDir+'ChocolatePipes_leaderboardTestData.csv')
    X_leader = X_leader.set_index('id')

    # preprocessing starts here
    X_grad = X_grad.drop(['Recorded by','Size of chocolate pool','Date of entry','Year constructed'],axis=1)    # drop these two feature because the former has the same value and the later has too many outliers
    X_grad['Height of pipe'] = X_grad['Height of pipe'].replace(0, np.nan)
    X_grad['management_group']=X_grad['management_group'].replace('unknown',np.nan)

    # OHE categorical feature
    non_numeric = X_grad.select_dtypes(include='object')
    cols = non_numeric.columns
    X_grad[cols]=X_grad[cols].fillna(X_grad.mode().iloc[0])    # impute with mode
    X_grad = pd.get_dummies(X_grad, prefix=cols, columns=cols)

    # the remaining categorical feature
    categoric = ['Cocoa farm','Oompa loompa management','Type of pump','management','Payment scheme','chocolate_quality','chocolate_quantity','chocolate_source','pipe_type']
    X_grad[categoric]=X_grad[categoric].fillna(X.mode().iloc[0])    # impute with mode
    X_grad = pd.get_dummies(X_grad,prefix=categoric,columns=categoric)

    # impute nan with mean in numeric columns
    X_grad = X_grad.fillna(X_grad.mean())

    # adding new columns to complete the test data
    col = X_grad.columns.tolist()
    col.insert(col.index('Oompa loompa management_11.0')+1,'Oompa loompa management_12.0')
    col.insert(col.index('Type of pump_9')+1,'Type of pump_10')
    col.insert(col.index('pipe_type_5')+1,'pipe_type_6')
    X_grad=X_grad.reindex(columns=col)
    X_grad['Oompa loompa management_12.0']=0
    X_grad['pipe_type_6'] = 0
    X_grad['Type of pump_10']=0

    # standaridize
    X_grad_stand = X_grad.values #returns a numpy array
    scaler = StandardScaler()
    X_grad_scale = scaler.fit_transform(X_grad_stand)
    X_grad = pd.DataFrame(X_grad_scale)

    # preprocessing starts here
    X_leader = X_leader.drop(['Recorded by','Size of chocolate pool','Date of entry','Year constructed'],axis=1)    # drop these two feature because the former has the same value and the later has too many outliers
    X_leader['Height of pipe'] = X_leader['Height of pipe'].replace(0, np.nan)
    X_leader['management_group']=X_leader['management_group'].replace('unknown',np.nan)

    # OHE categorical feature
    non_numeric = X_leader.select_dtypes(include='object')
    cols = non_numeric.columns
    X_leader[cols]=X_leader[cols].fillna(X_leader.mode().iloc[0])    # impute with mode
    X_leader = pd.get_dummies(X_leader, prefix=cols, columns=cols)

    # the remaining categorical feature
    categoric = ['Cocoa farm','Oompa loompa management','Type of pump','management','Payment scheme','chocolate_quality','chocolate_quantity','chocolate_source','pipe_type']
    X_leader[categoric]=X_leader[categoric].fillna(X.mode().iloc[0])    # impute with mode
    X_leader = pd.get_dummies(X_leader,prefix=categoric,columns=categoric)

    # impute nan with mean in numeric columns
    X_leader = X_leader.fillna(X_leader.mean())

    # adding new columns to complete the test data
    col = X_leader.columns.tolist()
    col.insert(col.index('Oompa loompa management_11.0')+1,'Oompa loompa management_12.0')
    X_leader=X_leader.reindex(columns=col)
    X_leader['Oompa loompa management_12.0']=0

    # standaridize
    X_leader_stand = X_leader.values #returns a numpy array
    scaler = StandardScaler()
    X_leader_scale = scaler.fit_transform(X_leader_stand)
    X_leader = pd.DataFrame(X_leader_scale)
        
    # Split the traing data into X,y matrices
    X_train = df.drop(['label'], axis=1)
    y_train = df['label']

    # standaridize
    X_train_stand = X_train.values #returns a numpy array
    scaler = StandardScaler()
    X_train_scale = scaler.fit_transform(X_train_stand)
    X_train = pd.DataFrame(X_train_scale)

    # train sklearn's SVM with kernel rfb
    modelSVM = svm.SVC(kernel='rbf',gamma='auto')
    modelSVM.fit(X_train, y_train)

    # output predictions on the test data
    ypred_SVM_grad = modelSVM.predict(X_grad)
    ypred_SVM_leader = modelSVM.predict(X_leader)

    # save the file
    ypred_SVM_grad = pd.DataFrame(list(ypred_SVM_grad))
    ypred_SVM_leader = pd.DataFrame(list(ypred_SVM_leader))

    ypred_SVM_grad.to_csv('prediction-grading-SVC.csv',index=True)
    ypred_SVM_leader.to_csv('prediction-leaderboard-SVC.csv',index=True)

svmData(full_data)

# %%
