#!/usr/bin/env python
# coding: utf-8

# #CIS 419/519 HW0 iPython Notebook
# 
# Complete the answers to Questions 5 and 6 by completing this notebook.
# 

# # 5.) Dynamic Programming

# In[2]:


import numpy as np

def CompletePath(s, w, h) -> str:
    '''This function is used to escape from a room whose size is w * h.
    You are trapped in the bottom-left corner and need to cross to the
    door in the upper-right corner to escape.
    @:param s: a string representing the partial path composed of {'U', 'D', 'L', 'R', '?'}
    @:param w: an integer representing the room width
    @:param h: an integer representing the room length
    @:return path: a string that represents the completed path, with all question marks in with the correct directions
    or None if there is no path possible.
    '''
    
    # # TODO # #
    
    #error-check argument#
    assert isinstance(s,str), "s is a string"
    assert isinstance(w,int), "w is an int"
    assert isinstance(h,int), "h is an int"
    
    #Algorithmn starts here#
    dir = {'U':(0,1),'D':(0,-1),'L':(-1,0),'R':(1,0),"?":(0,0)}
    next = "UDLR"
    start = (1,1) # the start point
    goal = (w,h) # the goal point
    x,y = start # the current point
    mark = np.zeros((w+1,h+1)) #mark the passed point
    mark[1][1] = 1 #mark the start point
    idx = 0  # mark the index of the string
    for i in s:
        dx,dy=dir[i]
        if not(dx==0 and dy==0): #process the path if there is no ?

            #current point's coordinate
            x+=dx 
            y+=dy

            #is the point out of range?
            if x<1 or x>w or y<1 or y>h:
                return
            else:
                mark[x][y]=1#mark the current point

            #is the point reach the goal?
            if x==w and y==h :
                if idx == len(s)-1:
                    return s
                else:
                    return

            if idx==len(s)-1:
                return
                
        else: #process the path if there is ?

            # is the point reach the goal?
            if x==w and y==h:
                return s

            #enumerate 4 directions
            for j in next:
                dx,dy=dir[j]
                
                # next point's coordinate
                next_x=x+dx
                next_y=y+dy
                
                # is next point out of the range?
                if next_x<1 or next_x>w or next_y<1 or next_y>h:
                    continue
                
                # is next point marked?
                if mark[next_x][next_y] ==0:
                    mark[next_x][next_y] =1 # mark the next point
                    s = s[:idx]+j+s[idx+1:] #update the new string
                    new_path = CompletePath(s,w,h) #try next point
                    mark[next_x][next_y] =0 #unmark the next point
                    if new_path:
                        return new_path
                    else:
                        continue
                        
        idx+=1 # index increase
    pass


# # 6.) Pandas Data Manipulation

# In this section, we use the `Pandas` package to carry out 3 common data manipulation tasks :
# 
# * **Calculate missing ratios of variables**
# * **Create numerical binary variables**
# * **Convert categorical variables using one-hot encoding**
# 
# For the exercise, we will be using the Titanic dataset, the details of which can be found [here](https://www.kaggle.com/c/titanic/overview). For each of the data manipulation tasks, we have defined a skeleton for the python functions that carry out the given the manipulation. Using the function documentation, fill in the functions to implement the data manipulation.
# 

# In[4]:


import pandas as pd
import numpy as np


# **Dataset Link** : https://github.com/rsk2327/CIS519/blob/master/train.csv
# 
# 
# The file can be downloaded by navigating to the above link, clicking on the 'Raw' option and then saving the file.
# 
# Linux/Mac users can use the `wget` command to download the file directly. This can be done by running the following code in a Jupyter notebook cell
# 
# ```
# !wget https://github.com/rsk2327/CIS519/blob/master/train.csv
# ```
# 
# 

# In[15]:


# Read in the datafile using Pandas

df = pd.read_csv("train.csv")            # # TODO # #


# In[ ]:


def getMissingRatio(inputDf):
    """
    Returns the percentage of missing values in each feature of the dataset.
    
    Ensure that the output dataframe has the column names: Feature, MissingPercent

    Args:
        inputDf (Pandas.DataFrame): Dataframe to be evaluated


    Returns:
        outDf (Pandas.DataFrame): Resultant dataframe with 2 columns (Feature, MissingPercent)
                                  Each row corresponds to one of the features in `inputDf`

    """
    
    
    ## TODO ##
    
    out = inputDf.isnull().sum()/len(inputDf) # sum of NaN/length of the column
    outDf=pd.DataFrame({'Feature':out.index, 'MissingPercent':out.values}) # naming the feature and columns
    
    return outDf


# In[20]:


def convertToBinary(inputDf, feature):
    """
    Converts a two-value (binary) categorical feature into a numerical 0-1 representation and appends it to the dataframe
    
    Args:
        inputDf (pandas.DataFrame): Input dataframe
        variable (str) : Categorical feature which has to be converted into a numerical (0-1) representation
        
    Returns:
        outDf : Resultant dataframe with the original two-value categorical feature replaced by a numerical 0-1 feature

    """
    
    ## TODO ##
    feature_list = inputDf[feature] #get the data of the categorical
    feature_new = feature_list.dropna(axis=0) #filter the NaN element
    element = set(feature_new) #sort the different element in the feature str
    element = list(element) # convert the set to list, for convenience
    
    def elementConvert(x):
        """ Convert a single element to 0 or 1
        Args:
            x: a single element
        
        Returns:
            x: 0 or 1 or NaN
        """
        if x == 0:
            x=0
        elif x == 1:
            x=1
        elif x == element[0]: # the fisr-appear element set to 0
            x=0
        elif x == element[1]: # the second-appear element set to 1
            x=1
        elif np.isnan(x):
            x=np.nan
        return x
    
    if len(element) == 2:
        inputDf[feature] = inputDf[feature].apply(lambda x:elementConvert(x)) #replace the original data to 0-1
        outDf = inputDf
    else:
        print("This is not a binary data-type")
        outDf = inputDf
    
    return outDf


# In[22]:


def addDummyVariables(inputDf, feature):
    """
    Create a one-hot-encoded version of a categorical feature and append it to the existing 
    dataframe.
    
    After one-hot encoding the categorical feature, ensure that the original categorical feature is dropped
    from the dataframe so that only the one-hot-encoded features are retained.
    
    For more on one-hot encoding (OHE) : https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f

    Arguments:
        inputDf (Pandas.DataFrame): input dataframe
        feature (str) : Feature for which the OHE is to be performed


    Returns:
        outDf (Pandas.DataFrame): Resultant dataframe with the OHE features appended and the original feature removed

    """
    
    
    ## TODO ##
    feature_list = inputDf[feature] #get the feature categorical
    feature_new = feature_list.dropna(axis=0) #filter the NaN element
    element = set(feature_new) #sort out the unique element
    element = list(element) #make it ordered
    idx = 0 #

    # one-hot-encoding
    for i in feature_list:
        for j in element:
            str_j = feature+'_'+str(j) #name the OHE feature
            if i == j:
                inputDf.loc[idx,str_j]=1 #append the dataframe
            else:
                inputDf.loc[idx,str_j]=0 #append the dataframe
        idx+=1

    del inputDf[feature] #remove the original feature
    outDf = inputDf

    return outDf
