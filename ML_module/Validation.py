#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 11:46:00 2022

@author: rmarrion
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
def LeaveOneOut(model,X,y,period):
    """
    Train and test a model with a leave one out method and return the score,
    the mean squared error and the prediction of each split.
    
    Should be use with a model that contains fit, predict and score method.

    Parameters
    ----------
    model : 
        Machine learning model choosen
    X : pandas.DataFrame
        Features used to predict
    y : pandas.Series
        Target feature
    period : Integer, optional
            Time series period The default is 5000.

    Returns
    -------
    score : list
            Score of each model trained 
    error : error
            mean squared error of each model trained 
    predict : np.array
              prediction of each model train
    """
    score=[]
    predict=[]
    error=[]
    for i in range(0,len(X),period):
        if i==0:
            X_train=X.iloc[period:,:]
            y_train=y.iloc[period:]
            X_test=X.iloc[0:period,:]
            y_test=y.iloc[0:period]
        elif i==len(X)-period:
            X_train=X.iloc[0:i,:]
            y_train=y.iloc[0:i]
            X_test=X.iloc[i:i+period,:]
            y_test=y.iloc[i:i+period]
        else:
            X_train=pd.concat([X.iloc[0:i,:],X.iloc[i+period:,:]],axis=0)
            y_train=pd.concat([y.iloc[0:i],y.iloc[i+period:]],axis=0)
            X_test=X.iloc[i:i+period,:]
            y_test=y.iloc[i:i+period]
        model.fit(X_train,y_train)
        y_pred=model.predict(X_test)
        predict=np.concatenate([predict,y_pred])
        score.append(model.score(X_test,y_test))
        error.append(mean_squared_error(y_test,y_pred))
    return score,error,predict


   