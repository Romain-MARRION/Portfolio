# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from LABAN import Vectorializer_Laban as vl
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

df_path=os.path.abspath('df_target.csv')
lms_path=os.path.abspath('lms_back_sitting_ego')
df_target=pd.read_csv(df_path).astype('float32')
df=pd.read_csv(lms_path,names=['id_Camera', 'date', 'index','id_individual','x','y','z','joint_name','track_accuracy'])

def test_VectorizerLaban():
    data=vl.VectorizerLaban(ms=df).astype('float32')
    for col in df_target:
        assert data[col].iloc[0]==df_target[col].iloc[0]
    
    
