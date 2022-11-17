# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 14:48:40 2022

@author: romai

This code allows to vectorize an AS according to Laban and returns a pandas dataframe.
Two ways to call the vectorizer with :

- VectorizerLaban: takes as input a pandas dataframe and outputs another dataframe from the vectorized AS

- VectorizerLabanFromFilename: takes as input a path from the AS and returns another dataframe from the AS to vectorize

"""


import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings("ignore")

def createXYZJointVariables(dataframe):
    '''
    Create a global pandas.DataFrame variable named ~joint_name~Coordinates for each joint present in the dataframe passed as argument representing the x, y, z coordinates of the position of the jointure (as columns) at each instant (as rows).
   
    Parameters
    ----------
    dataframe : pandas.DataFrame
        Pandas dataframe that must have a "joint_name" column and x, y, z coordinates at columns indices 4:7 (the name of them hence does not matter).
    
    Returns
    -------
    None   
    '''
    
    for jointure in dataframe.joint_name.unique():
        globals()[str(jointure)] = dataframe.loc[dataframe["joint_name"]==jointure].iloc[:,4:7].reset_index(drop=True)
        #create separate X Y Z dataframe variable for each jointure

def arrayCoordonateJointForATime(df,i):
    '''
    Create an array of the cartesien coordonate of the joint at a definite time

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset of the joint 
    jointName : String
        Joint's name which one we want the cartesien coordonate
    i : Integer
        Index of the sample

    Returns
    -------
    Array : numpy.array
        3D cartesien coordonate

    '''
    Array=np.array([df['x'].iloc[i],df['y'].iloc[i],df['z'].iloc[i]])
    return Array

def angleBetweenTwoVectors(vecA,vecB):
    '''
    Compute the angle between two vectors, in degree

    Parameters
    ----------
    vecA : numpy.array
        First vector
    vecB : numpy.array
        Second vetcor

    Returns
    -------
    angle : Float
        The angle between the two vectors

    '''
    angle=np.degrees(np.arccos(np.dot(vecA,vecB)/(np.linalg.norm(vecA)*np.linalg.norm(vecB))))
    return angle

def QuickHullSeries(df):
    '''
    Compute the Quick hull for each frame 

    (see "Robust human action recognition system using Laban Movement Analysis" by InsafAjili*,MalikMallem,Jean-YvesDidier,2017 )

    Parameters
    ----------
    df : TYPpandas.DataFrame
        Dataset of the Life moment we want to study

    Returns
    -------
    QuickHull : pandas.Series
        return a serie of the quick hull at each frame

    '''
    list_QuickHull=[]
    for i in range(0,len(df),25):
        list_QuickHull.append(ConvexHull(df[['x','y','z']].iloc[i:i+24]).volume)
    QuickHull=pd.Series(list_QuickHull)
    return QuickHull

def FlatteningSeries(df_joint_1,df_joint_2,df_joint_3,df_joint_4,df_joint_5):
    '''
    Compute the flattening of the life moment study.
    
    The flattening is defined in terms of major and minoraxis. 
    Here the major axis is the maximum between two distances,
    the first one is the distance between handsjoints (dhands) 
    and the second one is the distance between neck and shoulder center joints (dNeckShc).
    
    flattening = (max(dhands,dNeckSch)-min(dhands,dNeckSch)/max(dhands,dNeckSch)) 
    
    (see "Robust human action recognition system using Laban Movement Analysis" by InsafAjili*,MalikMallem,Jean-YvesDidier,2017 )

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset of the Life moment we want to study

    Returns
    -------
    Flattening : pandas.Series
        return a serie corresponding of the flattening of the Dataframe

    '''
    list_flattening=[]
    for i in range(len(df_joint_1)):
        distance_hand=np.linalg.norm(arrayCoordonateJointForATime(df_joint_1,i) \
                                     -arrayCoordonateJointForATime(df_joint_2,i))    
        distance_neck_shoulderCenter=np.linalg.norm((arrayCoordonateJointForATime(df_joint_3,i) \
                                              +arrayCoordonateJointForATime(df_joint_4,i))/2 \
                                              -arrayCoordonateJointForATime(df_joint_5,i))
        list_flattening.append((max([distance_hand,distance_neck_shoulderCenter] \
                            -min([distance_hand,distance_neck_shoulderCenter])) \
                            /max([distance_hand,distance_neck_shoulderCenter])))
    Flattening=pd.Series(list_flattening)
    return Flattening

def directionnalMovement(df):
    '''
    Compute the directionnalMovement of a joint 
    
    Directionnal movement is the summation of the angles, frormed by the position of a joint between 3 frames, over the time 

    (see "Robust human action recognition system using Laban Movement Analysis" by InsafAjili*,MalikMallem,Jean-YvesDidier,2017 )

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset of the Life moment we want to study
    jointName : String
        Joint's name which one we want the cartesien coordonate

    Returns
    -------
    DirectionalMovement : pandas.Series
        return a serie corresponding of the directionnal movement of a joint ofthe Dataframe

    '''
    list_directionnal_Movement=[]
    for i in range(1,len(df['x'])-1):
        vecA=arrayCoordonateJointForATime(df,i-1)-arrayCoordonateJointForATime(df,i)
        vecB=arrayCoordonateJointForATime(df,i)-arrayCoordonateJointForATime(df,i+1)
        if i>1: 
            list_directionnal_Movement.append(list_directionnal_Movement[-1]+angleBetweenTwoVectors(vecA,vecB))
        else: # trait the case of the first position
            list_directionnal_Movement.append(angleBetweenTwoVectors(vecA,vecB))
    DirectionalMovement=pd.Series(list_directionnal_Movement)
    return DirectionalMovement

def angleSeries(df_jointName1,df_jointName2,df_jointName3):
    '''
    return a serie corresponding to the angle between three joint over the time

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset of the Life moment we want to study
    jointName1 : String
        First Joint's name 
    jointName2 : String
        Second Joint's name
    jointName3 : String
        Third Joint's name 

    Returns
    -------
    angle : pandas.Series
        series corresponding to the angle between three joint over the time

    '''
    list_angle=[]
    for i in range(1,len(df_jointName1)):
        vecA=arrayCoordonateJointForATime(df_jointName1,i)-arrayCoordonateJointForATime(df_jointName2,i)
        vecB=arrayCoordonateJointForATime(df_jointName2,i)-arrayCoordonateJointForATime(df_jointName3,i)
        list_angle.append(np.degrees(angleBetweenTwoVectors(vecA,vecB)))
    angle=pd.Series(list_angle)
    return angle

def distanceBetweenTwoJoints(df_jointName1,df_jointName2,axis='y'):
    '''
    
    return the distance according one axis between two joints over the time 

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset of the Life moment we want to study
    jointName1 : String
        First Joint's name 
    jointName2 : String
        Second Joint's name
    axis : String, optional
        axis according which one we want the distance. The default is 'y'.

    Returns
    -------
    distance : pandas.Series
        Series corresponding to the distance between two joints

    '''
    distance=(df_jointName1[axis].reset_index(drop=True) - df_jointName2[axis].reset_index(drop=True))
    return distance

def carving(df,axis1,axis2):
    '''
    Return the carving of a joint according a map definite by two axis
    
    d = sqrt(sum((Pje-Pse)²)
    
    where P is the position feature, j represents each joint considered at each frame,sis the spine joint at initial frame andebelongs to oneof the following sets{x,y},{y,z}and{z,x}for each considered projection.

    (see "Robust human action recognition system using Laban Movement Analysis" by InsafAjili*,MalikMallem,Jean-YvesDidier,2017 )

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset of the Life moment we want to study
    jointName : String
        Joint's name which one we want the carving
    axis1 : String {'x','y' or 'z'}
     first axis to definite the map
    axis2 : String {'x','y' or 'z'}
        Second axis to definite the map

    Returns
    -------
    carving : pandas.Series
        Series corresponding to the carving of a joint according a map definite by two axis

    '''
    carving=((df[axis1].reset_index(drop=True)**2 + df[axis2].reset_index(drop=True)**2)**(1/2))
    return carving

def VectorizerLaban(ms,functions=['mean','std','min','max','median'] ):
    '''
    return a dataframe which one conten the vector of the life moment we want to study

    Parameters
    ----------
    ms : pandas.DataFrame
        Dataset of the Life moment we want to study
    function : list of String, optional
        function that we want to apply on our differerent Series of features. The default is ['mean','std','min','max','median'].
    joints_carving : List of String, optional
        joint on which one we want to compute the carving. The default is ['HandLeft','HandRight','FootLeft','FootRight','Head'].

    Returns
    -------
    df_series : pandas.DataFrame
        dataFrame with the laban's vector.

    '''
    createXYZJointVariables(ms)
    joints_carving=['HandLeft','HandRight','FootLeft','FootRight','Head']
    QuickHull=QuickHullSeries(ms)
    angle_up_left=angleSeries(globals() ['WristLeft'],globals()['ElbowLeft'],globals()['ShoulderLeft'])
    angle_up_Right=angleSeries(globals() ['WristRight'],globals() ['ElbowRight'],globals() ['ShoulderRight'])
    angle_down_left=angleSeries(globals() ['HipLeft'],globals() ['KneeLeft'],globals() ['FootLeft'])
    angle_down_Right=angleSeries(globals() ['HipRight'],globals() ['KneeRight'],globals() ['FootRight'])
    flattening=FlatteningSeries(df_joint_1=globals() ['HandLeft'],df_joint_2=globals() ['HandRight'],df_joint_3=globals() ['ShoulderLeft'],df_joint_4=globals() ['ShoulderRight'],df_joint_5=globals() ['Neck'])
    dm_hand_left=directionnalMovement(globals() ['HandLeft'])
    dm_hand_right=directionnalMovement(globals() ['HandRight'])
    
    df_series=pd.DataFrame(index=['0'])
    
    
    
    for function in functions:
        df_series[f'{function}_distance_hip_FootLeft']=distanceBetweenTwoJoints(globals() ['HipLeft'],globals() ['FootLeft']).agg([f'{function}'])[0]
        df_series[f'{function}_distance_hip_FootRight']=distanceBetweenTwoJoints(globals() ['HipRight'],globals() ['FootRight']).agg([f'{function}'])[0]
        df_series[f'{function}_dm_hand_left']=dm_hand_left.agg([f'{function}'])[0]
        df_series[f'{function}_dm_hand_right']=dm_hand_right.agg([f'{function}'])[0]
        df_series[f'{function}_QuickHull']=QuickHull.agg([f'{function}'])[0]
        df_series[f'{function}_flattening']=flattening.agg([f'{function}'])[0]
        df_series[f'{function}_angle_up_left']=angle_up_left.agg([f'{function}'])[0]
        df_series[f'{function}_angle_up_Right']=angle_up_Right.agg([f'{function}'])[0]
        df_series[f'{function}_angle_down_left']=angle_down_left.agg([f'{function}'])[0]
        df_series[f'{function}_angle_down_Right']=angle_down_Right.agg([f'{function}'])[0]
        
        for joint_name in joints_carving:
            df_series[f'{function}_carving_{joint_name}_XY']=carving(globals()[joint_name],'x','y').agg([f'{function}'])[0]
            df_series[f'{function}_carving_{joint_name}_XZ']=carving(globals()[joint_name],'x','z').agg([f'{function}'])[0]
            df_series[f'{function}_carving_{joint_name}_ZY']=carving(globals()[joint_name],'z','y').agg([f'{function}'])[0]
            
    return df_series

def VectorizerLabanFromFilename(filename):

    '''
    return a dataframe which one contain the vector of the life moment we want to study

    Parameters
    ----------
    filename : String
        name of the csv file
    Returns
    -------
    df: pandas.DataFrame
        dataFrame with the laban's vector.

    '''

    ms=pd.read_csv(filename, names=["id_Camera", "date", "index", "id_individual", "x", "y", "z", "joint_name", "track_accuracy"])

    df= VectorizerLaban(ms=ms)
    return df

    