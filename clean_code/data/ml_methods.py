import pandas as pd
import re
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec
import numpy as np
import re
import math 
from scipy import fftpack
from math import sqrt
from csv import reader
import datetime as dt
from datetime import datetime
import pdb
import seaborn as sns
import dask
import scipy as sp
import joblib


# for dask as a single machine 
from dask.distributed import Client, LocalCluster

# for ml computation
from dask_jobqueue import SLURMCluster
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.compose import make_column_transformer
from sklearn.impute import  SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder, MinMaxScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut

    




# gradientBoostingregression 


# randomForest


