import numpy as np
import sys
print(sys.version)
import sklearn
from sklearn import linear_model as lm
sklearn.__version__
import pandas as pd    
from sklearn.preprocessing import MinMaxScaler  
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score, recall_score
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
import cvxopt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
