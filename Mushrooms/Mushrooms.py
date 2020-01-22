#A Kaggle project to determine whether a mushroom is edible or poisonous based on its physical features. All variables are categorical.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
data = pd.read_csv('mushrooms.csv')
print 'Size of data: {}'.format(data.shape)
print 'Number of events: {}'.format(data.shape[0])
print 'Number of columns: {}'.format(data.shape[1])

print '\nList of features in dataset:'
for col in data.columns:
    print col

# look at column labels --- notice first one is "class"


print 'Number of signal events: {}'.format(len(data[data.class == 'e']))
print 'Number of background events: {}'.format(len(data[data.class == 'p']))
print 'Fraction signal: {}'.format(len(data[data.class == 'e'])/(float)(len(data[data.class == 'p']) + len(data[data.class == 'e'])))

#data['class'] = data.class.astype('category')

data_train = data[:8000]
data_test = data[8000:]

print 'Number of training samples: {}'.format(len(data_train))
print 'Number of testing samples: {}'.format(len(data_test))

print '\nNumber of signal events in training set: {}'.format(len(data_train[data_train.class == 's']))
print 'Number of background events in training set: {}'.format(len(data_train[data_train.class == 'b']))
print 'Fraction signal: {}'.format(len(data_train[data_train.class == 's'])/(float)(len(data_train[data_train.class == 's']) + len(data_train[data_train.class == 'b'])))

feature_names = data.columns[1:]

