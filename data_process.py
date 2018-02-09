# -*- coding: utf-8 -*-
"""
Created on Wed Apr 05 16:01:42 2017

@author: mw
"""

import numpy as np
import random
import pandas as pd
import mwlib as mb
import copy

data_load_LL = np.loadtxt('./temp_data/data/LL_data.txt', skiprows=0)
data_load_LR = np.loadtxt('./temp_data/data/LR_data.txt', skiprows=0)
data_load_RL = np.loadtxt('./temp_data/data/RL_data.txt', skiprows=0)


                
test_loc = random.sample(range(8575),575)

#data split for LL
test_data_LL = data_load_LL[test_loc, :]
train_data_LL = np.delete(data_load_LL, test_loc, 0) #delete test data from training data
#data split for LR
test_data_LR = data_load_LR[test_loc, :]
train_data_LR = np.delete(data_load_LR, test_loc, 0) #delete test data from training data
#data split for RL
test_data_RL = data_load_RL[test_loc, :]
train_data_RL = np.delete(data_load_RL, test_loc, 0) #delete test data from training data

#data output for LL
train_data_out_LL = pd.DataFrame(train_data_LL, columns = ['top_length', 'bottom_length', 'top_spacer', 'bottom_spacer', 'angle'] + ['data' + str(i) for i in range(201)])
train_data_out_LL.to_csv('./temp_data/data/train_data_LL.csv')
test_data_out_LL = pd.DataFrame(test_data_LL, columns = ['top_length', 'bottom_length', 'top_spacer', 'bottom_spacer', 'angle'] + ['data' + str(i) for i in range(201)])
test_data_out_LL.to_csv('./temp_data/data/test_data_LL.csv')
#data output for LR
train_data_out_LR = pd.DataFrame(train_data_LR, columns = ['top_length', 'bottom_length', 'top_spacer', 'bottom_spacer', 'angle'] + ['data' + str(i) for i in range(201)])
train_data_out_LR.to_csv('./temp_data/data/train_data_LR.csv')
test_data_out_LR = pd.DataFrame(test_data_LR, columns = ['top_length', 'bottom_length', 'top_spacer', 'bottom_spacer', 'angle'] + ['data' + str(i) for i in range(201)])
test_data_out_LR.to_csv('./temp_data/data/test_data_LR.csv')
#data output for RL
train_data_out_RL = pd.DataFrame(train_data_RL, columns = ['top_length', 'bottom_length', 'top_spacer', 'bottom_spacer', 'angle'] + ['data' + str(i) for i in range(201)])
train_data_out_RL.to_csv('./temp_data/data/train_data_RL.csv')
test_data_out_RL = pd.DataFrame(test_data_RL, columns = ['top_length', 'bottom_length', 'top_spacer', 'bottom_spacer', 'angle'] + ['data' + str(i) for i in range(201)])
test_data_out_RL.to_csv('./temp_data/data/test_data_RL.csv')

#copy data for normalization
test_data_norm_LL = copy.deepcopy(test_data_LL)
train_data_norm_LL = copy.deepcopy(train_data_LL)

test_data_norm_LR = copy.deepcopy(test_data_LR)
train_data_norm_LR = copy.deepcopy(train_data_LR)

test_data_norm_RL = copy.deepcopy(test_data_RL)
train_data_norm_RL = copy.deepcopy(train_data_RL)

#create LL data after normalization
test_data_norm_LL[:,0] = mb.mwnorm(test_data_LL[:,0], label='top_length', reverse=False)
test_data_norm_LL[:,1] = mb.mwnorm(test_data_LL[:,1], label='bottom_length', reverse=False)
test_data_norm_LL[:,2] = mb.mwnorm(test_data_LL[:,2], label='top_spacer', reverse=False)
test_data_norm_LL[:,3] = mb.mwnorm(test_data_LL[:,3], label='bottom_spacer', reverse=False)
test_data_norm_LL[:,4] = mb.mwnorm(test_data_LL[:,4], label='angle', reverse=False)

train_data_norm_LL[:,0] = mb.mwnorm(train_data_LL[:,0], label='top_length', reverse=False)
train_data_norm_LL[:,1] = mb.mwnorm(train_data_LL[:,1], label='bottom_length', reverse=False)
train_data_norm_LL[:,2] = mb.mwnorm(train_data_LL[:,2], label='top_spacer', reverse=False)
train_data_norm_LL[:,3] = mb.mwnorm(train_data_LL[:,3], label='bottom_spacer', reverse=False)
train_data_norm_LL[:,4] = mb.mwnorm(train_data_LL[:,4], label='angle', reverse=False)

#data output for LL
train_data_norm_out_LL = pd.DataFrame(train_data_norm_LL, columns = ['top_length', 'bottom_length', 'top_spacer', 'bottom_spacer', 'angle'] + ['data' + str(i) for i in range(201)])
train_data_norm_out_LL.to_csv('./temp_data/data/train_data_norm_LL.csv')
test_data_norm_out_LL = pd.DataFrame(test_data_norm_LL, columns = ['top_length', 'bottom_length', 'top_spacer', 'bottom_spacer', 'angle'] + ['data' + str(i) for i in range(201)])
test_data_norm_out_LL.to_csv('./temp_data/data/test_data_norm_LL.csv')


#create LR data after normalization
test_data_norm_LR[:,0] = mb.mwnorm(test_data_LR[:,0], label='top_length', reverse=False)
test_data_norm_LR[:,1] = mb.mwnorm(test_data_LR[:,1], label='bottom_length', reverse=False)
test_data_norm_LR[:,2] = mb.mwnorm(test_data_LR[:,2], label='top_spacer', reverse=False)
test_data_norm_LR[:,3] = mb.mwnorm(test_data_LR[:,3], label='bottom_spacer', reverse=False)
test_data_norm_LR[:,4] = mb.mwnorm(test_data_LR[:,4], label='angle', reverse=False)

train_data_norm_LR[:,0] = mb.mwnorm(train_data_LR[:,0], label='top_length', reverse=False)
train_data_norm_LR[:,1] = mb.mwnorm(train_data_LR[:,1], label='bottom_length', reverse=False)
train_data_norm_LR[:,2] = mb.mwnorm(train_data_LR[:,2], label='top_spacer', reverse=False)
train_data_norm_LR[:,3] = mb.mwnorm(train_data_LR[:,3], label='bottom_spacer', reverse=False)
train_data_norm_LR[:,4] = mb.mwnorm(train_data_LR[:,4], label='angle', reverse=False)

#data output for LR
train_data_norm_out_LR = pd.DataFrame(train_data_norm_LR, columns = ['top_length', 'bottom_length', 'top_spacer', 'bottom_spacer', 'angle'] + ['data' + str(i) for i in range(201)])
train_data_norm_out_LR.to_csv('./temp_data/data/train_data_norm_LR.csv')
test_data_norm_out_LR = pd.DataFrame(test_data_norm_LR, columns = ['top_length', 'bottom_length', 'top_spacer', 'bottom_spacer', 'angle'] + ['data' + str(i) for i in range(201)])
test_data_norm_out_LR.to_csv('./temp_data/data/test_data_norm_LR.csv')


#create RL data after normalization
test_data_norm_RL[:,0] = mb.mwnorm(test_data_RL[:,0], label='top_length', reverse=False)
test_data_norm_RL[:,1] = mb.mwnorm(test_data_RL[:,1], label='bottom_length', reverse=False)
test_data_norm_RL[:,2] = mb.mwnorm(test_data_RL[:,2], label='top_spacer', reverse=False)
test_data_norm_RL[:,3] = mb.mwnorm(test_data_RL[:,3], label='bottom_spacer', reverse=False)
test_data_norm_RL[:,4] = mb.mwnorm(test_data_RL[:,4], label='angle', reverse=False)

train_data_norm_RL[:,0] = mb.mwnorm(train_data_RL[:,0], label='top_length', reverse=False)
train_data_norm_RL[:,1] = mb.mwnorm(train_data_RL[:,1], label='bottom_length', reverse=False)
train_data_norm_RL[:,2] = mb.mwnorm(train_data_RL[:,2], label='top_spacer', reverse=False)
train_data_norm_RL[:,3] = mb.mwnorm(train_data_RL[:,3], label='bottom_spacer', reverse=False)
train_data_norm_RL[:,4] = mb.mwnorm(train_data_RL[:,4], label='angle', reverse=False)

#data output for RL
train_data_norm_out_RL = pd.DataFrame(train_data_norm_RL, columns = ['top_length', 'bottom_length', 'top_spacer', 'bottom_spacer', 'angle'] + ['data' + str(i) for i in range(201)])
train_data_norm_out_RL.to_csv('./temp_data/data/train_data_norm_RL.csv')
test_data_norm_out_RL = pd.DataFrame(test_data_norm_RL, columns = ['top_length', 'bottom_length', 'top_spacer', 'bottom_spacer', 'angle'] + ['data' + str(i) for i in range(201)])
test_data_norm_out_RL.to_csv('./temp_data/data/test_data_norm_RL.csv')