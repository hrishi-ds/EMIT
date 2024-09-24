# Adapted from https://arxiv.org/abs/2107.14293

import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

pred_window = 2 # hours
obs_windows = [12,16,20,24,28,32,36,40,44]

# Read data.
data_path = '../../data/pre_processed/physionet_2012_preprocessed.pkl'
data, oc, train_ind, valid_ind, test_ind = pickle.load(open(data_path, 'rb'))
# Correct test_ind.
test_ind = np.setdiff1d(oc.ts_ind.unique(), np.concatenate((train_ind, valid_ind), axis=-1))
# Get static data with mean fill and missingness indicator.
N = data.ts_ind.max() + 1
static_varis = ['Age', 'Gender', 'Height', 'ICUType_1', 'ICUType_2', 'ICUType_3', 'ICUType_4']
ii = data.variable.isin(static_varis)
static_data = data.loc[ii]
data = data.loc[~ii]
def inv_list(l, start=0):
    d = {}
    for i in range(len(l)):
        d[l[i]] = i+start
    return d
static_var_to_ind = inv_list(static_varis)
D = len(static_varis) + 2
demo = np.zeros((N, D))
for row in tqdm(static_data.itertuples()):
    demo[row.ts_ind, static_var_to_ind[row.variable]] = row.value
    if row.variable=='Gender':
        demo[row.ts_ind, D-2] = 1
    elif row.variable=='Height':
        demo[row.ts_ind, D-1] = 1
# Normalize static data.
means = demo.mean(axis=0, keepdims=True)
stds = demo.std(axis=0, keepdims=True)
stds = (stds==0)*1 + (stds!=0)*stds
demo = (demo-means)/stds
# Get variable indices.
varis = sorted(list(set(data.variable)))
V = len(varis)
def inv_list(l, start=0):
    d = {}
    for i in range(len(l)):
        d[l[i]] = i+start
    return d
var_to_ind = inv_list(varis, start=1)
data['vind'] = data.variable.map(var_to_ind)
data = data[['ts_ind', 'vind', 'hour', 'value']].sort_values(by=['ts_ind', 'vind', 'hour'])
# Find max_len.
# fore_max_len = data.loc[data.hour<np.max(obs_windows)].groupby('ts_ind').size().max()
fore_max_len = int(data.groupby('ts_ind').size().quantile(0.99))
print ('fore_max_len', fore_max_len)
# Get forecast inputs and outputs.
data = data.loc[~data.ts_ind.isin(test_ind)]
fore_times_ip = []
fore_values_ip = []
fore_varis_ip = []
fore_op = []
fore_inds = []
def f(x):
    mask = [0 for i in range(V)]
    values = [0 for i in range(V)]
    for vv in x:
        v = int(vv[0])-1
        mask[v] = 1
        values[v] = vv[1]
    return values+mask
def pad(x):
    return x+[0]*(fore_max_len-len(x))
for w in tqdm(obs_windows):
    pred_data = data.loc[(data.hour>=w)&(data.hour<=w+pred_window)]
    pred_data = pred_data.groupby(['ts_ind', 'vind']).agg({'value':'first'}).reset_index()
    pred_data['vind_value'] = pred_data[['vind', 'value']].values.tolist()
    pred_data = pred_data.groupby('ts_ind').agg({'vind_value':list}).reset_index()
    pred_data['vind_value'] = pred_data['vind_value'].apply(f)    
    obs_data = data.loc[data.hour<w]
    obs_data = obs_data.loc[obs_data.ts_ind.isin(pred_data.ts_ind)]
    obs_data = obs_data.groupby('ts_ind').head(fore_max_len)

    obs_data = obs_data.groupby('ts_ind').agg({'vind':list, 'hour':list, 'value':list}).reset_index()
    obs_data = obs_data.merge(pred_data, on='ts_ind')
    for col in ['vind', 'hour', 'value']:
        obs_data[col] = obs_data[col].apply(pad)
    fore_op.append(np.array(list(obs_data.vind_value)))
    fore_inds.append(np.array(list(obs_data.ts_ind)))
    fore_times_ip.append(np.array(list(obs_data.hour)))
    fore_values_ip.append(np.array(list(obs_data.value)))
    fore_varis_ip.append(np.array(list(obs_data.vind)))
del data
fore_times_ip = np.concatenate(fore_times_ip, axis=0)
fore_values_ip = np.concatenate(fore_values_ip, axis=0)
fore_varis_ip = np.concatenate(fore_varis_ip, axis=0)
fore_op = np.concatenate(fore_op, axis=0)
fore_inds = np.concatenate(fore_inds, axis=0)
fore_demo = demo[fore_inds]


# Generate 3 sets of inputs and outputs.
train_ind = np.argwhere(np.in1d(fore_inds, train_ind)).flatten()
valid_ind = np.argwhere(np.in1d(fore_inds, valid_ind)).flatten()
fore_train_ip = [ip[train_ind] for ip in [fore_times_ip, fore_values_ip, fore_varis_ip]]
fore_valid_ip = [ip[valid_ind] for ip in [fore_times_ip, fore_values_ip, fore_varis_ip]]

del fore_times_ip, fore_values_ip, fore_varis_ip, demo, fore_demo
fore_train_op = fore_op[train_ind]
fore_valid_op = fore_op[valid_ind]
del fore_op


np.savez(f'../../data/pre_training/pretraining_data_physionet_{fore_max_len}.npz', 
fore_train_ip=fore_train_ip, 
fore_train_op=fore_train_op, 
fore_valid_ip=fore_valid_ip, 
fore_valid_op=fore_valid_op)