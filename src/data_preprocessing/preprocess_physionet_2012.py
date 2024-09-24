#!/usr/bin/env python3

from tqdm import tqdm
import os
import pandas as pd
import pickle
import numpy as np

def inv_list(l, start=0):
    d = {}
    for i in range(len(l)):
        d[l[i]] = i+start
    return d

raw_data_path = '../../data/raw/physionet/physionet.org/files/challenge-2012/1.0.0/'
def read_dataset(d):
    ts = []
    pbar = tqdm(os.listdir(raw_data_path+'/set-'+d), desc='Reading time series set '+d)
    for f in pbar:
        data = pd.read_csv(raw_data_path+'/set-'+d+'/'+f).iloc[1:]
        data = data.loc[data.Parameter.notna()]
        if len(data)<=5:
            continue
        data = data.loc[data.Value>=0] # neg Value indicates missingness.
        data['RecordID'] = f[:-4]
        ts.append(data)
    ts = pd.concat(ts)
    return ts

ts = pd.concat((read_dataset('a'), read_dataset('b'), read_dataset('c')))
ts.Time = ts.Time.apply(lambda x:int(x[:2])+int(x[3:])/60) # No. of hours since admission.
ts.rename(columns={'Time':'hour', 'Parameter':'variable', 'Value':'value'}, inplace=True)
oc_a = pd.read_csv(raw_data_path+'/Outcomes-a.txt', usecols=['RecordID', 'Length_of_stay', 'In-hospital_death'])
oc_a['subset'] = 'a'
oc_b = pd.read_csv(raw_data_path+'/Outcomes-b.txt', usecols=['RecordID', 'Length_of_stay', 'In-hospital_death'])
oc_b['subset'] = 'b'
oc_c = pd.read_csv(raw_data_path+'/Outcomes-c.txt', usecols=['RecordID', 'Length_of_stay', 'In-hospital_death'])
oc_c['subset'] = 'c'
oc = pd.concat((oc_a,oc_b,oc_c))
oc.RecordID = oc.RecordID.astype(str)
oc.rename(columns={'Length_of_stay':'length_of_stay', 'In-hospital_death':'in_hospital_mortality'}, inplace=True)
rec_ids = sorted(list(ts.RecordID.unique()))
rid_to_ind = inv_list(rec_ids)
oc = oc.loc[oc.RecordID.isin(rec_ids)]
ts['ts_ind'] = ts.RecordID.map(rid_to_ind)
oc['ts_ind'] = oc.RecordID.map(rid_to_ind)
ts.drop(columns='RecordID', inplace=True)
oc.drop(columns='RecordID', inplace=True)

# Drop duplicates.
ts = ts.drop_duplicates()

# Convert categorical to numeric.
ii = (ts.variable=='ICUType')
for val in [4,3,2,1]:
    kk = ii&(ts.value==val)
    ts.loc[kk, 'variable'] = 'ICUType_'+str(val)
ts.loc[ii, 'value'] = 1
    
# Normalize data except Age, Gender, Height, ICUType.
means_stds = ts.groupby('variable').agg({'value':['mean', 'std']})
means_stds.columns = [col[1] for col in means_stds.columns]
means_stds.loc[means_stds['std']==0, 'std'] = 1
ts = ts.merge(means_stds.reset_index(), on='variable', how='left')
ii = ts.variable.apply(lambda x:not(x.startswith('ICUType')))&(~ts.variable.isin(['Age', 'Gender', 'Height']))
ts.loc[ii, 'value'] = (ts.loc[ii, 'value']-ts.loc[ii, 'mean'])/ts.loc[ii, 'std']

# Generate split.
train_valid_ind = np.array(oc.loc[oc.subset!='a'].ts_ind)
np.random.seed(123)
np.random.shuffle(train_valid_ind)
bp = int(0.8*len(train_valid_ind))
train_ind = train_valid_ind[:bp]
valid_ind = train_valid_ind[bp:]
test_ind = np.array(oc.loc[oc.subset=='a'].ts_ind)
oc.drop(columns='subset', inplace=True)

# Store data.
pickle.dump([ts, oc, train_ind, valid_ind, test_ind], open('../../data/pre_processed/physionet_2012_preprocessed.pkl','wb'))
