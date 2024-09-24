import hydra
from omegaconf import DictConfig
import omegaconf
import wandb
from wandb.keras import WandbMetricsLogger

from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Lambda, Concatenate
from tensorflow.keras.models import Model
import pickle
import numpy as np
from tqdm import tqdm
tqdm.pandas()
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback, EarlyStopping
import pandas as pd
import math
import tensorflow.keras as keras
import os
import tensorflow as tf

from src.model import build_strats

def load_and_generate_train_val_test_sets(data_path):

    """
    Adapted from https://arxiv.org/abs/2107.14293

    Load data and generate training, validation, and test sets.

    This function reads a dataset from the specified path, processes the data to separate 
    features and labels, and splits it into training, validation, and test sets.
    """
    
    data, oc, train_ind, valid_ind, test_ind = pickle.load(open(data_path, 'rb'))

    y = np.array(oc.sort_values(by='ts_ind')['in_hospital_mortality']).astype('float32')
    N = data.ts_ind.max() + 1
    # Correct test_ind.
    test_ind = np.setdiff1d(oc.ts_ind.unique(), np.concatenate((train_ind, valid_ind), axis=-1))
    # Get static data with mean fill and missingness indicator.
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

    N = data.ts_ind.max() + 1
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
    # Add obs index.
    data = data.sort_values(by=['ts_ind']).reset_index(drop=True)
    data = data.reset_index().rename(columns={'index':'obs_ind'})
    data = data.merge(data.groupby('ts_ind').agg({'obs_ind':'min'}).reset_index().rename(columns={ \
                                                                'obs_ind':'first_obs_ind'}), on='ts_ind')
    data['obs_ind'] = data['obs_ind'] - data['first_obs_ind']

    max_len = int(data.groupby('ts_ind')["obs_ind"].max().quantile(0.99) + 1)
    data = data.loc[data.obs_ind<max_len]
    print ('max_len', max_len)
    # Generate times_ip and values_ip matrices.
    times_inp = np.zeros((N, max_len), dtype='float32')
    values_inp = np.zeros((N, max_len), dtype='float32')
    varis_inp = np.zeros((N, max_len), dtype='int32')
    for row in tqdm(data.itertuples()):
        ts_ind = row.ts_ind
        l = row.obs_ind
        times_inp[ts_ind, l] = row.hour
        values_inp[ts_ind, l] = row.value
        varis_inp[ts_ind, l] = row.vind


    train_ip = [ip[train_ind] for ip in [times_inp, values_inp, varis_inp]]
    valid_ip = [ip[valid_ind] for ip in [times_inp, values_inp, varis_inp]]

    test_ip = [ip[test_ind] for ip in [times_inp, values_inp, varis_inp]]

    del times_inp, values_inp, varis_inp
    train_op = y[train_ind]
    valid_op = y[valid_ind]
    test_op = y[test_ind]
    del y

    return train_ip, valid_ip, test_ip, train_op, valid_op, test_op

def load_model_architecture_and_weights(weights_path, d, N, he, dropout, V, max_seq_length):
    """
    Load a model architecture and its pre-trained weights.

    This function builds a model with the specified architecture parameters and loads 
    pre-trained weights from the given file path.
    """

    model = build_strats(max_seq_length, V, d, N, he, dropout)
    model.load_weights(weights_path)

    return model
    

def extend_model(base_model, V, max_seq_length):
    """
    Extend a pre-trained model with a dense layer for binary classification.
    """
    
    input_layers = [Input(shape=(max_seq_length,)), Input(shape=(max_seq_length,)), Input(shape=(max_seq_length,))]

    pt_output = base_model(input_layers)

    output_layer = Dense(1, activation='sigmoid')(pt_output)

    extended_model = Model(inputs=input_layers, outputs=output_layer)

    return extended_model


def get_res(y_true, y_pred):
    """
    Compute ROC AUC, PR AUC, and min(precision, recall) for a given set of true and predicted labels.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)
    minrp = np.minimum(precision, recall).max()
    roc_auc = roc_auc_score(y_true, y_pred)
    return [roc_auc, pr_auc, minrp]



def mortality_loss(y_true, y_pred):
    """
    Compute a weighted binary cross-entropy loss for imbalanced mortality prediction.
    """
    class_weights = compute_class_weight(class_weight='balanced', classes=[0,1], y=train_op)
    sample_weights = (1-y_true)*class_weights[0] + y_true*class_weights[1]
    bce = K.binary_crossentropy(y_true, y_pred)
    return K.mean(sample_weights*bce, axis=-1)


def get_min_loss(weight):
    def min_loss(y_true, y_pred):
        return weight*y_pred
    return min_loss

class CustomCallback(Callback):
    def __init__(self, validation_data, batch_size):
        self.val_x, self.val_y = validation_data
        self.batch_size = batch_size
        super(Callback, self).__init__()

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.val_x, verbose=0, batch_size=self.batch_size)
        if type(y_pred)==type([]):
            y_pred = y_pred[0]
        precision, recall, thresholds = precision_recall_curve(self.val_y, y_pred)
        pr_auc = auc(recall, precision)
        roc_auc = roc_auc_score(self.val_y, y_pred)
        logs['custom_metric'] = pr_auc + roc_auc
        print ('val_aucs:', pr_auc, roc_auc)


def prepare_data(indices, input_data, output_data):
    """
    Prepare and filter data based on specified indices.
    """
    return [data[indices] for data in input_data], output_data[indices]

def train_and_evaluate_model(model, train_input, train_output, valid_input, valid_output, batch_size, learning_rate, epochs, patience, weight_decay):

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        learning_rate,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True)

    model.compile(loss=mortality_loss, optimizer=Adam(learning_rate=lr_schedule, weight_decay=weight_decay))
    es = EarlyStopping(monitor='custom_metric', patience=patience, mode='max', restore_best_weights=True)
    cus = CustomCallback(validation_data=(valid_input, valid_output), batch_size=batch_size)
    history = model.fit(train_input, train_output, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[cus, es]).history
    
    return history



def log_results(test_output, test_input, model, batch_size):
    rocauc, prauc, minrp = get_res(test_output, model.predict(test_input, verbose=0, batch_size=batch_size))
    print('Test results:', rocauc, prauc, minrp)
    return rocauc, prauc, minrp

@hydra.main(config_path="../../config", config_name="ft_config_physionet")
def main(cfg: DictConfig):

    # Load config parameters
    data_path = hydra.utils.to_absolute_path(cfg.data_path)
    pt_model_weights_path = hydra.utils.to_absolute_path(cfg.pt_model_weights_path)
    lds = cfg.lds
    repeats = cfg.repeats
    max_seq_length = cfg.max_seq_length
    V = cfg.V
    d = cfg.d
    N = cfg.N
    he = cfg.he
    dropout = cfg.dropout
    batch_size = cfg.batch_size
    learning_rate = cfg.learning_rate
    epochs = cfg.epochs
    patience = cfg.patience
    weight_decay = cfg.weight_decay
    file_name = cfg.file_name
    results_path = hydra.utils.to_absolute_path(cfg.results_path)

    
    print(file_name)

    config_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True)
    
    global train_op
    train_ip, valid_ip, test_ip, train_op, valid_op, test_op = load_and_generate_train_val_test_sets(data_path)

    train_inds = np.arange(len(train_op))
    valid_inds = np.arange(len(valid_op))
    gen_res = {}

    np.random.seed(2021)
    for ld in lds:

        np.random.shuffle(train_inds)
        np.random.shuffle(valid_inds)

        train_starts = [int(i) for i in np.linspace(0, len(train_inds) - int(ld * len(train_inds) / 100), repeats)]
        valid_starts = [int(i) for i in np.linspace(0, len(valid_inds) - int(ld * len(valid_inds) / 100), repeats)]

        all_test_res = []

        for i in range(repeats):
            print(f'Repeat {i}, ld {ld}')

            curr_train_ind = train_inds[train_starts[i]:train_starts[i] + int(ld * len(train_inds) / 100)]
            curr_valid_ind = valid_inds[valid_starts[i]:valid_starts[i] + int(ld * len(valid_inds) / 100)]

            curr_train_ip, curr_train_op = prepare_data(curr_train_ind, train_ip, train_op)
            curr_valid_ip, curr_valid_op = prepare_data(curr_valid_ind, valid_ip, valid_op)

            base_model = load_model_architecture_and_weights(pt_model_weights_path, d=d, N=N, he=he, dropout=dropout, V=V, max_seq_length=max_seq_length)
                                                                
            extended_model = extend_model(base_model, V, max_seq_length)

            history = train_and_evaluate_model(extended_model, curr_train_ip, curr_train_op, curr_valid_ip, curr_valid_op, batch_size, learning_rate, epochs, patience, weight_decay)
            test_results = log_results(test_op, test_ip, extended_model, batch_size)
            all_test_res.append(test_results)

        gen_res[ld] = [(np.mean([res[i] for res in all_test_res]), np.std([res[i] for res in all_test_res])) for i in range(len(all_test_res[0]))]

    with open(results_path, 'wb') as file:
        pickle.dump(gen_res, file)
        print("Results were saved successfully!")

if __name__ == "__main__":
    main()
    

    