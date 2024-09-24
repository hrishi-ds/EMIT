import hydra
from omegaconf import DictConfig
import omegaconf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense

import tensorflow as tf
import numpy as np
import pandas as pd
import datetime
import logging

from src.model import *

def load_data(data_path):
    """
    Function to load data from the given path
    """
    data = np.load(data_path)

    fore_train_ip = data['fore_train_ip']
    fore_train_op = data['fore_train_op']
    fore_valid_ip = data['fore_valid_ip']
    fore_valid_op = data['fore_valid_op']

    return fore_train_ip, fore_train_op, fore_valid_ip, fore_valid_op

def load_mask(mask_path):
    """
    Function to load mask from the given path
    """
    mask = np.load(mask_path)


    mask_train = mask['fore_train_masks']
    mask_val = mask['fore_val_masks']

    return mask_train, mask_val

def get_batches(data_path, mask_path, batch_size):
    """
    Function to get batches of data
    """
    fore_train_ip, fore_train_op, fore_valid_ip, fore_valid_op = load_data(data_path)
    mask_train, mask_val = load_mask(mask_path)

    valid_dataset_batched = tf.data.Dataset.from_tensor_slices((fore_valid_ip[0], fore_valid_ip[1], fore_valid_ip[2], fore_valid_op, mask_val)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return fore_train_ip, fore_train_op, valid_dataset_batched, mask_train
    
def create_randomly_masked_embeddings(times_emb, values_emb, variables_emb, mask, times_mask_token, values_mask_token, variables_mask_token):
    """
    Randomly mask embeddings for times, values, and variables based on a given mask.

    This function randomly selects which embedding (times, values, or variables) to mask 
    and replaces the corresponding values with mask tokens where the mask is True.
    """

    # Shape of the embeddings
    shape = tf.shape(times_emb)
    batch_size, seq_len, _ = shape[0], shape[1], shape[2]

    mask = tf.cast(mask, tf.bool)
    
    # Expand the mask token shapes to match the embeddings' shape for broadcasting
    times_mask_token = tf.reshape(times_mask_token, [1, 1, -1])
    values_mask_token = tf.reshape(values_mask_token, [1, 1, -1])
    variables_mask_token = tf.reshape(variables_mask_token, [1, 1, -1])
    
    # Create a random tensor of the same shape as the mask, with values in [0, 3),
    # indicating which embedding to mask: 0 for times, 1 for values, 2 for variables
    choices = tf.random.uniform(shape=tf.shape(mask[:,:,0]), minval=0, maxval=3, dtype=tf.int32)
    
    # Create boolean masks for each choice
    times_choice = tf.equal(choices, 0)
    values_choice = tf.equal(choices, 1)
    variables_choice = tf.equal(choices, 2)
    
    # Apply the mask token based on the choice, only where mask is True
    masked_times_emb = tf.where(tf.expand_dims(times_choice, -1) & mask, times_mask_token, times_emb)
    masked_values_emb = tf.where(tf.expand_dims(values_choice, -1) & mask, values_mask_token, values_emb)
    masked_variables_emb = tf.where(tf.expand_dims(variables_choice, -1) & mask, variables_mask_token, variables_emb)
    
    return masked_times_emb, masked_values_emb, masked_variables_emb


def forecast_loss(y_true, y_pred, V):
    y_true = tf.cast(y_true, dtype=y_pred.dtype)
    return K.sum(y_true[:,V:]*(y_true[:,:V]-y_pred)**2, axis=-1)

@hydra.main(config_path="../../config", config_name="pt_config_mimic")
def main(cfg: DictConfig):

    # parse config
    data_path = hydra.utils.to_absolute_path(cfg.data_path)
    mask_path = hydra.utils.to_absolute_path(cfg.mask_path)
    batch_size = cfg.batch_size
    learning_rate = cfg.learning_rate
    epochs = cfg.epochs
    patience = cfg.patience
    max_len = cfg.max_len
    V = cfg.V
    d = cfg.d
    N = cfg.N
    he = cfg.he
    dropout = cfg.dropout
    weight_decay = cfg.weight_decay
    error_coefficient = cfg.error_coefficient
    samples_per_epoch = cfg.samples_per_epoch
    mask_threshold = cfg.mask_threshold
    insignificant_prob = cfg.insignificant_prob
    model_name = hydra.utils.to_absolute_path(cfg.model_name)

    config_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True)

    # get the data
    fore_train_ip, fore_train_op, valid_dataset_batched, event_mask_train = get_batches(data_path, mask_path, batch_size)

    # create the model
    pretraining_model = build_strats(max_len, V, d, N, he, dropout)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        learning_rate,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, weight_decay=weight_decay)

    # define mask token(s)
    times_mask_token = tf.Variable(initial_value=tf.keras.initializers.GlorotUniform()(shape=(1,d)), trainable=True)
    values_mask_token = tf.Variable(initial_value=tf.keras.initializers.GlorotUniform()(shape=(1,d)), trainable=True)
    variables_mask_token = tf.Variable(initial_value=tf.keras.initializers.GlorotUniform()(shape=(1,d)), trainable=True)

    # define best loss
    best_loss = float('inf')
    train_epoch_loss, val_epoch_loss = [], []

    # start training
    for epoch in range(epochs):

        train_running_loss, val_running_loss = [], []
        
        e_indices = np.random.choice(range(len(fore_train_op)), size=samples_per_epoch, replace=False)
        for ix, start in enumerate(range(0, len(e_indices), batch_size)):
            ind = e_indices[start:start+batch_size]
            batch = [ip[ind] for ip in fore_train_ip]
            

            with tf.GradientTape() as tape:

                    times, values, varis = batch
                    event_mask = np.array([event_mask_train[i] for i in ind])
                    event_mask = tf.convert_to_tensor(event_mask, dtype=tf.float32)
                    event_mask = tf.tile(tf.expand_dims(event_mask, -1), [1, 1, d])

                    varis_emb = pretraining_model.layers[3](varis)
                    values_emb = pretraining_model.layers[4](values)
                    times_emb = pretraining_model.layers[5](times)

                    masked_times_emb, masked_values_emb, masked_variables_emb = create_randomly_masked_embeddings(times_emb, values_emb, varis_emb, event_mask, times_mask_token, values_mask_token, variables_mask_token)

                    comb_emb_masked = masked_variables_emb + masked_values_emb + masked_times_emb

                    comb_emb_unmasked = varis_emb + values_emb + times_emb

                    input_mask = pretraining_model.layers[7](varis)

                    # masking error
                    logits_for_masking = pretraining_model.layers[8](x=comb_emb_masked, mask=tf.cast(input_mask, tf.float32))
                    masking_error = (logits_for_masking[tf.cast(event_mask, tf.bool)] - comb_emb_unmasked[tf.cast(event_mask, tf.bool)])

                     # forecasting forward pass
                    attention_weights = pretraining_model.layers[9](logits_for_masking, mask=tf.cast(input_mask, tf.float32))
                    fused_embeddings = pretraining_model.layers[10]([logits_for_masking, attention_weights])
                    logits_for_forecasting = pretraining_model.layers[11](fused_embeddings)
                    forecasting_error = forecast_loss(fore_train_op[ind], logits_for_forecasting, V)

                    # total loss calculation
                    total_loss =  tf.reduce_mean(forecasting_error) + error_coefficient*tf.reduce_mean(tf.square(masking_error))
                    train_running_loss.append(total_loss)

                    logging.info(f"Epoch {epoch + 1}/{epochs} Batch {ix+1}/{len(e_indices)//batch_size} Loss {total_loss}")


            gradients = tape.gradient(total_loss, pretraining_model.trainable_variables + [times_mask_token, values_mask_token, variables_mask_token])
            optimizer.apply_gradients(zip(gradients, pretraining_model.trainable_variables + [times_mask_token, values_mask_token, variables_mask_token]))



        train_epoch_loss.append(np.mean(train_running_loss))


        # validation
        for val_batch in valid_dataset_batched:
            times, values, varis, op, mask = val_batch
            mask = tf.tile(tf.expand_dims(mask, -1), [1, 1, d])

            varis_emb = pretraining_model.layers[3](varis)
            values_emb = pretraining_model.layers[4](values)
            times_emb = pretraining_model.layers[5](times)

            masked_times_emb, masked_values_emb, masked_variables_emb = create_randomly_masked_embeddings(times_emb, values_emb, varis_emb, mask, times_mask_token, values_mask_token, variables_mask_token)

            comb_emb_masked = masked_variables_emb + masked_values_emb + masked_times_emb

            comb_emb_unmasked = varis_emb + values_emb + times_emb

            input_mask = pretraining_model.layers[7](varis)

            # masking error
            logits_for_masking = pretraining_model.layers[8](x=comb_emb_masked, mask=tf.cast(input_mask, tf.float32))
            masking_error = (logits_for_masking[tf.cast(mask, tf.bool)] - comb_emb_unmasked[tf.cast(mask, tf.bool)])

            # forecasting forward pass
            attention_weights = pretraining_model.layers[9](logits_for_masking, mask=tf.cast(input_mask, tf.float32))
            fused_embeddings = pretraining_model.layers[10]([logits_for_masking, attention_weights])
            logits_for_forecasting = pretraining_model.layers[11](fused_embeddings)
            forecasting_error = forecast_loss(op, logits_for_forecasting, V)

            # total loss calculation
            total_loss =  tf.reduce_mean(forecasting_error) + error_coefficient*tf.reduce_mean(tf.square(masking_error))
            val_running_loss.append(total_loss)

        # get average loss for epoch
        val_epoch_loss.append(np.mean(val_running_loss))


        # Early stopping and model saving logic based on validation loss
        if val_epoch_loss[-1] < best_loss:
            best_loss = val_epoch_loss[-1]
            patience_counter = 0
            model_path = model_name + ".h5"
            pretraining_model.save_weights(model_path)
            
            logging.info(f"Saved model to {model_name}.h5")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logging.info("Early stopping triggered")
            break


if __name__ == "__main__":
    main()

    