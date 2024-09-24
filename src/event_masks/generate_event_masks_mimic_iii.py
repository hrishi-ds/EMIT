#!/usr/bin/env python3
import numpy as np
from numba import jit, prange
import numba
import fire


@numba.njit
def calculate_rate_of_change_numba(values, times):
    time_diffs = np.diff(times)
    value_diffs = np.diff(values)
    rates = np.zeros(len(values))  # Initialize with the same length as the input
    for j in range(len(time_diffs)):
        if time_diffs[j] != 0:
            rates[j] = value_diffs[j] / time_diffs[j]
        else:
            rates[j] = 0  # or np.nan if you prefer
    rates[-1] = 0  # or np.nan, for the last element
    return rates

@numba.njit
def batch_indices_to_mask_numba(times_array, values_array, variables_array, threshold):
    batch_masks = np.zeros(values_array.shape, dtype=np.bool_)
    
    for i in range(times_array.shape[0]):
        times = times_array[i]
        values = values_array[i]
        variables = variables_array[i]

        unique_vars = np.unique(variables)
        for var in unique_vars:
            var_indices = np.where(variables == var)[0]
            if len(var_indices) > 1:
                var_values = values[var_indices]
                var_times = times[var_indices]

                rate_of_change = calculate_rate_of_change_numba(var_values, var_times)
                significant_change = np.abs(rate_of_change) > threshold

                significant_indices = var_indices[1:][significant_change]
                batch_masks[i, significant_indices] = True

    return batch_masks

@numba.njit
def batch_indices_to_mask_numba_with_insignificant_masks(times_array, values_array, variables_array, threshold, insignificant_prob=0.0):
    batch_masks = np.zeros(values_array.shape, dtype=np.bool_)

    rate_of_change_list = []
    
    
    for i in prange(times_array.shape[0]):
        times = times_array[i]
        values = values_array[i]
        variables = variables_array[i]

        running_rate_of_change_list = []

        unique_vars = np.unique(variables)
        unique_vars = unique_vars[unique_vars != 0]
        for var in unique_vars:
        
            var_indices = np.where(variables == var)[0]
            if len(var_indices) > 1:
                var_values = values[var_indices]
                var_times = times[var_indices]

                rate_of_change = calculate_rate_of_change_numba(var_values, var_times)

                
                running_rate_of_change_list.extend(rate_of_change)

                significant_change = np.abs(rate_of_change) > threshold

                significant_indices = var_indices[:][significant_change]
                insignificant_indices = var_indices[:][~significant_change]

                # Mask significant changes with (1-insignificant_prob) probability
                for idx in significant_indices:
                    if np.random.rand() < 1 - insignificant_prob:
                        batch_masks[i, idx] = True
                
                # Mask insignificant changes with insignificant_prob probability
                for idx in insignificant_indices:
                    if np.random.rand() < insignificant_prob:
                        batch_masks[i, idx] = True

        rate_of_change_list.append(running_rate_of_change_list)

    return batch_masks

def generate_event_masks(threshold=0.001, insignificant_prob=0.4):
    data_path='../../data/pre_training/pretraining_data_mimic_iii_880.npz'
    data = np.load(data_path)

    fore_train_ip = data['fore_train_ip']
    fore_valid_ip = data['fore_valid_ip']

    fore_train_masks = batch_indices_to_mask_numba_with_insignificant_masks(
        fore_train_ip[0], fore_train_ip[1], fore_train_ip[2], threshold, insignificant_prob
    )
    fore_val_masks = batch_indices_to_mask_numba_with_insignificant_masks(
        fore_valid_ip[0], fore_valid_ip[1], fore_valid_ip[2], threshold, insignificant_prob
    )

    np.savez(f"../../data/pre_training/mimic_event_masks_threshold_{threshold}_insignificant_prob_{insignificant_prob}.npz",
             fore_train_masks=fore_train_masks, fore_val_masks=fore_val_masks)

    print(f"mimic_event_masks_threshold_{threshold}_insignificant_prob_{insignificant_prob}.npz saved")

if __name__ == "__main__":
    fire.Fire(generate_event_masks)


