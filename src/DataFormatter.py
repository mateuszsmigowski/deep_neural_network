import numpy as np


class DataFormatter:

    def min_max_scale(self, data):
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        range_val = max_val - min_val
        range_val[range_val == 0] = 1e-8
        
        scaled_data = (data - min_val) / range_val
        return scaled_data, min_val, max_val

    def inverse_min_max_scale(self, scaled_data, min_val, max_val):
        if hasattr(min_val, "__len__") and len(min_val) > 1:
            return scaled_data * (max_val[0] - min_val[0]) + min_val[0]
        else:
            return scaled_data * (max_val - min_val) + min_val

    def create_sequences_with_local_norm(self, data, seq_length, target_col=0):
        X = []
        Y = []
        scales = []

        for i in range(len(data) - seq_length):
            x_seq = data[i : i + seq_length].copy()
            y_val = data[i + seq_length, target_col]
            
            seq_mean = x_seq[:, target_col].mean()
            seq_std = x_seq[:, target_col].std()
            if seq_std < 1e-8:
                seq_std = 1e-8
            
            x_norm = (x_seq - seq_mean) / seq_std
            y_norm = (y_val - seq_mean) / seq_std
            
            X.append(x_norm)
            Y.append(y_norm)
            scales.append((seq_mean, seq_std))

        return np.array(X), np.array(Y), scales

    def inverse_local_norm(self, scaled_val, mean, std):
        return scaled_val * std + mean

    def split_data(self, X, Y, scales=None):

        num_samples = len(X)
        train_size = int(0.7 * num_samples)
        val_size = int(0.15 * num_samples)
        test_size = num_samples - train_size - val_size

        X_train, Y_train = X[:train_size], Y[:train_size]
        X_val, Y_val = (
            X[train_size : train_size + val_size],
            Y[train_size : train_size + val_size],
        )
        X_test, Y_test = X[train_size + val_size :], Y[train_size + val_size :]

        if scales is not None:
            scales_train = scales[:train_size]
            scales_val = scales[train_size : train_size + val_size]
            scales_test = scales[train_size + val_size :]
            return X_train, Y_train, X_val, Y_val, X_test, Y_test, scales_train, scales_val, scales_test

        return X_train, Y_train, X_val, Y_val, X_test, Y_test
