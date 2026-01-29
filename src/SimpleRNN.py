import numpy as np


class SimpleRNN:

    def __init__(self, input_size, hidden_size, output_size):

        self.hidden_size = hidden_size

        self.W_xh = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_hy = np.random.randn(output_size, hidden_size) * 0.01

        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))

        self.h = np.zeros((hidden_size, 1))
        self.cache = None

    def forward(self, inputs):

        outputs = []
        cache = {"h": {-1: self.h}, "inputs": inputs}

        for t, x in enumerate(inputs):
            x = np.array(x).reshape(-1, 1)

            cache["h"][t] = np.tanh(np.dot(self.W_xh, x) + np.dot(self.W_hh, cache["h"][t - 1]) + self.b_h)

            y = np.dot(self.W_hy, cache["h"][t]) + self.b_y
            outputs.append(y)

        return outputs, cache

    def backward(self, outputs, targets, cache, learning_rate=0.01, loss_only_last=False):

        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)

        dh_next = np.zeros((self.hidden_size, 1))

        inputs = cache["inputs"]
        h_states = cache["h"]

        loss = 0

        last_t = len(inputs) - 1
        for t in reversed(range(len(inputs))):
            if loss_only_last and t != last_t:
                dy = np.zeros_like(outputs[t])
            else:
                dy = outputs[t] - targets[t]
                loss += np.sum(dy**2) / 2

            dW_hy += np.dot(dy, h_states[t].T)
            db_y += dy

            dh = np.dot(self.W_hy.T, dy) + dh_next
            dh_raw = (1 - h_states[t] * h_states[t]) * dh

            db_h += dh_raw
            dW_xh += np.dot(dh_raw, np.array(inputs[t]).reshape(-1, 1).T)
            dW_hh += np.dot(dh_raw, h_states[t - 1].T)

            dh_next = np.dot(self.W_hh.T, dh_raw)

        for dparam in [dW_xh, dW_hh, dW_hy, db_h, db_y]:
            np.clip(dparam, -5, 5, out=dparam)

        self.W_xh -= learning_rate * dW_xh
        self.W_hh -= learning_rate * dW_hh
        self.W_hy -= learning_rate * dW_hy
        self.b_h -= learning_rate * db_h
        self.b_y -= learning_rate * db_y

        return loss
