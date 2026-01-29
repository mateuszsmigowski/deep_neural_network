import numpy as np


class LSTM:

    def __init__(self, input_size, hidden_size, output_size):

        self.hidden_size = hidden_size

        std_x = 1.0 / np.sqrt(input_size)
        std_h = 1.0 / np.sqrt(hidden_size)

        self.W_xf = np.random.randn(hidden_size, input_size) * std_x
        self.W_hf = np.random.randn(hidden_size, hidden_size) * std_h
        self.b_f = np.ones((hidden_size, 1))

        self.W_xi = np.random.randn(hidden_size, input_size) * std_x
        self.W_hi = np.random.randn(hidden_size, hidden_size) * std_h
        self.b_i = np.zeros((hidden_size, 1))

        self.W_xo = np.random.randn(hidden_size, input_size) * std_x
        self.W_ho = np.random.randn(hidden_size, hidden_size) * std_h
        self.b_o = np.zeros((hidden_size, 1))

        self.W_xc = np.random.randn(hidden_size, input_size) * std_x
        self.W_hc = np.random.randn(hidden_size, hidden_size) * std_h
        self.b_c = np.zeros((hidden_size, 1))

        self.W_hy = np.random.randn(output_size, hidden_size) * (2.0 / hidden_size)
        self.b_y = np.ones((output_size, 1)) * 0.5

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, inputs):

        outputs = []
        self.h = np.zeros((self.hidden_size, 1))
        self.c = np.zeros((self.hidden_size, 1))
        
        cache = {
            "h": {-1: self.h},
            "c": {-1: self.c},
            "f": {},
            "i": {},
            "o": {},
            "c_hat": {},
            "inputs": inputs
        }

        for t, x in enumerate(inputs):
            x = np.array(x).reshape(-1, 1)

            h_prev = cache["h"][t - 1]
            c_prev = cache["c"][t - 1]

            f = self.sigmoid(np.dot(self.W_xf, x) + np.dot(self.W_hf, h_prev) + self.b_f)
            i = self.sigmoid(np.dot(self.W_xi, x) + np.dot(self.W_hi, h_prev) + self.b_i)
            o = self.sigmoid(np.dot(self.W_xo, x) + np.dot(self.W_ho, h_prev) + self.b_o)
            
            c_hat = np.tanh(np.dot(self.W_xc, x) + np.dot(self.W_hc, h_prev) + self.b_c)

            c = f * c_prev + i * c_hat
            h = o * np.tanh(c)

            y = np.dot(self.W_hy, h) + self.b_y

            cache["h"][t] = h
            cache["c"][t] = c
            cache["f"][t] = f
            cache["i"][t] = i
            cache["o"][t] = o
            cache["c_hat"][t] = c_hat
            
            outputs.append(y)

        return outputs, cache

    def backward(self, outputs, targets, cache, learning_rate=0.01, loss_only_last=False):

        dW_xf, dW_hf, db_f = np.zeros_like(self.W_xf), np.zeros_like(self.W_hf), np.zeros_like(self.b_f)
        dW_xi, dW_hi, db_i = np.zeros_like(self.W_xi), np.zeros_like(self.W_hi), np.zeros_like(self.b_i)
        dW_xo, dW_ho, db_o = np.zeros_like(self.W_xo), np.zeros_like(self.W_ho), np.zeros_like(self.b_o)
        dW_xc, dW_hc, db_c = np.zeros_like(self.W_xc), np.zeros_like(self.W_hc), np.zeros_like(self.b_c)
        dW_hy, db_y = np.zeros_like(self.W_hy), np.zeros_like(self.b_y)

        dh_next = np.zeros((self.hidden_size, 1))
        dc_next = np.zeros((self.hidden_size, 1))

        inputs = cache["inputs"]
        loss = 0
        last_t = len(inputs) - 1

        for t in reversed(range(len(inputs))):
            if loss_only_last and t != last_t:
                dy = np.zeros_like(outputs[t])
            else:
                dy = outputs[t] - targets[t]
                loss += np.sum(dy**2) / 2

            dW_hy += np.dot(dy, cache["h"][t].T)
            db_y += dy

            dh = np.dot(self.W_hy.T, dy) + dh_next
            c, c_prev = cache["c"][t], cache["c"][t-1]
            f, i, o, c_hat = cache["f"][t], cache["i"][t], cache["o"][t], cache["c_hat"][t]
            tc = np.tanh(c)

            do = dh * tc
            do_raw = o * (1 - o) * do
            
            dW_xo += np.dot(do_raw, np.array(inputs[t]).reshape(-1, 1).T)
            dW_ho += np.dot(do_raw, cache["h"][t-1].T)
            db_o += do_raw

            dc = dc_next + dh * o * (1 - tc**2)
            
            dc_hat = dc * i
            dc_hat_raw = (1 - c_hat**2) * dc_hat
            
            dW_xc += np.dot(dc_hat_raw, np.array(inputs[t]).reshape(-1, 1).T)
            dW_hc += np.dot(dc_hat_raw, cache["h"][t-1].T)
            db_c += dc_hat_raw

            di = dc * c_hat
            di_raw = i * (1 - i) * di

            dW_xi += np.dot(di_raw, np.array(inputs[t]).reshape(-1, 1).T)
            dW_hi += np.dot(di_raw, cache["h"][t-1].T)
            db_i += di_raw

            df = dc * c_prev
            df_raw = f * (1 - f) * df

            dW_xf += np.dot(df_raw, np.array(inputs[t]).reshape(-1, 1).T)
            dW_hf += np.dot(df_raw, cache["h"][t-1].T)
            db_f += df_raw

            dh_next = (np.dot(self.W_hf.T, df_raw) + 
                       np.dot(self.W_hi.T, di_raw) + 
                       np.dot(self.W_hc.T, dc_hat_raw) + 
                       np.dot(self.W_ho.T, do_raw))
            
            dc_next = dc * f

        for dparam in [dW_xf, dW_hf, dW_xi, dW_hi, dW_xo, dW_ho, dW_xc, dW_hc, dW_hy, db_f, db_i, db_o, db_c, db_y]:
            np.clip(dparam, -5, 5, out=dparam)

        self.W_xf -= learning_rate * dW_xf
        self.W_hf -= learning_rate * dW_hf
        self.b_f -= learning_rate * db_f

        self.W_xi -= learning_rate * dW_xi
        self.W_hi -= learning_rate * dW_hi
        self.b_i -= learning_rate * db_i

        self.W_xo -= learning_rate * dW_xo
        self.W_ho -= learning_rate * dW_ho
        self.b_o -= learning_rate * db_o

        self.W_xc -= learning_rate * dW_xc
        self.W_hc -= learning_rate * dW_hc
        self.b_c -= learning_rate * db_c

        self.W_hy -= learning_rate * dW_hy
        self.b_y -= learning_rate * db_y

        return loss
