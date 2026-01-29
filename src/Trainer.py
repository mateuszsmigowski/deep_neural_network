import numpy as np


class Trainer:

    def __init__(self, model, epochs):

        self.model = model
        self.epochs = epochs

    def train(self, X, Y):
        losses = []

        for epoch in range(self.epochs):
            totalLoss = 0

            for i in range(len(X)):
                inputs = X[i]
                target = np.array(Y[i]).reshape(-1, 1)

                outputs, cache = self.model.forward(inputs)
                
                targets_list = [np.zeros_like(outputs[t]) for t in range(len(outputs))]
                targets_list[-1] = target
                
                loss = self.model.backward(outputs, targets_list, cache, loss_only_last=True)

                totalLoss += loss

            losses.append(totalLoss / len(X))
            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {losses[-1]}")

        return losses, self.model

    def validate(self, X, Y):
        loss = 0

        for i in range(len(X)):
            inputs = X[i]
            target = np.array(Y[i]).reshape(-1, 1)

            outputs, _ = self.model.forward(inputs)

            loss += np.sum((outputs[-1] - target) ** 2) / 2

        return loss / len(X)

    def test(self, X, Y, scales=None):
        predictions = []
        targets = []

        for i in range(len(X)):
            inputs = X[i]
            target = Y[i]

            outputs, _ = self.model.forward(inputs)
            pred = outputs[-1].flatten()[0]

            if scales is not None:
                mean, std = scales[i]
                pred = pred * std + mean
                target = target * std + mean

            predictions.append(pred)
            targets.append(target)

        return np.array(predictions), np.array(targets)

    def calculate_naive_baseline(self, X, Y):
        naive_loss = 0
        count = 0

        for i in range(len(X)):
            last_input = X[i][-1, 0]
            target = Y[i]

            naive_loss += (last_input - target) ** 2 / 2
            count += 1

        return naive_loss / count

