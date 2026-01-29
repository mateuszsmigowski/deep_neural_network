import matplotlib.pyplot as plt
import numpy as np


class Visualizer:

    def __init__(self):
        pass

    def plot_loss(self, losses, filename="loss.png"):

        plt.figure(figsize=(10, 5))
        plt.plot(losses, label="Training Loss")
        plt.title("Training Loss over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()
        print(f"Loss plot saved to {filename}")

    def plot_predictions(
        self,
        targets,
        predictions,
        title="Predictions vs Targets",
        filename="predictions.png",
    ):

        plt.figure(figsize=(10, 5))
        targets = np.array(targets).flatten()
        predictions = np.array(predictions).flatten()

        plt.plot(targets, label="Target", linestyle="--")
        plt.plot(predictions, label="Prediction")
        plt.title(title)
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()
        print(f"Prediction plot saved to {filename}")

    def plot_zoomed_predictions(
        self,
        targets,
        predictions,
        num_points=50,
        title="Zoomed Predictions vs Targets (Last 50 steps)",
        filename="predictions_zoomed.png",
    ):
        plt.figure(figsize=(10, 5))

        targets = np.array(targets).flatten()
        predictions = np.array(predictions).flatten()

        targets = targets[-num_points:]
        predictions = predictions[-num_points:]

        plt.plot(targets, label="Target", linestyle="--", marker="o", markersize=4)
        plt.plot(predictions, label="Prediction", marker="x", markersize=4)
        plt.title(title)
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()
        print(f"Zoomed prediction plot saved to {filename}")
