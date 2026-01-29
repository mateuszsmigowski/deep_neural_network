from src.FinancialData import FinancialData
from src.SimpleRNN import SimpleRNN
from src.Trainer import Trainer
from src.Visualizer import Visualizer
from src.DataFormatter import DataFormatter
import numpy as np


def main():

    input_size = 1
    hidden_size = 20
    output_size = 1
    seq_length = 30
    learning_rate = 0.001
    epochs = 20

    financialDataObject = FinancialData()
    financialData = financialDataObject.get_financial_data(
        "AAPL", "2020-01-01", "2024-01-01"
    )

    dataFormatter = DataFormatter()
    financialData, minVal, maxVal = dataFormatter.min_max_scale(financialData)
    X, Y = dataFormatter.create_sequences(financialData, seq_length)
    X_train, Y_train, X_val, Y_val, X_test, Y_test = dataFormatter.split_data(X, Y)

    rnn = SimpleRNN(input_size, hidden_size, output_size)
    visualizer = Visualizer()
    trainer = Trainer(rnn, epochs)

    losses, rnn = trainer.train(X_train, Y_train)
    validationLoss = trainer.validate(X_val, Y_val)
    visualizer.plot_loss(losses)

    predictions, targets = trainer.test(X_test, Y_test)
    originalPredictions = dataFormatter.inverse_min_max_scale(
        predictions, minVal, maxVal
    )
    originalTargets = dataFormatter.inverse_min_max_scale(targets, minVal, maxVal)

    visualizer.plot_predictions(originalTargets, originalPredictions)
    visualizer.plot_zoomed_predictions(originalTargets, originalPredictions)

    naive_mse = trainer.calculate_naive_baseline(X_test, Y_test)
    model_mse = np.mean((predictions - targets) ** 2) / 2

    print(f"\n--- Model Evaluation ---")
    print(f"Model MSE (scaled): {model_mse:.6f}")
    print(f"Naive Baseline MSE (scaled): {naive_mse:.6f}")

    if model_mse < naive_mse:
        print("SUCCESS: Model outperforms naive baseline!")
    else:
        print("WARNING: Model performs worse or equal to naive baseline (Persistence).")


if __name__ == "__main__":
    main()
