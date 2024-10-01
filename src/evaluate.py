import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import load_model

def load_test_data():
    pass

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R-squared Score: {r2}")

    return predictions

def plot_predictions(y_test, predictions):
    plt.figure(figsize=(12, 6))
    plt.plot(y_test[:, 0], label='Actual')
    plt.plot(predictions[:, 0], label='Predicted')
    plt.legend()
    plt.title('Actual vs Predicted Values (First Target Variable)')
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.show()

def plot_residuals(y_test, predictions):
    residuals = y_test - predictions
    
    plt.figure(figsize=(10, 6))
    plt.scatter(predictions, residuals)
    plt.title('Residuals vs Predicted')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=50)
    plt.title('Distribution of Residuals')
    plt.xlabel('Residual Value')
    plt.ylabel('Frequency')
    plt.show()

if __name__ == "__main__":
    # Load the model
    model = load_model('lob_full_model.keras')

    # Load test data
    X_test, y_test = load_test_data()

    # Evaluate the model
    predictions = evaluate_model(model, X_test, y_test)

    # Plot predictions vs actual
    plot_predictions(y_test, predictions)

    # Plot residuals
    plot_residuals(y_test, predictions)