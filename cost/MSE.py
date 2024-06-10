# MSE: Mean Squared Error
import numpy as np
from .. import cost

## Function to calculate the Mean Squared Error
# y     [output_dim,]: Actual values
# y_hat [output_dim,]: Predicted values
def MSE_function(y:np.ndarray, y_hat:np.ndarray) -> float:
    return np.mean((y - y_hat) ** 2)

## Function to calculate the derivative of the Mean Squared Error
# y     [output_dim,]: Actual values
# y_hat [output_dim,]: Predicted values
def MSE_derivative(y:np.ndarray, y_hat:np.ndarray) -> np.ndarray:
    return 2 * (y - y_hat)


MSE = cost.CostFunction("MSE", MSE_function, MSE_derivative)