from .CostFunction import CostFunction

MSE = CostFunction("MSE", lambda x, y: (x - y)**2, lambda x, y: 2*(x - y))

COST_FUNCTIONS = {
    "MSE": MSE
}