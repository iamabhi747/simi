from .CostFunction import CostFunction

MSD = CostFunction("MSD", lambda x, y: (x - y)**2, lambda x, y: 2*(x - y))

COST_FUNCTIONS = {
    "MSD": MSD
}