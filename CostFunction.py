
class CostFunction:
    def __init__(self, type, call, derv):
        self.type = type
        assert callable(call) and callable(derv), "call and derv must be callable"
        self.call = call
        self.derv = derv

