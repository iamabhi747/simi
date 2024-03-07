
class NonLinear:
    def __init__(self, type:str, call, derv) -> None:
        self.type = type
        assert callable(call) and callable(derv), "call and derv must be callable"
        self.call = call
        self.derv = derv