
class NonLinear:
    def __init__(self, type:str, call) -> None:
        self.type = type
        assert callable(call)
        self.call = call