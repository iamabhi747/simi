
class NonLinear:
    def __init__(self, name:str, call, derv) -> None:
        ## Name of the non-linear function
        self.name = name
        assert callable(call) and callable(derv), "call and derv must be callable"

        ## Call: The non-linear function
        self.call = call

        ## Derv: The derivative of the non-linear function
        self.derv = derv

    ## Define the __call__ method to allow for the object to be called as a function
    def __call__(self, *args, **kwds):
        return self.call(*args, **kwds)