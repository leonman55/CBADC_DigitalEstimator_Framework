class SystemVerilogPortDirection:
    direction: str = ""

class Input(SystemVerilogPortDirection):

    def __init__(self):
        self.direction = "input"

class Output(SystemVerilogPortDirection):

    def __init__(self):
        self.direction = "output"
