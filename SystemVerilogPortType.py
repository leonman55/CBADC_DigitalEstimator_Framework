class SystemVerilogPortType:
    type: str = ""

class NoType(SystemVerilogPortType):

    def __init__(self):
        self.type = ""

class Wire(SystemVerilogPortType):

    def __init__(self):
        self.type = "wire"

class Reg(SystemVerilogPortType):

    def __init__(self):
        self.type = "reg"

class Logic(SystemVerilogPortType):

    def __init__(self):
        self.type = "logic"