class SystemVerilogSignalSign:
    type: str = ""

class NoSign(SystemVerilogSignalSign):
    
    def __init__(self) -> None:
        super().__init__()
        self.type = ""

class Signed(SystemVerilogSignalSign):
    
    def __init__(self) -> None:
        super().__init__()
        self.type = "signed"

class Unsigned(SystemVerilogSignalSign):

    def __init__(self) -> None:
        super().__init__()
        self.type = "unsigned"
        