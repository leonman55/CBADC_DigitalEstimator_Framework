class SystemVerilogDimension:
    upper_bound: int = 0
    lower_bound: int = 0

    def __init__(self, upper_bound: int = 0, lower_bound: int = 0) -> None:
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

    def get_representation(self) -> str:
        if self.upper_bound != 0 or self.lower_bound != 0:
            return "[" + str(self.upper_bound) + ":" + str(self.lower_bound) + "]"
        else:
            return ""