import SystemVerilogDimension
import SystemVerilogPortType
import SystemVerilogSignalSign


class SystemVerilogLocalParameter():
    name: str = ""
    type: SystemVerilogPortType.SystemVerilogPortType = SystemVerilogPortType.NoType()
    sign: SystemVerilogSignalSign.SystemVerilogSignalSign = SystemVerilogSignalSign.NoSign()
    msb: int = 0
    lsb: int = 0
    dimensions: list[SystemVerilogDimension.SystemVerilogDimension] = list[SystemVerilogDimension.SystemVerilogDimension]()
    values: str = ""

    def __init__(self, name: str, type: SystemVerilogPortType.SystemVerilogPortType = SystemVerilogPortType.NoType(),
        sign: SystemVerilogSignalSign.SystemVerilogSignalSign = SystemVerilogSignalSign.NoSign(), msb: int = 0, lsb: int = 0,
        dimensions: list[SystemVerilogDimension.SystemVerilogDimension] = list[SystemVerilogDimension.SystemVerilogDimension](),
        values: str = "") -> None:
        self.name = name
        self.type = type
        self.sign = sign
        self.msb = msb
        self.lsb = lsb
        self.dimensions = dimensions
        self.values = values
