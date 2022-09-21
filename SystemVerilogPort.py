import SystemVerilogPortDirection
import SystemVerilogPortType


class SystemVerilogPort:
    port_name: str = ""
    port_direction: SystemVerilogPortDirection.SystemVerilogPortDirection = None
    port_type: SystemVerilogPortType.SystemVerilogPortType = None
    port_msb: int = -1
    port_lsb: int = -1

    def __init__(self, port_name: str, port_direction: SystemVerilogPortDirection.SystemVerilogPortDirection, port_type: SystemVerilogPortType.SystemVerilogPortType, port_msb: int, port_lsb: int):
        self.port_name = port_name
        self.port_direction = port_direction
        self.port_type = port_type
        self.port_msb = port_msb
        self.port_lsb = port_lsb

    def __lt__(self, other):
        if isinstance(other, SystemVerilogPort):
            return self.port_name < other.port_name
        else:
            return 0
