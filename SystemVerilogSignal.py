import SystemVerilogPort
import SystemVerilogPortType


class SystemVerilogSignal(SystemVerilogPort.SystemVerilogPort):
    signal_array_top: int = -1
    signal_array_bottom: int = -1

    def __init__(self, signal_name: str, signal_type: SystemVerilogPortType.SystemVerilogPortType, signal_msb: int, signal_lsb: int, signal_array_top: int, signal_array_bottom: int):
        super().__init__(signal_name, None, signal_type, signal_msb, signal_lsb)
        self.signal_array_top = signal_array_top
        self.signal_array_bottom = signal_array_bottom
