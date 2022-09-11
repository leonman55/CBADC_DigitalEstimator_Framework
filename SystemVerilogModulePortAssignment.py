import SystemVerilogPort


class SystemVerilogModulePortAssignment:
    module_port: SystemVerilogPort.SystemVerilogPort = None
    module_port_msb: int = -1
    module_port_lsb: int = -1
    external_port: SystemVerilogPort.SystemVerilogPort = None
    external_port_msb: int = -1
    external_port_lsb: int = -1

    def __init__(self, module_port: SystemVerilogPort.SystemVerilogPort, module_port_msb: int, module_port_lsb: int, external_port: SystemVerilogPort.SystemVerilogPort, external_port_msb: int, external_port_lsb: int):
        self.module_port = module_port
        self.module_port_msb = module_port_msb
        self.module_port_lsb = module_port_lsb
        self.external_port = external_port
        self.external_port_msb = external_port_msb
        self.external_port_lsb = external_port_lsb