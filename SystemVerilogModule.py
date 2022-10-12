import SystemVerilogSyntaxGenerator
import SystemVerilogPort
import SystemVerilogPortDirection
import SystemVerilogPortType
import SystemVerilogSignal


class SystemVerilogModule:
    name: str = ""
    path: str = ""
    syntax_generator: SystemVerilogSyntaxGenerator.SystemVerilogSyntaxGenerator = None

    def __init__(self, path: str, name: str):
        self.name = name
        self.path = path
        self.syntax_generator = SystemVerilogSyntaxGenerator.SystemVerilogSyntaxGenerator(self.path, self.name + ".sv")
        self.port_list: list[SystemVerilogPort.SystemVerilogPort] = list()
        self.parameter_list: dict[str, str] = dict()
        self.submodules: dict[SystemVerilogModule, dict[SystemVerilogPort.SystemVerilogPort, SystemVerilogPort.SystemVerilogPort]] = dict[SystemVerilogModule, dict[SystemVerilogPort.SystemVerilogPort, SystemVerilogPort.SystemVerilogPort]]()


    def generate(self):
        pass

    def add_parameter(self, new_parameter: dict[str, str]) -> dict[str, str]:
        self.parameter_list.update(new_parameter)
        return new_parameter

    def add_port(self, name: str, direction: SystemVerilogPortDirection.SystemVerilogPortDirection, type: SystemVerilogPortType.SystemVerilogPortType, msb: int = -1, lsb: int = -1) -> SystemVerilogPort.SystemVerilogPort:
        new_port = SystemVerilogPort.SystemVerilogPort(name, direction, type, msb, lsb)
        self.port_list.append(new_port)
        return new_port

    def add_signal(self, name: str, type: SystemVerilogPortType.SystemVerilogPortType, msb: int = -1, lsb: int = -1, array_top: int = -1, array_bottom: int = -1) -> SystemVerilogSignal:
        return self.syntax_generator.signal(name, type, msb, lsb, array_top, array_bottom)

    def add_submodule(self, new_submodule, connections: dict[SystemVerilogPort.SystemVerilogPort, SystemVerilogPort.SystemVerilogPort]):
        self.submodules[new_submodule] = connections
