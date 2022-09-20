import SystemVerilogModule
import SystemVerilogPort
import SystemVerilogPortDirection
import SystemVerilogPortType
import DigitalEstimatorModules.TestModule


class DigitalEstimatorWrapper(SystemVerilogModule.SystemVerilogModule):
    clk: SystemVerilogPort.SystemVerilogPort = None
    rst: SystemVerilogPort.SystemVerilogPort = None

    def __init__(self, path: str, name: str):
        super().__init__(path, name)

    def generate(self):
        self.clk = self.add_port("clk", SystemVerilogPortDirection.Input(), SystemVerilogPortType.NoType(), -1, -1)
        self.rst = self.add_port("rst", SystemVerilogPortDirection.Input(), SystemVerilogPortType.NoType(), -1, -1)

        test_module: DigitalEstimatorModules.TestModule.TestModule = DigitalEstimatorModules.TestModule.TestModule(self.path, "TestModule")
        test_module.parameter_width = {"alu_width": "32"}
        test_module.parameter_length = {"alu_input_channels": "2"}
        test_module.parameter_alus = {"ALUS": "1"}
        test_module.generate()
        test_module_port_connections: dict[SystemVerilogPort.SystemVerilogPort, SystemVerilogPort.SystemVerilogPort] = dict[SystemVerilogPort.SystemVerilogPort, SystemVerilogPort.SystemVerilogPort]()
        test_module_port_connections[test_module.clk] = self.clk
        test_module_port_connections[test_module.rst] = self.rst

        self.add_submodule(test_module, test_module_port_connections)

        self.syntax_generator.module_head(self.name, self.parameter_list, self.port_list)
        count_modules: int = 0
        module: SystemVerilogModule.SystemVerilogModule = None
        for module in self.submodules.keys():
            self.syntax_generator.module_instance(module.name, module.name + "_" + str(count_modules), module.parameter_list, module.port_list, self.submodules.get(module))
            count_modules += 1
        self.syntax_generator.end_module()