import SystemVerilogModule
import SystemVerilogPort
import SystemVerilogPortDirection
import SystemVerilogPortType
import SystemVerilogClockEdge


class TestModule(SystemVerilogModule.SystemVerilogModule):
    parameter_width: dict[str, str] = {"width": "1"}
    parameter_length: dict[str, str] = {"length": "5"}
    parameter_alus: dict[str, str] = {"ALUs": "4"}

    clk: SystemVerilogPort.SystemVerilogPort = None
    rst: SystemVerilogPort.SystemVerilogPort = None

    def __init__(self, path: str, name: str):
        super().__init__(path, name)

    def generate(self):
        self.parameter_width = self.add_parameter(self.parameter_width)
        self.parameter_length = self.add_parameter(self.parameter_length)
        self.parameter_alus = self.add_parameter(self.parameter_alus)
        self.clk = self.add_port("clk", SystemVerilogPortDirection.Input(), SystemVerilogPortType.NoType(), -1, -1)
        self.rst = self.add_port("rst", SystemVerilogPortDirection.Input(), SystemVerilogPortType.NoType(), -1, -1)

        self.syntax_generator.module_head(self.name, self.parameter_list, self.port_list)
        alu_input_0 = self.add_signal("alu_input_0", SystemVerilogPortType.Logic(), 63, 0, 1, 0)
        alu_input_1 = self.add_signal("alu_input_1", SystemVerilogPortType.Logic(), 63, 0, 1, 0)
        self.syntax_generator.blank_line()
        self.syntax_generator.assign(alu_input_0, alu_input_1)
        self.syntax_generator.blank_line()
        self.syntax_generator.always_combinatorial()
        self.syntax_generator.assign(alu_input_0, alu_input_1)
        self.syntax_generator.end_always()
        self.syntax_generator.blank_line()
        self.syntax_generator.always_synchronous({self.clk: SystemVerilogClockEdge.SystemVerilogPosedge(), self.rst: SystemVerilogClockEdge.SystemVerilogNegedge()})
        self.syntax_generator.assign(alu_input_0, alu_input_1)
        self.syntax_generator.end_always()
        self.syntax_generator.blank_line()
        self.syntax_generator.end_module()
        self.syntax_generator.close()
