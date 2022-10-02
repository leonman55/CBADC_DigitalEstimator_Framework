import SystemVerilogModule
import SystemVerilogPort
import SystemVerilogPortDirection
import SystemVerilogPortType
import SystemVerilogClockEdge

from SystemVerilogSyntaxGenerator import set_parameter_value_by_parameter, get_parameter_value


class SimpleAdder(SystemVerilogModule.SystemVerilogModule):
    configuration_has_overflow_bit_output: bool = False
    parameter_alu_input_width: dict[str, str] = {"ALU_INPUT_WIDTH": "32"}
    parameter_alu_output_width: dict[str, str] = {"ALU_OUTPUT_WIDTH": "0"}

    clk: SystemVerilogPort.SystemVerilogPort = None
    rst: SystemVerilogPort.SystemVerilogPort = None
    alu_input: list[SystemVerilogPort.SystemVerilogPort] = list[SystemVerilogPort.SystemVerilogPort]()
    alu_output: SystemVerilogPort.SystemVerilogPort = None
    alu_carry_bit: SystemVerilogPort.SystemVerilogPort = None

    def __init__(self, path: str, name: str):
        super().__init__(path, name)

    def generate(self):
        self.parameter_alu_input_width = self.add_parameter(self.parameter_alu_input_width)
        if self.parameter_alu_output_width[next(iter(self.parameter_alu_output_width))] == "0":
            set_parameter_value_by_parameter(self.parameter_alu_output_width, self.parameter_alu_input_width)
        self.parameter_alu_output_width = self.add_parameter(self.parameter_alu_output_width)

        self.clk = self.add_port("clk", SystemVerilogPortDirection.Input(), SystemVerilogPortType.NoType(), -1, -1)
        self.rst = self.add_port("rst", SystemVerilogPortDirection.Input(), SystemVerilogPortType.NoType(), -1, -1)
        alu_input_0 = self.add_port("alu_input_0", SystemVerilogPortDirection.Input(), SystemVerilogPortType.NoType(), int(get_parameter_value(self.parameter_alu_input_width)) - 1, 0)
        self.alu_input.append(alu_input_0)
        alu_input_1 = self.add_port("alu_input_1", SystemVerilogPortDirection.Input(), SystemVerilogPortType.NoType(), int(get_parameter_value(self.parameter_alu_input_width)) - 1, 0)
        self.alu_input.append(alu_input_1)
        self.alu_output = self.add_port("alu_output", SystemVerilogPortDirection.Output(), SystemVerilogPortType.Logic(), int(get_parameter_value(self.parameter_alu_output_width)) - 1, 0)

        self.syntax_generator.timescale()
        self.syntax_generator.module_head(self.name, self.parameter_list, self.port_list)
        self.syntax_generator.always_synchronous({self.clk: SystemVerilogClockEdge.SystemVerilogPosedge()})
        self.syntax_generator.add_signals(self.alu_output, alu_input_0, alu_input_1)
        self.syntax_generator.end_always()
        self.syntax_generator.blank_line()
        self.syntax_generator.end_module()
        self.syntax_generator.close()
