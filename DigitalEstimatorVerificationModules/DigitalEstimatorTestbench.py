import random

import SystemVerilogModule
import DigitalEstimatorModules.DigitalEstimatorWrapper
import SystemVerilogPort
import SystemVerilogSignal
import SystemVerilogPortType
from SystemVerilogSyntaxGenerator import get_parameter_value
from SystemVerilogSyntaxGenerator import set_parameter_value_by_parameter
from SystemVerilogSyntaxGenerator import connect_port_array
from SystemVerilogSyntaxGenerator import decimal_number
from SystemVerilogComparisonOperator import *


class DigitalEstimatorTestbench(SystemVerilogModule.SystemVerilogModule):
    configuration_number_of_timesteps_in_clock_cycle: int = 10

    parameter_alu_input_width: dict[str, str] = {"ALU_INPUT_WIDTH": "32"}

    def __init__(self, path: str, name: str):
        super().__init__(path, name)

    def generate(self):
        self.parameter_alu_input_width = self.add_parameter(self.parameter_alu_input_width)

        self.syntax_generator.timescale()
        self.syntax_generator.module_head(self.name)
        clk: SystemVerilogSignal.SystemVerilogSignal = self.add_signal("clk", SystemVerilogPortType.Logic())
        rst: SystemVerilogSignal.SystemVerilogSignal = self.add_signal("rst", SystemVerilogPortType.Logic())
        alu_inputs: list[SystemVerilogSignal.SystemVerilogSignal] = list[SystemVerilogSignal.SystemVerilogSignal]()
        alu_input_0: SystemVerilogSignal.SystemVerilogSignal = self.add_signal("input_0", SystemVerilogPortType.Logic(), int(get_parameter_value(self.parameter_alu_input_width)) - 1, 0)
        alu_inputs.append(alu_input_0)
        alu_input_1: SystemVerilogSignal.SystemVerilogSignal = self.add_signal("input_1", SystemVerilogPortType.Logic(), int(get_parameter_value(self.parameter_alu_input_width)) - 1, 0)
        alu_inputs.append(alu_input_1)
        alu_output: SystemVerilogSignal.SystemVerilogSignal = self.add_signal("alu_output", SystemVerilogPortType.Logic(), int(get_parameter_value(self.parameter_alu_input_width)) - 1, 0)
        self.syntax_generator.blank_line()
        self.syntax_generator.generate_clock(clk, self.configuration_number_of_timesteps_in_clock_cycle)
        self.syntax_generator.initial()
        self.syntax_generator.assign_construct(rst, decimal_number(0))
        for alu_input in alu_inputs:
            self.syntax_generator.assign_construct(alu_input, decimal_number(0))
        self.syntax_generator.blank_line()
        for stimuli_counter in range(5):
            values_alu_inputs: list[int] = list[int]()
            for alu_input in alu_inputs:
                value_alu_input: int = random.randint(0, 1000)
                values_alu_inputs.append(value_alu_input)
                self.syntax_generator.assign_construct(alu_input, decimal_number(value_alu_input))
            result_alu_output: int = 0
            for value in values_alu_inputs:
                result_alu_output += value
            self.syntax_generator.wait_timesteps(self.configuration_number_of_timesteps_in_clock_cycle)
            self.syntax_generator.assert_signal_construct(alu_output, decimal_number(result_alu_output), Equal, True, True)
            self.syntax_generator.blank_line()
        self.syntax_generator.finish()
        self.syntax_generator.end_initial()

        dut: DigitalEstimatorModules.DigitalEstimatorWrapper.DigitalEstimatorWrapper = DigitalEstimatorModules.DigitalEstimatorWrapper.DigitalEstimatorWrapper(self.path, "DigitalEstimator")
        set_parameter_value_by_parameter(self.parameter_alu_input_width, dut.parameter_alu_input_width)
        dut.generate()
        dut_port_connections: dict[SystemVerilogPort.SystemVerilogPort, SystemVerilogPort.SystemVerilogPort] = dict[SystemVerilogPort.SystemVerilogPort, SystemVerilogPort.SystemVerilogPort]()
        dut_port_connections[dut.clk] = clk
        dut_port_connections[dut.rst] = rst
        dut_port_connections.update(connect_port_array(dut.adder_input, alu_inputs))
        dut_port_connections[dut.adder_output] = alu_output

        self.add_submodule(dut, dut_port_connections)

        self.syntax_generator.instantiate_submodules(self.submodules)

        self.syntax_generator.blank_line()
        self.syntax_generator.end_module()
        self.syntax_generator.close()