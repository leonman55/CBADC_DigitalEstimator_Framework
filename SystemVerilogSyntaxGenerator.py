import math
from threading import local

from numpy import array_str, ndarray

import FileGenerator
import SystemVerilogModule
import SystemVerilogPort
import SystemVerilogPortType
import SystemVerilogSignalSign
import SystemVerilogDimension
import SystemVerilogSignal
import SystemVerilogLocalParameter
import SystemVerilogClockEdge
from SystemVerilogComparisonOperator import *


def set_parameter_value(parameter: dict[str, str], value: str):
    parameter[next(iter(parameter))] = value

def set_parameter_value_by_parameter(parameter: dict[str, str], source_parameter: dict[str, str]):
    parameter[next(iter(parameter))] = source_parameter[next(iter(source_parameter))]

def get_parameter_name(parameter: dict[str, str]) -> str:
    return next(iter(parameter))

def get_parameter_value(parameter: dict[str, str]) -> str:
    return parameter[next(iter(parameter))]

def connect_port_array(external_ports: list[SystemVerilogPort.SystemVerilogPort], internal_ports: list[SystemVerilogPort.SystemVerilogPort]) -> dict[SystemVerilogPort.SystemVerilogPort, SystemVerilogPort.SystemVerilogPort]:
    port_connections: dict[SystemVerilogPort.SystemVerilogPort, SystemVerilogPort.SystemVerilogPort] = dict[SystemVerilogPort.SystemVerilogPort, SystemVerilogPort.SystemVerilogPort]()
    external_ports_sorted: list[SystemVerilogPort.SystemVerilogPort] = external_ports.copy()
    external_ports_sorted.sort()
    internal_ports_sorted: list[SystemVerilogPort.SystemVerilogPort] = internal_ports.copy()
    internal_ports_sorted.sort()
    for index in range(len(external_ports_sorted)):
        port_connections[external_ports_sorted[index]] = internal_ports_sorted[index]
    return port_connections

def decimal_number(value: int) -> str:
    if value == 0:
        number_of_bits = 1
    else:
        number_of_bits: int = math.ceil(math.log2(abs(value) + 1))
    if value < 0:
        return "-" + str(number_of_bits) + "'d" + str(abs(value))
    else:
        return str(number_of_bits) + "'d" + str(value)

def ndarray_to_system_verilog_array(input_array: ndarray) -> str:
    total_number_of_elements: int = 1
    dimensions: list[int] = list[int]()
    array_string: str = ""
    for dimension_index in range(len(input_array.shape)):
        total_number_of_elements *= input_array.shape[dimension_index]
        if input_array.shape[dimension_index] != 1:
            array_string += "'{"
            dimensions.append(input_array.shape[dimension_index])
    input_array_flattened = input_array.reshape(total_number_of_elements)
    dimension_breaks: list[int] = dimensions.copy()
    for dimension_index in range(len(dimensions) - 1, -1, -1):
        if dimension_index == len(dimensions) - 1:
            continue
        else:
            dimension_breaks[dimension_index] = dimension_breaks[dimension_index] * dimension_breaks[dimension_index + 1]
    print("Total number of elements: ", total_number_of_elements)
    print("Dimensions: ", dimensions)
    print("Dimension breaks: ", dimension_breaks)
    for element_index in range(total_number_of_elements):
        array_string += decimal_number(input_array_flattened[element_index])
        for dimension_break in dimension_breaks:
            if ((element_index + 1) % dimension_break) == 0:
                array_string += "}"
        if element_index + 1 != total_number_of_elements:
            array_string += ", "
            for dimension_break in dimension_breaks:
                if ((element_index + 1) % dimension_break) == 0:
                    array_string += "'{"
    print("System Verilog array: \n", array_string)
    return array_string

def ndarray_to_system_verilog_concatenation(input_array: list[list[int]], array_bit_widths: list[tuple[int, int]]) -> str:
    system_verilog_concatenation: str = "{"
    for lut_index in range(len(input_array)):
        for element_index in range(len(input_array[lut_index])):
            if input_array[lut_index][element_index] < 0:
                system_verilog_concatenation += "-"
            system_verilog_concatenation += str(array_bit_widths[lut_index][0]) + "'d" + str(abs(input_array[lut_index][element_index])) + ", "
    system_verilog_concatenation = system_verilog_concatenation.removesuffix(", ")
    system_verilog_concatenation += "}"
    return system_verilog_concatenation
        

class SystemVerilogSyntaxGenerator:
    output: FileGenerator.FileGenerator = None
    indentation_level: int = 0
    combinatorial: bool = 0
    synchronous: bool = 0
    assertion_count: int = 0

    def __init__(self, path: str, name: str):
        self.output = FileGenerator.FileGenerator()
        self.output.set_path(path)
        self.output.set_name(name)
        self.output.open_output_file()

    def close(self):
        self.output.close_output_file()

    def get_indentation(self) -> str:
        indentation: str = ""
        for level in range(self.indentation_level):
            indentation += "\t"
        return indentation

    def single_line_no_linebreak(self, line: str, indentation: bool = 1):
        if indentation == 1:
            self.output.write_line_no_linebreak(self.get_indentation() + line)
        else:
            self.output.write_line_no_linebreak(line)

    def single_line_linebreak(self, line: str, indentation: bool = 1):
        if indentation == 1:
            self.output.write_line_linebreak(self.get_indentation() + line)
        else:
            self.output.write_line_linebreak(line)

    def blank_line(self):
        self.single_line_linebreak("", 0)

    def module_head(self, module_name: str, parameter_list: dict = None, port_list: list = None):
        self.single_line_no_linebreak("module " + module_name)
        if parameter_list is not None and len(parameter_list) != 0:
            self.single_line_no_linebreak(" ", False)
            self.parameter_list(parameter_list)
        else:
            self.indentation_level += 1
        if port_list is not None and len(port_list) != 0:
            self.single_line_no_linebreak(" ", False)
            self.port_list(port_list)
        else:
            self.single_line_linebreak("();", False)
        self.blank_line()

    def end_module(self):
        self.indentation_level -= 1
        self.single_line_no_linebreak("endmodule")

    def parameter_list(self, parameter_list: dict):
        if parameter_list is None or len(parameter_list) == 0:
            return
        self.single_line_linebreak("#(", bool(0))
        self.indentation_level += 2
        count_parameter: int = 0
        for parameter in parameter_list.keys():
            count_parameter += 1
            if count_parameter == len(parameter_list):
                self.single_line_linebreak("parameter " + parameter + " = " + parameter_list.get(parameter))
            else:
                self.single_line_linebreak("parameter " + parameter + " = " + parameter_list.get(parameter) + ",")
        self.indentation_level -= 1
        self.single_line_no_linebreak(")")

    def port_list(self, port_list: list):
        if port_list is None or len(port_list) == 0:
            return
        self.single_line_linebreak("(", bool(0))
        self.indentation_level += 1
        count_ports: int = 0
        port: SystemVerilogPort.SystemVerilogPort = None
        for port in port_list:
            count_ports += 1
            if count_ports == len(port_list):
                if port.port_msb < 0 or port.port_lsb < 0:
                    if isinstance(port.port_type, SystemVerilogPortType.NoType):
                        self.single_line_linebreak(port.port_direction.direction + " " + port.port_name)
                    else:
                        self.single_line_linebreak(port.port_direction.direction + " " + port.port_type.type + " " + port.port_name)
                else:
                    if isinstance(port.port_type, SystemVerilogPortType.NoType):
                        self.single_line_linebreak(port.port_direction.direction + " [" + str(port.port_msb) + ":" + str(port.port_lsb) + "] " + port.port_name)
                    else:
                        self.single_line_linebreak(port.port_direction.direction + " " + port.port_type.type + " [" + str(port.port_msb) + ":" + str(port.port_lsb) + "] " + port.port_name)
            else:
                if port.port_msb < 0 or port.port_lsb < 0:
                    if isinstance(port.port_type, SystemVerilogPortType.NoType):
                        self.single_line_linebreak(port.port_direction.direction + " " + port.port_name + ",")
                    else:
                        self.single_line_linebreak(port.port_direction.direction + " " + port.port_type.type + " " + port.port_name + ",")
                else:
                    if isinstance(port.port_type, SystemVerilogPortType.NoType):
                        self.single_line_linebreak(port.port_direction.direction + " [" + str(port.port_msb) + ":" + str(port.port_lsb) + "] " + port.port_name + ",")
                    else:
                        self.single_line_linebreak(port.port_direction.direction + " " + port.port_type.type + " [" + str(port.port_msb) + ":" + str(port.port_lsb) + "] " + port.port_name + ",")
        self.indentation_level -= 2
        self.single_line_linebreak(");")
        self.indentation_level += 1

    def signal_representation(self, signal: SystemVerilogPort.SystemVerilogPort, msb: int = -1, lsb: int = -1, array_index: int = -1, array_top: int = -1, array_bottom: int = -1, initialization: int = 0) -> str:
        if initialization == 1:
            if msb < 0 or lsb < 0:
                if array_top < 0 or array_bottom < 0:
                    return signal.port_type.type + " " + signal.port_name + ";"
                else:
                    return signal.port_type.type + " " + signal.port_name + " [" + str(array_top) + ":" + str(array_bottom) + "];"
            else:
                if array_top < 0 or array_bottom < 0:
                    return signal.port_type.type + " [" + str(signal.port_msb) + ":" + str(signal.port_lsb) + "] " + signal.port_name + ";"
                else:
                    return signal.port_type.type + " [" + str(signal.port_msb) + ":" + str(signal.port_lsb) + "] " + signal.port_name + " [" + str(array_top) + ":" + str(array_bottom) + "];"
        else:
            if msb < 0 or lsb < 0:
                if array_index < 0:
                    return signal.port_name
                else:
                    return signal.port_name + "[" + str(array_index) + "]"
            else:
                if array_index < 0:
                    if array_top < 0 or array_bottom < 0:
                        return signal.port_name + "[" + str(msb) + ":" + str(lsb) + "]"
                    else:
                        return signal.port_name + "[" + str(msb) + ":" + str(lsb) + "][" + str(array_top) + ":" + str(array_bottom) + "]"
                else:
                    return signal.port_name + "[" + str(msb) + ":" + str(lsb) + "][" + str(array_index) + "]"

    def signal(self, signal_name: str, signal_type: SystemVerilogPortType.SystemVerilogPortType, signal_msb: int, signal_lsb: int, signal_array_top: int, signal_array_bottom: int):
        signal = SystemVerilogSignal.SystemVerilogSignal(signal_name, signal_type, signal_msb, signal_lsb, signal_array_top, signal_array_bottom)
        self.single_line_linebreak(self.signal_representation(signal, signal.port_msb, signal.port_lsb, -1, signal.signal_array_top, signal.signal_array_bottom, 1))
        return signal

    def local_parameter(self, name: str, type: SystemVerilogPortType.SystemVerilogPortType = SystemVerilogPortType.NoType(),
        sign: SystemVerilogSignalSign.SystemVerilogSignalSign = SystemVerilogSignalSign.NoSign(), msb: int = 0, lsb: int = 0,
        dimensions: list[SystemVerilogDimension.SystemVerilogDimension] = list[SystemVerilogDimension.SystemVerilogDimension](),
        values: str = "") -> SystemVerilogLocalParameter.SystemVerilogLocalParameter:
        local_parameter: SystemVerilogLocalParameter.SystemVerilogLocalParameter = SystemVerilogLocalParameter.SystemVerilogLocalParameter(name, type, sign, msb, lsb, dimensions, values)
        initialization_string: str = ""
        initialization_string += "localparam"
        if not isinstance(local_parameter.type, SystemVerilogPortType.NoType):
            initialization_string += " " + local_parameter.type.type
        if not isinstance(local_parameter.sign, SystemVerilogSignalSign.NoSign):
            initialization_string += " " + local_parameter.sign.type
        initialization_string += " [" + str(local_parameter.msb) + ":" + str(local_parameter.lsb) + "]"
        initialization_string += " " + local_parameter.name
        dimension: SystemVerilogDimension.SystemVerilogDimension = None
        for dimension in local_parameter.dimensions:
            initialization_string += dimension.get_representation()
        initialization_string += " = "
        initialization_string += values
        initialization_string += ";"
        self.single_line_linebreak(initialization_string)
        self.blank_line()

    def assign(self, left_side: SystemVerilogPort.SystemVerilogPort, right_side: SystemVerilogPort.SystemVerilogPort, left_side_msb: int = -1, left_side_lsb: int = -1, left_side_array_index: int = -1, right_side_msb: int = -1, right_side_lsb: int = -1, rifht_side_array_index: int = -1):
        if self.combinatorial == 1:
            self.single_line_linebreak(self.signal_representation(left_side,left_side_msb, left_side_lsb, left_side_array_index) + " = " + self.signal_representation(right_side, right_side_msb, right_side_lsb, rifht_side_array_index) + ";")
        elif self.synchronous == 1:
            self.single_line_linebreak(self.signal_representation(left_side, left_side_msb, left_side_lsb, left_side_array_index) + " <= " + self.signal_representation(right_side, right_side_msb, right_side_lsb, rifht_side_array_index) + ";")
        else:
            self.single_line_linebreak("assign " + self.signal_representation(left_side, left_side_msb, left_side_lsb, left_side_array_index) + " = " + self.signal_representation(right_side, right_side_msb, right_side_lsb, rifht_side_array_index) + ";")

    def assign_construct(self, left_side: SystemVerilogPort.SystemVerilogPort, right_side: str, left_side_msb: int = -1, left_side_lsb: int = -1, left_side_array_index: int = -1):
        if self.combinatorial == 1:
            self.single_line_linebreak(self.signal_representation(left_side,left_side_msb, left_side_lsb, left_side_array_index) + " = " + right_side + ";")
        elif self.synchronous == 1:
            self.single_line_linebreak(self.signal_representation(left_side, left_side_msb, left_side_lsb, left_side_array_index) + " <= " + right_side + ";")
        else:
            self.single_line_linebreak("assign " + self.signal_representation(left_side, left_side_msb, left_side_lsb, left_side_array_index) + " = " + right_side + ";")

    def always(self):
        self.combinatorial = bool(1)
        self.single_line_linebreak("always begin")
        self.indentation_level += 1

    def always_combinatorial(self):
        self.combinatorial = bool(1)
        self.single_line_linebreak("always_comb begin")
        self.indentation_level += 1

    def always_synchronous(self, sensitivity_list: dict):
        self.synchronous = bool(1)
        sensitivity_list_string: str = ""
        count_signals: int = 0
        for signal in sensitivity_list.keys():
            count_signals += 1
            edge_type: SystemVerilogClockEdge.SystemVerilogClockEdge = sensitivity_list.get(signal)
            if count_signals == len(sensitivity_list):
                sensitivity_list_string += edge_type.edge_type + " " + signal.port_name
            else:
                sensitivity_list_string += edge_type.edge_type + " " + signal.port_name + ", "
        self.single_line_linebreak("always_ff @(" + sensitivity_list_string + ") begin")
        self.indentation_level += 1

    def end_always(self):
        self.combinatorial = bool(0)
        self.synchronous = bool(0)
        self.indentation_level -= 1
        self.single_line_linebreak("end")

    def module_instance(self, module_name: str, instance_name: str, parameter_list: dict[str, str], port_list: list[SystemVerilogPort.SystemVerilogPort], connections: dict[SystemVerilogPort.SystemVerilogPort, SystemVerilogPort.SystemVerilogPort]):
        self.single_line_no_linebreak(module_name)
        self.indentation_level += 1
        if parameter_list is not None and len(parameter_list) > 0:
            self.single_line_linebreak(" #(", False)
            self.indentation_level += 1
            count_parameters: int = 0
            for parameter in parameter_list:
                count_parameters += 1
                if count_parameters == len(parameter_list):
                    self.single_line_linebreak("." + parameter + "(" + parameter_list.get(parameter) + ")")
                else:
                    self.single_line_linebreak("." + parameter + "(" + parameter_list.get(parameter) + "),")
            self.indentation_level -= 1
            self.single_line_linebreak(")")
        self.single_line_linebreak(instance_name + " (")
        count_ports: int = 0
        self.indentation_level += 1
        for module_port in connections:
            count_ports += 1
            if count_ports == len(connections):
                self.single_line_linebreak("." + module_port.port_name + "(" + connections.get(module_port).port_name + ")")
            else:
                self.single_line_linebreak("." + module_port.port_name + "(" + connections.get(module_port).port_name + "),")
        self.indentation_level -= 2
        self.single_line_linebreak(");")
        self.blank_line()

    def timescale(self, timestep: str = "1ns", simulation_unit: str = "10ps"):
        self.single_line_linebreak("`timescale " + timestep + "/" + simulation_unit)
        self.blank_line()

    def add_signals(self, result: SystemVerilogPort.SystemVerilogPort, operand0: SystemVerilogPort.SystemVerilogPort, operand1: SystemVerilogPort.SystemVerilogPort):
        addition: str = self.signal_representation(operand0) + " + " + self.signal_representation(operand1)
        self.assign_construct(result, addition)

    def instantiate_submodules(self, submodules: dict[SystemVerilogModule, dict[SystemVerilogPort.SystemVerilogPort, SystemVerilogPort.SystemVerilogPort]]):
        count_modules: int = 0
        module: SystemVerilogModule.SystemVerilogModule = None
        for module in submodules.keys():
            self.module_instance(module.name, module.name + "_" + str(count_modules), module.parameter_list, module.port_list, submodules.get(module))
            count_modules += 1

    def initial(self):
        self.single_line_linebreak("initial begin")
        self.indentation_level += 1

    def end_initial(self):
        self.indentation_level -= 1
        self.single_line_linebreak("end")
        self.blank_line()

    def wait_timesteps(self, number_of_timesteps: int):
        self.single_line_linebreak("#" + str(number_of_timesteps) + ";")

    def generate_clock(self, clock_signal: SystemVerilogSignal.SystemVerilogSignal, number_of_cycle_timesteps: int):
        self.always()
        self.assign_construct(clock_signal, "1'b0")
        self.wait_timesteps(int(number_of_cycle_timesteps / 2))
        self.assign_construct(clock_signal, "1'b1")
        self.wait_timesteps(int(number_of_cycle_timesteps / 2))
        self.end_always()
        self.blank_line()

    def assert_signals(self, left_side: SystemVerilogPort.SystemVerilogPort, right_side: SystemVerilogPort.SystemVerilogPort, comparison_operator: SystemVerilogComparisonOperator, show_pass_message: bool = False, show_fail_message: bool = False):
        self.assertion_count += 1
        if show_pass_message == False and show_fail_message == False:
            self.single_line_linebreak("assert (" + self.signal_representation(left_side) + " " + comparison_operator.operator + " " + self.signal_representation(right_side) + ");")
        else:
            self.single_line_linebreak("assert (" + self.signal_representation(left_side) + " " + comparison_operator.operator + " " + self.signal_representation(right_side) + ")")
            if show_pass_message == True:
                self.indentation_level += 1
                self.display_pass_message_port_port(left_side, right_side)
                self.indentation_level -= 1
            if show_fail_message == True:
                self.single_line_linebreak("else")
                self.indentation_level += 1
                self.display_expected_value_port_port(left_side, right_side)
                self.indentation_level -=1

    def assert_signal_construct(self, left_side: SystemVerilogPort.SystemVerilogPort, right_side: str, comparison_operator: SystemVerilogComparisonOperator, show_pass_message: bool = False, show_fail_message: bool = False):
        self.assertion_count += 1
        if show_pass_message == False and show_fail_message == False:
            self.single_line_linebreak("assert (" + self.signal_representation(left_side) + " " + comparison_operator.operator + " " + right_side + ");")
        else:
            self.single_line_linebreak("assert (" + self.signal_representation(left_side) + " " + comparison_operator.operator + " " + right_side + ")")
            if show_pass_message == True:
                self.indentation_level += 1
                self.display_pass_message_port_value(left_side, right_side)
                self.indentation_level -= 1
            if show_fail_message == True:
                self.single_line_linebreak("else")
                self.indentation_level += 1
                self.display_expected_value_port_value(left_side, right_side)
                self.indentation_level -=1

    def display_pass_message_port_value(self, signal: SystemVerilogPort.SystemVerilogPort, expected_value: str):
        self.single_line_linebreak("$display(\"PASS: The signal: " + signal.port_name + " has the expected value: " + expected_value + ".\");")

    def display_pass_message_port_port(self, signal: SystemVerilogPort.SystemVerilogPort, expected_value_port: SystemVerilogPort.SystemVerilogPort):
        self.single_line_linebreak("$display(\"PASS: The signal: " + signal.port_name + " has the expected value: %d.\", " + expected_value_port.port_name + ");")

    def display_expected_value_port_value(self, signal: SystemVerilogPort.SystemVerilogPort, expected_value: str):
        self.single_line_linebreak("$display(\"FAIL: The signal: " + signal.port_name + " was expected to have the value: " + expected_value + ", but it has value: %d\", " + signal.port_name + ");")

    def display_expected_value_port_port(self, signal: SystemVerilogPort.SystemVerilogPort, expected_value_port: SystemVerilogPort.SystemVerilogPort):
        self.single_line_linebreak("$display(\"FAIL: The signal: " + signal.port_name + " was expected to be the same as: " + expected_value_port.port_name + ", but the values are: %d\t%d\", " + signal.port_name + ", " + expected_value_port.port_name + ");")
        
    def finish(self):
        self.single_line_linebreak("$finish;")
        