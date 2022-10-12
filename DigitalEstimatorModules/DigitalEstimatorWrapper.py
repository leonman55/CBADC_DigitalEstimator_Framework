from tkinter.messagebox import NO
import SystemVerilogModule
import SystemVerilogPort
import SystemVerilogPortDirection
import SystemVerilogPortType
import SystemVerilogDimension
import DigitalEstimatorModules.TestModule
import DigitalEstimatorModules.SimpleAdder
import SystemVerilogSignalSign
from SystemVerilogSyntaxGenerator import decimal_number, get_parameter_value, connect_port_array, ndarray_to_system_verilog_array, set_parameter_value
import CBADC_HighLevelSimulation
import cbadc


class DigitalEstimatorWrapper(SystemVerilogModule.SystemVerilogModule):
    configuration_n_number_of_analog_states: int = 6
    configuration_m_number_of_digital_states: int = configuration_n_number_of_analog_states
    configuration_beta: float = 6250.0
    configuration_rho: float = -1e-2
    configuration_kappa: float = -1.0
    configuration_eta2: float = 1e7
    configuration_lookback_length: int = 5
    configuration_lookahead_length: int = 1

    parameter_control_signal_input_width: dict[str, str] = {"CONTROL_SIGNAL_INPUT_WIDTH": str(configuration_m_number_of_digital_states)}
    parameter_alu_input_width: dict[str, str] = {"ALU_INPUT_WIDTH": "32"}

    clk: SystemVerilogPort.SystemVerilogPort = None
    rst: SystemVerilogPort.SystemVerilogPort = None
    control_signal_sample_input: SystemVerilogPort.SystemVerilogPort = None
    signal_estimation_output: SystemVerilogPort.SystemVerilogPort = None
    filter_coefficient_shift_register_enable: SystemVerilogPort.SystemVerilogPort = None
    adder_input: list[SystemVerilogPort.SystemVerilogPort] = list[SystemVerilogPort.SystemVerilogPort]()
    adder_output: SystemVerilogPort.SystemVerilogPort = None

    def __init__(self, path: str, name: str):
        super().__init__(path, name)

    def generate(self):
        self.parameter_alu_input_width = self.add_parameter(self.parameter_alu_input_width)
        set_parameter_value(self.parameter_control_signal_input_width, str(self.configuration_m_number_of_digital_states))
        self.parameter_control_signal_input_width = self.add_parameter(self.parameter_control_signal_input_width)

        self.clk = self.add_port("clk", SystemVerilogPortDirection.Input(), SystemVerilogPortType.NoType(), -1, -1)
        self.rst = self.add_port("rst", SystemVerilogPortDirection.Input(), SystemVerilogPortType.NoType(), -1, -1)

        self.control_signal_sample_input = self.add_port("control_signal_sample_input", SystemVerilogPortDirection.Input(), SystemVerilogPortType.NoType(), int(get_parameter_value(self.parameter_control_signal_input_width)) - 1, 0)
        self.signal_estimation_output = self.add_port("signal_estimation_output", SystemVerilogPortDirection.Output(), SystemVerilogPortType.Logic(), int(get_parameter_value(self.parameter_control_signal_input_width)) - 1, 0)
        self.filter_coefficient_shift_register_enable = self.add_port("filter_coefficient_shift_register_enable", SystemVerilogPortDirection.Input(), SystemVerilogPortType.NoType(), -1, -1)

        simple_adder: DigitalEstimatorModules.SimpleAdder = DigitalEstimatorModules.SimpleAdder.SimpleAdder(self.path, "SimpleAdder")
        simple_adder.generate()
        simple_adder_port_connections: dict[SystemVerilogPort.SystemVerilogPort, SystemVerilogPort.SystemVerilogPort] = dict[SystemVerilogPort.SystemVerilogPort, SystemVerilogPort.SystemVerilogPort]()
        simple_adder_port_connections[simple_adder.clk] = self.clk
        simple_adder_port_connections[simple_adder.rst] = self.rst
        adder_input_0: SystemVerilogPort.SystemVerilogPort = self.add_port("adder_input_0", SystemVerilogPortDirection.Input(), SystemVerilogPortType.NoType(), int(get_parameter_value(self.parameter_alu_input_width)) - 1, 0)
        self.adder_input.append(adder_input_0)
        adder_input_1: SystemVerilogPort.SystemVerilogPort = self.add_port("adder_input_1", SystemVerilogPortDirection.Input(), SystemVerilogPortType.NoType(), int(get_parameter_value(self.parameter_alu_input_width)) - 1, 0)
        self.adder_input.append(adder_input_1)
        simple_adder_port_connections.update(connect_port_array(simple_adder.alu_input, self.adder_input))
        self.adder_output = self.add_port("adder_output", SystemVerilogPortDirection.Output(), SystemVerilogPortType.NoType(), int(get_parameter_value(self.parameter_alu_input_width)) - 1, 0)
        simple_adder_port_connections[simple_adder.alu_output] = self.adder_output

        self.add_submodule(simple_adder, simple_adder_port_connections)
        high_level_simulation: CBADC_HighLevelSimulation.DigitalEstimatorParameterGenerator = CBADC_HighLevelSimulation.DigitalEstimatorParameterGenerator(
            n_number_of_analog_states = self.configuration_n_number_of_analog_states,
            m_number_of_digital_states = self.configuration_m_number_of_digital_states,
            k1 = self.configuration_lookback_length,
            k2 = self.configuration_lookahead_length,
            beta = self.configuration_beta,
            eta2 = self.configuration_eta2,
            kappa = self.configuration_kappa,
            rho = self.configuration_rho
        )
        digital_estimator_fir: cbadc.digital_estimator.FIRFilter = high_level_simulation.simulate_digital_estimator_fir()
        print("hb:\n\n", high_level_simulation.get_fir_lookback_coefficient_matrix())
        print("hf:\n\n", high_level_simulation.get_fir_lookahead_coefficient_matrix())

        self.syntax_generator.timescale()
        self.syntax_generator.module_head(self.name, self.parameter_list, self.port_list)

        fir_lookback_coefficient_matrix_dimensions: list[SystemVerilogDimension.SystemVerilogDimension] = list[SystemVerilogDimension.SystemVerilogDimension]()
        fir_lookback_coefficient_matrix_rows: SystemVerilogDimension.SystemVerilogDimension = SystemVerilogDimension.SystemVerilogDimension(self.configuration_lookback_length - 1, 0)
        fir_lookback_coefficient_matrix_columns: SystemVerilogDimension.SystemVerilogDimension = SystemVerilogDimension.SystemVerilogDimension(self.configuration_m_number_of_digital_states - 1, 0)
        fir_lookback_coefficient_matrix_dimensions.append(fir_lookback_coefficient_matrix_rows)
        fir_lookback_coefficient_matrix_dimensions.append(fir_lookback_coefficient_matrix_columns)
        fir_lookback_coefficient_matrix = self.syntax_generator.local_parameter("fir_lookback_coefficient_matrix", SystemVerilogPortType.Logic(), SystemVerilogSignalSign.Signed(), 63, 0, fir_lookback_coefficient_matrix_dimensions, ndarray_to_system_verilog_array(high_level_simulation.get_fir_lookback_coefficient_matrix()))
        self.syntax_generator.blank_line()
        fir_lookahead_coefficient_matrix_dimensions: list[SystemVerilogDimension.SystemVerilogDimension] = list[SystemVerilogDimension.SystemVerilogDimension]()
        fir_lookahead_coefficient_matrix_rows: SystemVerilogDimension.SystemVerilogDimension = SystemVerilogDimension.SystemVerilogDimension(self.configuration_lookahead_length - 1, 0)
        fir_lookahead_coefficient_matrix_columns: SystemVerilogDimension.SystemVerilogDimension = SystemVerilogDimension.SystemVerilogDimension(self.configuration_m_number_of_digital_states - 1, 0)
        fir_lookahead_coefficient_matrix_dimensions.append(fir_lookahead_coefficient_matrix_rows)
        fir_lookahead_coefficient_matrix_dimensions.append(fir_lookahead_coefficient_matrix_columns)
        fir_lookahead_coefficient_matrix = self.syntax_generator.local_parameter("fir_lookahead_coefficient_matrix", SystemVerilogPortType.Logic(), SystemVerilogSignalSign.Signed(), 63, 0, fir_lookahead_coefficient_matrix_dimensions, ndarray_to_system_verilog_array(high_level_simulation.get_fir_lookahead_coefficient_matrix()))

        self.syntax_generator.instantiate_submodules(self.submodules)
        self.syntax_generator.end_module()
        self.syntax_generator.close()
        