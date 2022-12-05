import random

import numpy

import SystemVerilogModule
import DigitalEstimatorModules.DigitalEstimatorWrapper
import SystemVerilogPort
import SystemVerilogSignal
import SystemVerilogPortType
from SystemVerilogSignalSign import SystemVerilogSignalSign
from SystemVerilogSyntaxGenerator import SystemVerilogSyntaxGenerator, get_parameter_value, set_parameter_value
from SystemVerilogSyntaxGenerator import set_parameter_value_by_parameter
from SystemVerilogSyntaxGenerator import connect_port_array
from SystemVerilogSyntaxGenerator import decimal_number
from SystemVerilogSyntaxGenerator import ndarray_to_system_verilog_array
from SystemVerilogComparisonOperator import *
import CBADC_HighLevelSimulation


class DigitalEstimatorTestbench(SystemVerilogModule.SystemVerilogModule):
    configuration_number_of_timesteps_in_clock_cycle: int = 10
    configuration_n_number_of_analog_states: int = 6
    configuration_m_number_of_digital_states: int = configuration_n_number_of_analog_states
    configuration_beta: float = 6250.0
    configuration_rho: float = -1e-2
    configuration_kappa: float = -1.0
    configuration_eta2: float = 1e7
    configuration_lookback_length: int = 5
    configuration_lookahead_length: int = 1
    configuration_fir_data_width: int = 64
    configuration_fir_lut_input_width: int = 4
    configuration_simulation_length: int = 2 << 12
    configuration_offset: float = 0.0
    configuration_down_sample_rate: int = 1
    configuration_over_sample_rate: int = 25
    configuration_downsample_clock_counter_type: str = "binary"

    high_level_simulation: CBADC_HighLevelSimulation.DigitalEstimatorParameterGenerator

    parameter_alu_input_width: dict[str, str] = {"ALU_INPUT_WIDTH": "32"}
    parameter_control_signal_input_width: dict[str, str] = {"CONTROL_SIGNAL_INPUT_WIDTH": str(configuration_m_number_of_digital_states)}

    def __init__(self, path: str, name: str):
        super().__init__(path, name)

    """def generate(self):
        self.parameter_alu_input_width = self.add_parameter(self.parameter_alu_input_width)
        set_parameter_value(self.parameter_control_signal_input_width, str(self.configuration_m_number_of_digital_states))
        self.parameter_control_signal_input_width = self.add_parameter(self.parameter_control_signal_input_width)

        self.syntax_generator.timescale()
        self.syntax_generator.module_head(self.name)
        clk: SystemVerilogSignal.SystemVerilogSignal = self.add_signal("clk", SystemVerilogPortType.Logic())
        rst: SystemVerilogSignal.SystemVerilogSignal = self.add_signal("rst", SystemVerilogPortType.Logic())

        self.syntax_generator.blank_line()

        digital_estimator_control_signal_sample_input: SystemVerilogSignal.SystemVerilogSignal = self.add_signal("digital_estimator_control_signal_sample_input", SystemVerilogPortType.Logic(), int(get_parameter_value(self.parameter_control_signal_input_width)) - 1, 0)
        digital_estimator_signal_estimation_output: SystemVerilogSignal.SystemVerilogSignal = self.add_signal("digital_estimator_signal_estimation_output", SystemVerilogPortType.Logic(), int(get_parameter_value(self.parameter_control_signal_input_width)) - 1, 0)
        digital_estimator_filter_coefficient_shift_register_enable: SystemVerilogSignal.SystemVerilogSignal = self.add_signal("digital_estimator_filter_coefficient_shift_register_enable", SystemVerilogPortType.Logic(), -1, -1)
        self.syntax_generator.blank_line()

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
        self.syntax_generator.blank_line()
        self.syntax_generator.assign_construct(digital_estimator_control_signal_sample_input, decimal_number(0))
        self.syntax_generator.assign_construct(digital_estimator_filter_coefficient_shift_register_enable, decimal_number(0))
        self.syntax_generator.blank_line()
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
        dut.configuration_rho = self.configuration_rho
        dut.configuration_beta = self.configuration_beta
        dut.configuration_eta2 = self.configuration_eta2
        dut.configuration_kappa = self.configuration_kappa
        dut.configuration_lookback_length = self.configuration_lookback_length
        dut.configuration_lookahead_length = self.configuration_lookahead_length
        dut.configuration_n_number_of_analog_states = self.configuration_n_number_of_analog_states
        dut.configuration_m_number_of_digital_states = self.configuration_m_number_of_digital_states
        set_parameter_value_by_parameter(self.parameter_alu_input_width, dut.parameter_alu_input_width)
        dut.generate()
        dut_port_connections: dict[SystemVerilogPort.SystemVerilogPort, SystemVerilogPort.SystemVerilogPort] = dict[SystemVerilogPort.SystemVerilogPort, SystemVerilogPort.SystemVerilogPort]()
        dut_port_connections[dut.clk] = clk
        dut_port_connections[dut.rst] = rst
        dut_port_connections[dut.control_signal_sample_input] = digital_estimator_control_signal_sample_input
        dut_port_connections[dut.signal_estimation_output] = digital_estimator_signal_estimation_output
        dut_port_connections[dut.filter_coefficient_shift_register_enable] = digital_estimator_filter_coefficient_shift_register_enable
        dut_port_connections.update(connect_port_array(dut.adder_input, alu_inputs))
        dut_port_connections[dut.adder_output] = alu_output

        self.add_submodule(dut, dut_port_connections)

        self.syntax_generator.instantiate_submodules(self.submodules)

        self.syntax_generator.blank_line()
        self.syntax_generator.end_module()
        self.syntax_generator.close()"""

    def generate(self):
        #high_level_simulation: CBADC_HighLevelSimulation.DigitalEstimatorParameterGenerator = CBADC_HighLevelSimulation.DigitalEstimatorParameterGenerator(
        #    self.path,
        #    self.configuration_n_number_of_analog_states,
        #    self.configuration_m_number_of_digital_states,
        #    self.configuration_beta,
        #    self.configuration_rho,
        #    self.configuration_kappa,
        #    self.configuration_eta2,
        #    self.configuration_lookback_length,
        #    self.configuration_lookahead_length,
        #    self.configuration_fir_data_width,
        #    self.configuration_down_sample_rate,
        #    offset = self.configuration_offset,
        #    OSR = self.configuration_over_sample_rate,
        #    size = self.configuration_simulation_length
        #)
        self.high_level_simulation.simulate_digital_estimator_fir()

        content: str = f"""module DigitalEstimatorTestbench #(
    parameter CLOCK_PERIOD = 10,
    parameter CLOCK_HALF_PERIOD = $ceil(CLOCK_PERIOD / 2.0),
    parameter DOWN_SAMPLE_RATE = {self.configuration_down_sample_rate},

    parameter N_NUMBER_ANALOG_STATES = {self.configuration_n_number_of_analog_states},
    parameter M_NUMBER_DIGITAL_STATES = {self.configuration_m_number_of_digital_states},
    parameter LOOKAHEAD_SIZE = {self.configuration_lookahead_length},
    parameter LOOKBACK_SIZE = {self.configuration_lookback_length},
    localparam TOTAL_LOOKUP_REGISTER_LENGTH = LOOKAHEAD_SIZE + LOOKBACK_SIZE,
    parameter OUTPUT_DATA_WIDTH = {self.configuration_fir_data_width},

    parameter INPUT_WIDTH = {self.configuration_fir_lut_input_width},

    localparam LOOKBACK_LOOKUP_TABLE_ENTRY_COUNT = (LOOKBACK_SIZE * M_NUMBER_DIGITAL_STATES) / INPUT_WIDTH * (2**INPUT_WIDTH) + (((LOOKBACK_SIZE * M_NUMBER_DIGITAL_STATES) % INPUT_WIDTH) == 0 ? 0 : (2**((LOOKBACK_SIZE * M_NUMBER_DIGITAL_STATES) % INPUT_WIDTH))),
    localparam LOOKAHEAD_LOOKUP_TABLE_ENTRY_COUNT = (LOOKAHEAD_SIZE * M_NUMBER_DIGITAL_STATES) / INPUT_WIDTH * (2**INPUT_WIDTH) + (((LOOKAHEAD_SIZE * M_NUMBER_DIGITAL_STATES) % INPUT_WIDTH) == 0 ? 0 : (2**((LOOKAHEAD_SIZE * M_NUMBER_DIGITAL_STATES) % INPUT_WIDTH)))
);"""
        content += """

    logic rst;
    logic clk;
    logic enable_lookup_table_coefficient_shift_in;
    logic [OUTPUT_DATA_WIDTH - 1 : 0] lookup_table_coefficient;
    logic [M_NUMBER_DIGITAL_STATES - 1 : 0] digital_control_input;
    //logic [(LOOKBACK_LOOKUP_TABLE_ENTRY_COUNT * OUTPUT_DATA_WIDTH) - 1 : 0] lookback_lookup_table_entries;
    logic [LOOKBACK_LOOKUP_TABLE_ENTRY_COUNT - 1 : 0][OUTPUT_DATA_WIDTH - 1 : 0] lookback_lookup_table_entries;
    //logic [(LOOKAHEAD_LOOKUP_TABLE_ENTRY_COUNT * OUTPUT_DATA_WIDTH) - 1 : 0] lookahead_lookup_table_entries;
    logic [LOOKAHEAD_LOOKUP_TABLE_ENTRY_COUNT - 1 : 0][OUTPUT_DATA_WIDTH - 1 : 0] lookahead_lookup_table_entries;
    wire signed [OUTPUT_DATA_WIDTH - 1 : 0] signal_estimation_output;
    wire signal_estimation_valid_out;


    initial begin
        rst = 1'b1;
        #(2 * CLOCK_PERIOD);
        rst = 1'b0;
        #CLOCK_HALF_PERIOD;
    end

    always begin
        clk = 1'b0;
        #CLOCK_HALF_PERIOD;
        clk = 1'b1;
        #CLOCK_HALF_PERIOD;
    end

    initial begin
        enable_lookup_table_coefficient_shift_in = 1'b0;
        lookup_table_coefficient = {OUTPUT_DATA_WIDTH{1'b0}};

        @(negedge rst);

        enable_lookup_table_coefficient_shift_in = 1'b1;

        $info("Starting coefficient shift in.\\n");

        for(integer lookahead_index = LOOKAHEAD_LOOKUP_TABLE_ENTRY_COUNT - 1; lookahead_index >= 0; lookahead_index--) begin
        //for(integer lookahead_index = 0; lookahead_index <= LOOKAHEAD_LOOKUP_TABLE_ENTRY_COUNT - 1; lookahead_index++) begin
            lookup_table_coefficient = lookahead_lookup_table_entries[lookahead_index];
            @(negedge clk);
        end
        
        for(integer lookback_index = LOOKBACK_LOOKUP_TABLE_ENTRY_COUNT - 1; lookback_index >= 0; lookback_index--) begin
        //for(integer lookback_index = 0; lookback_index <= LOOKBACK_LOOKUP_TABLE_ENTRY_COUNT - 1; lookback_index++) begin
            lookup_table_coefficient = lookback_lookup_table_entries[lookback_index];
            @(negedge clk);
        end

        enable_lookup_table_coefficient_shift_in = 1'b0;

        $info("Coefficient shift in finished.\\n");
    end

    initial begin
        static int control_signal_file = $fopen("./control_signal.csv", "r");
        if(control_signal_file == 0) begin
            $error("Control signal input file could not be opened!");
            $finish;
        end

        digital_control_input = {N_NUMBER_ANALOG_STATES{1'b0}};

        @(negedge enable_lookup_table_coefficient_shift_in);
        """
        if self.configuration_down_sample_rate == 1:
            content += """@(posedge clk);
        
        """
        else:
            content += """@(posedge dut_digital_estimator.clk_sample_shift_register);
        repeat(DOWN_SAMPLE_RATE + 1) begin
            @(posedge clk);
        end
        
        """
        content += """while($fscanf(control_signal_file, "%b,\\n", digital_control_input) > 0) begin
            @(posedge clk);
        end

        $fclose(control_signal_file);

        $finish(0);
    end

    initial begin
        static int digital_estimation_output_file = $fopen("./digital_estimation.csv", "w");
        if(digital_estimation_output_file == 0) begin
            $error("Digital estimation output file could not be opened!");
            $finish;
        end

        @(negedge enable_lookup_table_coefficient_shift_in);
        """
        if self.configuration_down_sample_rate == 1:
            content += """repeat(3) begin
            @(posedge clk);
        end

        """
        else:
            content += """repeat(4) begin
            @(negedge dut_digital_estimator.clk_sample_shift_register);
        end
        
        """
        content += """/*repeat((DOWN_SAMPLE_RATE == 1) ? 1 : int'($ceil(real'(DOWN_SAMPLE_RATE) / 2.0) - 1)) begin
            @(posedge clk);
        end*/

        forever begin
            //$fwrite(digital_estimation_output_file, "%d, %d, %d\\n", signal_estimation_output, dut_digital_estimator.adder_block_lookback_result, dut_digital_estimator.adder_block_lookahead_result);
            """
        if self.configuration_fir_data_width < 32:
            content += """$fwrite(digital_estimation_output_file, "%0.18f, %0.18f, %0.18f\\n", real'(signal_estimation_output) / (2**(OUTPUT_DATA_WIDTH - 1)), real'(dut_digital_estimator.adder_block_lookback_result) / (2**(OUTPUT_DATA_WIDTH - 1)), real'(dut_digital_estimator.adder_block_lookahead_result) / (2**(OUTPUT_DATA_WIDTH - 1)));
            """
        else:
            content += """$fwrite(digital_estimation_output_file, "%0.18f, %0.18f, %0.18f\\n", real'(signal_estimation_output >>> (OUTPUT_DATA_WIDTH - 31)) / (2**(30)), real'(dut_digital_estimator.adder_block_lookback_result >>> (OUTPUT_DATA_WIDTH - 31)) / (2**(30)), real'(dut_digital_estimator.adder_block_lookahead_result >>> (OUTPUT_DATA_WIDTH - 31)) / (2**(30)));
            """
        if self.configuration_down_sample_rate == 1:
            content += """
            @(posedge clk);
        """
        else:
            content += """
            @(negedge dut_digital_estimator.clk_sample_shift_register);
        """
        content += """end
    end

    initial begin
        lookback_lookup_table_entries = """
        content += ndarray_to_system_verilog_array(numpy.array(CBADC_HighLevelSimulation.convert_coefficient_matrix_to_lut_entries(self.high_level_simulation.get_fir_lookback_coefficient_matrix(), self.configuration_fir_lut_input_width))) + ";\n\n\t\t"
        content += "lookahead_lookup_table_entries = "
        content += ndarray_to_system_verilog_array(numpy.array(CBADC_HighLevelSimulation.convert_coefficient_matrix_to_lut_entries(self.high_level_simulation.get_fir_lookahead_coefficient_matrix(), self.configuration_fir_lut_input_width))) + ";\n"
        content += f"""\tend

    DigitalEstimator #(
            .N_NUMBER_ANALOG_STATES(N_NUMBER_ANALOG_STATES),
            .M_NUMBER_DIGITAL_STATES(M_NUMBER_DIGITAL_STATES),
            .LOOKBACK_SIZE(LOOKBACK_SIZE),
            .LOOKAHEAD_SIZE(LOOKAHEAD_SIZE),
            .LOOKUP_TABLE_INPUT_WIDTH(INPUT_WIDTH),
            .LOOKUP_TABLE_DATA_WIDTH(OUTPUT_DATA_WIDTH),
            .OUTPUT_DATA_WIDTH(OUTPUT_DATA_WIDTH),
            .DOWN_SAMPLE_RATE({self.configuration_down_sample_rate})
        )
        dut_digital_estimator (
            .rst(rst),
            .clk(clk),
            .enable_lookup_table_coefficient_shift_in(enable_lookup_table_coefficient_shift_in),
            .lookup_table_coefficient(lookup_table_coefficient),
            .digital_control_input(digital_control_input),
            .signal_estimation_valid_out(signal_estimation_valid_out),
            .signal_estimation_output(signal_estimation_output)
    );


    bind AdderCombinatorial AdderCombinatorialAssertions #(
            .INPUT_WIDTH(INPUT_WIDTH)
        )
        adder_combinatorial_bind (
            .rst(rst),
            .input_0(input_0),
            .input_1(input_1),
            .out(out)
    );

    bind AdderBlockCombinatorial AdderBlockCombinatorialAssertions #(
            .INPUT_COUNT(INPUT_COUNT),
            .INPUT_WIDTH(INPUT_WIDTH)
        )
        adder_block_combinatorial_bind (
            .rst(rst),
            .in(in),
            .out(out)
    );

    bind LookUpTable LookUpTableAssertions #(
            .INPUT_WIDTH(INPUT_WIDTH),
            .DATA_WIDTH(DATA_WIDTH)
        )
        look_up_table_bind (
            .rst(rst),
            .in(in),
            .memory(memory),
            .out(out)
    );

    bind LookUpTableBlock LookUpTableBlockAssertions #(
            .TOTAL_INPUT_WIDTH(TOTAL_INPUT_WIDTH),
            .LOOKUP_TABLE_INPUT_WIDTH(LOOKUP_TABLE_INPUT_WIDTH),
            .LOOKUP_TABLE_DATA_WIDTH(LOOKUP_TABLE_DATA_WIDTH)
        )
        look_up_table_block_bind (
            .rst(rst),
            .input_register(input_register),
            .lookup_table_entries(lookup_table_entries),
            .lookup_table_results(lookup_table_results)
    );
    
"""
        if self.configuration_down_sample_rate > 1:
            content += """\tbind ClockDivider ClockDividerAssertions #(
            .DOWN_SAMPLE_RATE(DOWN_SAMPLE_RATE)
        )
        clock_divider_bind (
            .rst(rst),
            .clk(clk),
            .clk_downsample(clk_downsample),
            .edge_counter(edge_counter),
            .clock_divider_counter(clock_divider_counter)
    );

    bind InputDownsampleAccumulateRegister InputDownsampleAccumulateRegisterAssertions #(
            .REGISTER_LENGTH(REGISTER_LENGTH),
            .DATA_WIDTH(DATA_WIDTH)
        )
        input_downsample_accumulate_register_bind (
            .rst(rst),
            .clk(clk),
            .in(in),
            .out(out)
    );
    
"""
            if self.configuration_downsample_clock_counter_type == "gray":
                content += """\tbind GrayCodeToBinary GrayCodeToBinaryAssertions #(
            .BIT_SIZE(BIT_SIZE)
        )
        gray_code_to_binary_bind (
            .rst(rst),
            .gray_code(gray_code),
            .binary(binary)
    );

    bind GrayCounter GrayCounterAssertions #(
            .BIT_SIZE(BIT_SIZE),
            .TOP_VALUE(TOP_VALUE)
        )
        gray_counter_bind (
            .rst(rst),
            .clk(clk),
            .counter(counter)
    );
    
"""
        content += "\nendmodule"

        self.syntax_generator.single_line_no_linebreak(content, indentation = 0)
        self.syntax_generator.close()
