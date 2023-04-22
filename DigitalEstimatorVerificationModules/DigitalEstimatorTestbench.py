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
from SystemVerilogSyntaxGenerator import ndarray_to_system_verilog_concatenation
from SystemVerilogComparisonOperator import *
import CBADC_HighLevelSimulation


class DigitalEstimatorTestbench(SystemVerilogModule.SystemVerilogModule):
    top_module_name: str = "DigitalEstimator"
    configuration_number_of_timesteps_in_clock_cycle: int = 10
    configuration_n_number_of_analog_states: int = 6
    configuration_m_number_of_digital_states: int = configuration_n_number_of_analog_states
    #configuration_beta: float = 6250.0
    #configuration_rho: float = -1e-2
    #configuration_kappa: float = -1.0
    #configuration_eta2: float = 1e7
    configuration_lookback_length: int = 5
    configuration_lookahead_length: int = 1
    configuration_fir_data_width: int = 64
    configuration_fir_lut_input_width: int = 4
    configuration_simulation_length: int = 2 << 12
    #configuration_offset: float = 0.0
    configuration_down_sample_rate: int = 1
    configuration_over_sample_rate: int = 25
    configuration_downsample_clock_counter_type: str = "binary"
    configuration_combinatorial_synchronous: str = "combinatorial"
    configuration_coefficients_variable_fixed: str = "variable"
    configuration_mapped_simulation: bool = False
    configuration_synthesis_program: str = "genus"
    configuration_reduce_size: bool = False

    high_level_simulation: CBADC_HighLevelSimulation.DigitalEstimatorParameterGenerator

    parameter_alu_input_width: dict[str, str] = {"ALU_INPUT_WIDTH": "32"}
    parameter_control_signal_input_width: dict[str, str] = {"CONTROL_SIGNAL_INPUT_WIDTH": str(configuration_m_number_of_digital_states)}

    def __init__(self, path: str, name: str):
        super().__init__(path, name)

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
        
        if self.configuration_reduce_size == True:
            lookback_lut_entries = CBADC_HighLevelSimulation.convert_coefficient_matrix_to_lut_entries(self.high_level_simulation.fir_hb_matrix, self.configuration_fir_lut_input_width)
            lookback_lut_entries_mapped = CBADC_HighLevelSimulation.map_lut_entries_to_luts(lut_entries = lookback_lut_entries, lut_input_width = self.configuration_fir_lut_input_width)
            lookback_lut_entries_bit_mapping: tuple[list[list[int]], int] = CBADC_HighLevelSimulation.get_lut_entry_bit_mapping(lut_entry_matrix = lookback_lut_entries, lut_input_width = self.configuration_fir_lut_input_width)
            lookback_lut_entries_max_widths = CBADC_HighLevelSimulation.get_maximum_bitwidth_per_lut(lookback_lut_entries_bit_mapping[0])
            lookback_lut_entries_max_widths_sorted: list[tuple[int, int]] = CBADC_HighLevelSimulation.sort_luts_by_size(lookback_lut_entries_max_widths)
            lookback_lut_entries_mapped_reordered: list[list[int]] = CBADC_HighLevelSimulation.reorder_lut_entries(lookback_lut_entries_mapped, lookback_lut_entries_max_widths_sorted)
            lookahead_lut_entries = CBADC_HighLevelSimulation.convert_coefficient_matrix_to_lut_entries(self.high_level_simulation.fir_hf_matrix, self.configuration_fir_lut_input_width)
            lookahead_lut_entries_mapped = CBADC_HighLevelSimulation.map_lut_entries_to_luts(lut_entries = lookahead_lut_entries, lut_input_width = self.configuration_fir_lut_input_width)
            lookahead_lut_entries_bit_mapping: tuple[list[list[int]], int] = CBADC_HighLevelSimulation.get_lut_entry_bit_mapping(lut_entry_matrix = lookahead_lut_entries, lut_input_width = self.configuration_fir_lut_input_width)
            lookahead_lut_entries_max_widths = CBADC_HighLevelSimulation.get_maximum_bitwidth_per_lut(lookahead_lut_entries_bit_mapping[0])
            lookahead_lut_entries_max_widths_sorted: list[tuple[int, int]] = CBADC_HighLevelSimulation.sort_luts_by_size(lookahead_lut_entries_max_widths)
            lookahead_lut_entries_mapped_reordered: list[list[int]] = CBADC_HighLevelSimulation.reorder_lut_entries(lookahead_lut_entries_mapped, lookahead_lut_entries_max_widths_sorted)
            maximum_data_width: int = 0
            for entry in lookback_lut_entries_max_widths_sorted:
                if maximum_data_width < entry[0]:
                    maximum_data_width = entry[0]
            for entry in lookahead_lut_entries_max_widths_sorted:
                if maximum_data_width < entry[0]:
                    maximum_data_width = entry[0]

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
    localparam LOOKAHEAD_LOOKUP_TABLE_ENTRY_COUNT = (LOOKAHEAD_SIZE * M_NUMBER_DIGITAL_STATES) / INPUT_WIDTH * (2**INPUT_WIDTH) + (((LOOKAHEAD_SIZE * M_NUMBER_DIGITAL_STATES) % INPUT_WIDTH) == 0 ? 0 : (2**((LOOKAHEAD_SIZE * M_NUMBER_DIGITAL_STATES) % INPUT_WIDTH)))"""
        if self.configuration_combinatorial_synchronous == "combinatorial":
            content += """
);"""
        elif self.configuration_combinatorial_synchronous == "synchronous":
            content += """,
    
    localparam LOOKBACK_LUT_COUNT = int'($ceil(real'(LOOKBACK_SIZE) * real'(M_NUMBER_DIGITAL_STATES) / real'(INPUT_WIDTH))),
    localparam REGISTER_DELAY = 1 + $clog2(LOOKBACK_LUT_COUNT) + 1
);"""
        content += """

    logic rst;
    logic clk;
    """
        if self.configuration_coefficients_variable_fixed == "variable":
            content += """logic enable_lookup_table_coefficient_shift_in;
    """
            if self.configuration_reduce_size == True:
                content += f"""logic lookup_table_coefficient_shift_in_lookback_lookahead_switch;
    logic [{maximum_data_width} - 1 : 0] lookup_table_coefficient;
    """
            elif self.configuration_reduce_size == False:
                content += """logic [OUTPUT_DATA_WIDTH - 1 : 0] lookup_table_coefficient;
    """
            content += """//logic [(LOOKBACK_LOOKUP_TABLE_ENTRY_COUNT * OUTPUT_DATA_WIDTH) - 1 : 0] lookback_lookup_table_entries;
    logic [LOOKBACK_LOOKUP_TABLE_ENTRY_COUNT - 1 : 0][OUTPUT_DATA_WIDTH - 1 : 0] lookback_lookup_table_entries;
    //logic [(LOOKAHEAD_LOOKUP_TABLE_ENTRY_COUNT * OUTPUT_DATA_WIDTH) - 1 : 0] lookahead_lookup_table_entries;
    logic [LOOKAHEAD_LOOKUP_TABLE_ENTRY_COUNT - 1 : 0][OUTPUT_DATA_WIDTH - 1 : 0] lookahead_lookup_table_entries;
    """
        content += """logic [M_NUMBER_DIGITAL_STATES - 1 : 0] digital_control_input;
    wire signed [OUTPUT_DATA_WIDTH - 1 : 0] signal_estimation_output;
    wire signal_estimation_valid_out;
    wire clk_sample_shift_register;


    initial begin
        $assertoff;
    end

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

    """
    
        if self.configuration_coefficients_variable_fixed == "variable":
            content += """initial begin
        enable_lookup_table_coefficient_shift_in = 1'b0;
        """
            if self.configuration_reduce_size == True:
                content += """lookup_table_coefficient_shift_in_lookback_lookahead_switch = 1'b0;
        """
            content += """lookup_table_coefficient = {OUTPUT_DATA_WIDTH{1'b0}};

        @(negedge rst);

        enable_lookup_table_coefficient_shift_in = 1'b1;
        
        """
            if self.configuration_reduce_size == True:
                content += """lookup_table_coefficient_shift_in_lookback_lookahead_switch = 1'b1;
                
        """
            content += """$info("Starting coefficient shift in.\\n");

        for(integer lookahead_index = LOOKAHEAD_LOOKUP_TABLE_ENTRY_COUNT - 1; lookahead_index >= 0; lookahead_index--) begin
        //for(integer lookahead_index = 0; lookahead_index <= LOOKAHEAD_LOOKUP_TABLE_ENTRY_COUNT - 1; lookahead_index++) begin
            lookup_table_coefficient = lookahead_lookup_table_entries[lookahead_index];
            @(negedge clk);
        end
        
        """
            if self.configuration_reduce_size == True:
                content += """lookup_table_coefficient_shift_in_lookback_lookahead_switch = 1'b0;
                
        """
            content += """for(integer lookback_index = LOOKBACK_LOOKUP_TABLE_ENTRY_COUNT - 1; lookback_index >= 0; lookback_index--) begin
        //for(integer lookback_index = 0; lookback_index <= LOOKBACK_LOOKUP_TABLE_ENTRY_COUNT - 1; lookback_index++) begin
            lookup_table_coefficient = lookback_lookup_table_entries[lookback_index];
            @(negedge clk);
        end

        enable_lookup_table_coefficient_shift_in = 1'b0;

        $info("Coefficient shift in finished.\\n");
    end

    """
    
        content += """initial begin
        static int control_signal_file = $fopen("./control_signal.csv", "r");
        if(control_signal_file == 0) begin
            $error("Control signal input file could not be opened!");
            $finish;
        end

        digital_control_input = {N_NUMBER_ANALOG_STATES{1'b0}};

        """
        if self.configuration_coefficients_variable_fixed == "variable":
            content += """@(negedge enable_lookup_table_coefficient_shift_in);
        """
        elif self.configuration_coefficients_variable_fixed == "fixed":
            content += """@(negedge rst);
        """
        if self.configuration_down_sample_rate == 1:
            content += """@(posedge clk);
        
        """
        else:
            content += """@(posedge clk_sample_shift_register);
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
        """
        if self.configuration_mapped_simulation == True:
            content += f"""static int digital_estimation_output_file = $fopen("./digital_estimation_mapped_{self.configuration_synthesis_program}.csv", "w");"""
        else:
            content += """static int digital_estimation_output_file = $fopen("./digital_estimation.csv", "w");
        """
        content += """if(digital_estimation_output_file == 0) begin
            $error("Digital estimation output file could not be opened!");
            $finish;
        end
        
        """
        if self.configuration_reduce_size == True:
            content += """$fwrite(digital_estimation_output_file, "0.0, 0.0, 0.0\\n");
        """
        content += """$fwrite(digital_estimation_output_file, "0.0, 0.0, 0.0\\n");

        """
        if self.configuration_coefficients_variable_fixed == "variable":
            content += """@(negedge enable_lookup_table_coefficient_shift_in);
        """
        elif self.configuration_coefficients_variable_fixed == "fixed":
            content += """@(negedge rst);
        """
        if self.configuration_down_sample_rate == 1:
            content += """repeat(3) begin
            @(posedge clk);
        end

        """
        else:
            content += """repeat(4) begin
            @(negedge clk_sample_shift_register);
        end
        
        """

        if self.configuration_combinatorial_synchronous == "synchronous":
            content += """repeat(REGISTER_DELAY) begin
            @(negedge clk_sample_shift_register);
        end
        
        """

        content += """forever begin
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
            @(negedge clk_sample_shift_register);
        """
        content += """end
    end

    """
        
        if self.configuration_coefficients_variable_fixed == "variable":
            if self.configuration_reduce_size == False:
                content += """initial begin
        lookback_lookup_table_entries = """
                content += ndarray_to_system_verilog_array(numpy.array(CBADC_HighLevelSimulation.convert_coefficient_matrix_to_lut_entries(self.high_level_simulation.get_fir_lookback_coefficient_matrix(), self.configuration_fir_lut_input_width))) + ";\n\t\t"
                content += "lookahead_lookup_table_entries = "
                content += ndarray_to_system_verilog_array(numpy.array(CBADC_HighLevelSimulation.convert_coefficient_matrix_to_lut_entries(self.high_level_simulation.get_fir_lookahead_coefficient_matrix(), self.configuration_fir_lut_input_width))) + ";\n"
                content += f"""\tend
    
    """
            elif self.configuration_reduce_size == True:
                lookback_lut_entries = CBADC_HighLevelSimulation.convert_coefficient_matrix_to_lut_entries(self.high_level_simulation.fir_hb_matrix, self.configuration_fir_lut_input_width)
                lookback_lut_entries_mapped = CBADC_HighLevelSimulation.map_lut_entries_to_luts(lut_entries = lookback_lut_entries, lut_input_width = self.configuration_fir_lut_input_width)
                lookback_lut_entries_bit_mapping: tuple[list[list[int]], int] = CBADC_HighLevelSimulation.get_lut_entry_bit_mapping(lut_entry_matrix = lookback_lut_entries, lut_input_width = self.configuration_fir_lut_input_width)
                lookback_lut_entries_max_widths = CBADC_HighLevelSimulation.get_maximum_bitwidth_per_lut(lookback_lut_entries_bit_mapping[0])
                lookback_lut_entries_max_widths_sorted: list[tuple[int, int]] = CBADC_HighLevelSimulation.sort_luts_by_size(lookback_lut_entries_max_widths)
                lookback_lut_entries_mapped_reordered: list[list[int]] = CBADC_HighLevelSimulation.reorder_lut_entries(lookback_lut_entries_mapped, lookback_lut_entries_max_widths_sorted)
                lookahead_lut_entries = CBADC_HighLevelSimulation.convert_coefficient_matrix_to_lut_entries(self.high_level_simulation.fir_hf_matrix, self.configuration_fir_lut_input_width)
                lookahead_lut_entries_mapped = CBADC_HighLevelSimulation.map_lut_entries_to_luts(lut_entries = lookahead_lut_entries, lut_input_width = self.configuration_fir_lut_input_width)
                lookahead_lut_entries_bit_mapping: tuple[list[list[int]], int] = CBADC_HighLevelSimulation.get_lut_entry_bit_mapping(lut_entry_matrix = lookahead_lut_entries, lut_input_width = self.configuration_fir_lut_input_width)
                lookahead_lut_entries_max_widths = CBADC_HighLevelSimulation.get_maximum_bitwidth_per_lut(lookahead_lut_entries_bit_mapping[0])
                lookahead_lut_entries_max_widths_sorted: list[tuple[int, int]] = CBADC_HighLevelSimulation.sort_luts_by_size(lookahead_lut_entries_max_widths)
                lookahead_lut_entries_mapped_reordered: list[list[int]] = CBADC_HighLevelSimulation.reorder_lut_entries(lookahead_lut_entries_mapped, lookahead_lut_entries_max_widths_sorted)
                content += """initial begin
        lookback_lookup_table_entries = '"""
                content += ndarray_to_system_verilog_concatenation(lookback_lut_entries_mapped_reordered, lookback_lut_entries_max_widths_sorted) + ";\n\t\t"
                content += "lookahead_lookup_table_entries = '"
                content += ndarray_to_system_verilog_concatenation(lookahead_lut_entries_mapped_reordered, lookahead_lut_entries_max_widths_sorted) + ";\n"
                content += f"""\tend
    
    """

        content += f"""
    {self.top_module_name} """
        if self.configuration_mapped_simulation == False:
            content += """#(
            .N_NUMBER_ANALOG_STATES(N_NUMBER_ANALOG_STATES),
            .M_NUMBER_DIGITAL_STATES(M_NUMBER_DIGITAL_STATES),
            .LOOKBACK_SIZE(LOOKBACK_SIZE),
            .LOOKAHEAD_SIZE(LOOKAHEAD_SIZE),
            .LOOKUP_TABLE_INPUT_WIDTH(INPUT_WIDTH),
            .LOOKUP_TABLE_DATA_WIDTH(OUTPUT_DATA_WIDTH),
            .OUTPUT_DATA_WIDTH(OUTPUT_DATA_WIDTH),
            .DOWN_SAMPLE_RATE(DOWN_SAMPLE_RATE)
        )
        """
        content += """dut_digital_estimator (
            .rst(rst),
            .clk(clk),
            """
        if self.configuration_coefficients_variable_fixed == "variable":
            content += """.enable_lookup_table_coefficient_shift_in(enable_lookup_table_coefficient_shift_in),
            """
            if self.configuration_reduce_size == True:
                content += """.lookup_table_coefficient_shift_in_lookback_lookahead_switch(lookup_table_coefficient_shift_in_lookback_lookahead_switch),
            """
            content += """.lookup_table_coefficient(lookup_table_coefficient),
            """
        content += """.digital_control_input(digital_control_input),
            .signal_estimation_valid_out(signal_estimation_valid_out),
            .signal_estimation_output(signal_estimation_output),
            .clk_sample_shift_register(clk_sample_shift_register)
    );


    """
        if self.configuration_mapped_simulation == False:
            if self.configuration_combinatorial_synchronous == "combinatorial":
                content += """bind AdderCombinatorial AdderCombinatorialAssertions #(
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
            elif self.configuration_combinatorial_synchronous == "synchronous":
                content += """bind AdderSynchronous AdderSynchronousAssertions #(
            .INPUT_WIDTH(INPUT_WIDTH)
        )
        adder_synchronous_bind (
            .rst(rst),
            .clk(clk),
            .input_0(input_0),
            .input_1(input_1),
            .out(out)
    );

    bind AdderBlockSynchronous AdderBlockSynchronousAssertions #(
            .INPUT_COUNT(INPUT_COUNT),
            .INPUT_WIDTH(INPUT_WIDTH)
        )
        adder_block_synchronous_bind (
            .rst(rst),
            .clk(clk),
            .in(in),
            .out(out)
    );

    bind LookUpTableSynchronous LookUpTableSynchronousAssertions #(
            .INPUT_WIDTH(INPUT_WIDTH),
            .DATA_WIDTH(DATA_WIDTH)
        )
        look_up_table_synchronous_bind (
            .rst(rst),
            .clk(clk),
            .in(in),
            .memory(memory),
            .out(out)
    );

"""     
        if self.configuration_reduce_size == False:
            content += """\tbind LookUpTableBlockSynchronous LookUpTableBlockSynchronousAssertions #(
            .TOTAL_INPUT_WIDTH(TOTAL_INPUT_WIDTH),
            .LOOKUP_TABLE_INPUT_WIDTH(LOOKUP_TABLE_INPUT_WIDTH),
            .LOOKUP_TABLE_DATA_WIDTH(LOOKUP_TABLE_DATA_WIDTH)
        )
        look_up_table_block_synchronous_bind (
            .rst(rst),
            .clk(clk),
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
        content += "endmodule"

        self.syntax_generator.single_line_no_linebreak(content, indentation = 0)
        self.syntax_generator.close()
