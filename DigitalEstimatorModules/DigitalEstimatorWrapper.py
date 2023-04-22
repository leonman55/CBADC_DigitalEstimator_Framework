#from tkinter.messagebox import NO
import SystemVerilogModule
import SystemVerilogPort
import SystemVerilogPortDirection
import SystemVerilogPortType
import SystemVerilogDimension
import DigitalEstimatorModules.TestModule
import DigitalEstimatorModules.SimpleAdder
import DigitalEstimatorModules.LookUpTable
import SystemVerilogSignalSign
from SystemVerilogSyntaxGenerator import decimal_number, get_parameter_value, connect_port_array, ndarray_to_system_verilog_array, ndarray_to_system_verilog_concatenation, set_parameter_value
import CBADC_HighLevelSimulation
import cbadc
import numpy
import math


class DigitalEstimatorWrapper(SystemVerilogModule.SystemVerilogModule):
    module_name: str = "DigitalEstimator"
    configuration_n_number_of_analog_states: int = 6
    configuration_m_number_of_digital_states: int = configuration_n_number_of_analog_states
    configuration_fir_lut_input_width: int = 4
    configuration_data_width: int = 31
    configuration_beta: float = 6250.0
    configuration_rho: float = -1e-2
    configuration_kappa: float = -1.0
    configuration_eta2: float = 1e7
    configuration_lookback_length: int = 5
    configuration_lookahead_length: int = 1
    configuration_down_sample_rate: int = 1
    configuration_combinatorial_synchronous: str = "combinatorial"
    configuration_coefficients_variable_fixed: str = "variable"
    configuration_reduce_size: bool = False

    high_level_simulation: CBADC_HighLevelSimulation.DigitalEstimatorParameterGenerator
    
    lookback_mapped_reordered_lut_entries: list[list[int]] = list[list[int]]()
    lookback_mapped_reordered_bit_widths: list[int] = list[int]()
    lookahead_mapped_reordered_lut_entries: list[list[int]] = list[list[int]]()
    lookahead_mapped_reordered_bit_widths: list[int] = list[int]()

    def __init__(self, path: str, name: str):
        super().__init__(path, name)

    def generate(self):
        maximum_data_width: int = 0
        lookback_register_total_width: int = 0
        lookahead_register_total_width: int = 0
        if self.configuration_reduce_size == True:
            if self.lookback_mapped_reordered_bit_widths < self.lookahead_mapped_reordered_bit_widths:
                maximum_data_width = self.lookahead_mapped_reordered_bit_widths[len(self.lookahead_mapped_reordered_bit_widths) - 1][0]
            else:
                maximum_data_width = self.lookback_mapped_reordered_bit_widths[len(self.lookback_mapped_reordered_bit_widths) - 1][0]
            for index in range(len(self.lookback_mapped_reordered_lut_entries)):
                lookback_register_total_width += len(self.lookback_mapped_reordered_lut_entries[index]) * self.lookback_mapped_reordered_bit_widths[index][0]
            for index in range(len(self.lookahead_mapped_reordered_lut_entries)):
                lookahead_register_total_width += len(self.lookahead_mapped_reordered_lut_entries[index]) * self.lookahead_mapped_reordered_bit_widths[index][0]
            
        content: str = f"""module {self.module_name} #(
        parameter N_NUMBER_ANALOG_STATES = {self.configuration_n_number_of_analog_states},
        parameter M_NUMBER_DIGITAL_STATES = {self.configuration_m_number_of_digital_states},
        parameter LOOKBACK_SIZE = {self.configuration_lookback_length},
        parameter LOOKAHEAD_SIZE = {self.configuration_lookahead_length},
        localparam TOTAL_LOOKUP_REGISTER_LENGTH = LOOKAHEAD_SIZE + LOOKBACK_SIZE,
        parameter LOOKUP_TABLE_INPUT_WIDTH = {self.configuration_fir_lut_input_width},
        parameter LOOKUP_TABLE_DATA_WIDTH = {self.configuration_data_width},
        localparam LOOKBACK_LOOKUP_TABLE_COUNT = int'($ceil(real'(LOOKBACK_SIZE * M_NUMBER_DIGITAL_STATES) / real'(LOOKUP_TABLE_INPUT_WIDTH))),
        localparam LOOKBACK_LOOKUP_TABLE_ENTRIES_COUNT = int'($ceil((LOOKBACK_SIZE * M_NUMBER_DIGITAL_STATES) / LOOKUP_TABLE_INPUT_WIDTH)) * (2**LOOKUP_TABLE_INPUT_WIDTH) + (((LOOKBACK_SIZE * M_NUMBER_DIGITAL_STATES) % LOOKUP_TABLE_INPUT_WIDTH) == 0 ? 0 : (2**((LOOKBACK_SIZE * M_NUMBER_DIGITAL_STATES) % LOOKUP_TABLE_INPUT_WIDTH))),
        //localparam LOOKBACK_LOOKUP_TABLE_ENTRIES_COUNT = LOOKBACK_LOOKUP_TABLE_COUNT * (2**LOOKUP_TABLE_INPUT_WIDTH),
        localparam LOOKAHEAD_LOOKUP_TABLE_COUNT = int'($ceil(real'(LOOKAHEAD_SIZE * M_NUMBER_DIGITAL_STATES) / real'(LOOKUP_TABLE_INPUT_WIDTH))),
        localparam LOOKAHEAD_LOOKUP_TABLE_ENTRIES_COUNT = int'($ceil((LOOKAHEAD_SIZE * M_NUMBER_DIGITAL_STATES) / LOOKUP_TABLE_INPUT_WIDTH)) * (2**LOOKUP_TABLE_INPUT_WIDTH) + (((LOOKAHEAD_SIZE * M_NUMBER_DIGITAL_STATES) % LOOKUP_TABLE_INPUT_WIDTH) == 0 ? 0 : (2**((LOOKAHEAD_SIZE * M_NUMBER_DIGITAL_STATES) % LOOKUP_TABLE_INPUT_WIDTH))),
        //localparam LOOKAHEAD_LOOKUP_TABLE_ENTRIES_COUNT = LOOKAHEAD_LOOKUP_TABLE_COUNT * (2**LOOKUP_TABLE_INPUT_WIDTH),
        parameter OUTPUT_DATA_WIDTH = {self.configuration_data_width},
        parameter DOWN_SAMPLE_RATE = {self.configuration_down_sample_rate}
    ) (
        input wire rst,
        input wire clk,
        """
        if self.configuration_coefficients_variable_fixed == "variable":
            content += """input wire enable_lookup_table_coefficient_shift_in,
        """
            if self.configuration_reduce_size == True:
                content += f"""input wire lookup_table_coefficient_shift_in_lookback_lookahead_switch,
        input wire signed [{maximum_data_width} - 1 : 0] lookup_table_coefficient,
        """
            elif self.configuration_reduce_size == False:    
                content += f"""input wire signed [LOOKUP_TABLE_DATA_WIDTH - 1 : 0] lookup_table_coefficient,
        """
        content += """input wire [M_NUMBER_DIGITAL_STATES - 1 : 0] digital_control_input,
        output logic signal_estimation_valid_out,
        output logic signed [OUTPUT_DATA_WIDTH - 1 : 0] signal_estimation_output,
        output logic clk_sample_shift_register
);

    """
        if self.configuration_down_sample_rate > 1:
            content += f"""logic clk_downsample;
    """
        content += f"""//logic clk_sample_shift_register;
    logic [DOWN_SAMPLE_RATE - 1 : 0][M_NUMBER_DIGITAL_STATES - 1 : 0] downsample_accumulate_output;
    logic [TOTAL_LOOKUP_REGISTER_LENGTH - 1 : 0][M_NUMBER_DIGITAL_STATES - 1 : 0] sample_shift_register;
    """
        if self.configuration_reduce_size == False:
            content += f"""wire signed [LOOKBACK_LOOKUP_TABLE_ENTRIES_COUNT - 1 : 0][LOOKUP_TABLE_DATA_WIDTH - 1 : 0] lookback_lookup_table_entries;
    wire signed [LOOKAHEAD_LOOKUP_TABLE_ENTRIES_COUNT - 1 : 0][LOOKUP_TABLE_DATA_WIDTH - 1 : 0] lookahead_lookup_table_entries;
    
    """
        elif self.configuration_reduce_size == True:
            content += f"""wire signed [{lookback_register_total_width} - 1 : 0] lookback_lookup_table_entries;
    wire signed [{lookahead_register_total_width} - 1 : 0] lookahead_lookup_table_entries;
    
    """
        if self.configuration_down_sample_rate > 1:
            content += f"""assign clk_sample_shift_register = clk_downsample;
    
    """
        else:
            content += f"""assign clk_sample_shift_register = clk;
    
    """
        if self.configuration_coefficients_variable_fixed == "fixed":
            if self.configuration_reduce_size == False:
                content += "assign lookback_lookup_table_entries = "
                content += ndarray_to_system_verilog_array(numpy.array(CBADC_HighLevelSimulation.convert_coefficient_matrix_to_lut_entries(self.high_level_simulation.get_fir_lookback_coefficient_matrix(), self.configuration_fir_lut_input_width))) + ";\n\t"
                content += "assign lookahead_lookup_table_entries = "
                content += ndarray_to_system_verilog_array(numpy.array(CBADC_HighLevelSimulation.convert_coefficient_matrix_to_lut_entries(self.high_level_simulation.get_fir_lookahead_coefficient_matrix(), self.configuration_fir_lut_input_width))) + ";"
                content += """
        
    """
            elif self.configuration_reduce_size == True:
                content += "assign lookback_lookup_table_entries = "
                content += ndarray_to_system_verilog_concatenation(self.lookback_mapped_reordered_lut_entries, self.lookback_mapped_reordered_bit_widths) + ";\n\t"
                content += "assign lookahead_lookup_table_entries = "
                content += ndarray_to_system_verilog_concatenation(self.lookahead_mapped_reordered_lut_entries, self.lookahead_mapped_reordered_bit_widths) + ";"
                content += """
        
    """
        content += """
    always_ff @(posedge clk_sample_shift_register) begin
        if(rst == 1'b1) begin
            sample_shift_register <= {{(M_NUMBER_DIGITAL_STATES * TOTAL_LOOKUP_REGISTER_LENGTH) - 1{{1'b0}}}};
        end
        else begin
            sample_shift_register <= {{sample_shift_register[TOTAL_LOOKUP_REGISTER_LENGTH - 1 - DOWN_SAMPLE_RATE : 0], downsample_accumulate_output}};
        end
    end

    logic [LOOKBACK_SIZE - 1 : 0][M_NUMBER_DIGITAL_STATES - 1 : 0] lookback_register;
    """
        if self.configuration_reduce_size == True:
            content += """logic [(LOOKBACK_SIZE * M_NUMBER_DIGITAL_STATES) - 1 : 0] lookback_register_distribution;
    
    assign lookback_register_distribution = lookback_register;
    """
        content += """
    assign lookback_register = sample_shift_register[TOTAL_LOOKUP_REGISTER_LENGTH - 1 : LOOKAHEAD_SIZE];
    //assign lookback_register = sample_shift_register[LOOKBACK_SIZE - 1 : 0];

    logic [LOOKAHEAD_SIZE - 1 : 0][M_NUMBER_DIGITAL_STATES - 1 : 0] lookahead_register;
    """
        if self.configuration_reduce_size == True:
            content += """logic [(LOOKAHEAD_SIZE * M_NUMBER_DIGITAL_STATES) - 1 : 0] lookahead_register_distribution;
    
    assign lookahead_register_distribution = lookahead_register;
    """
        content += """
    generate
        for(genvar lookahead_index = 0; lookahead_index < LOOKAHEAD_SIZE; lookahead_index++) begin
            assign lookahead_register[lookahead_index] = sample_shift_register[LOOKAHEAD_SIZE - 1 - lookahead_index];
        end
    endgenerate
    //assign lookahead_register = sample_shift_register[TOTAL_LOOKUP_REGISTER_LENGTH - 1 : LOOKBACK_SIZE];
    //assign lookahead_register = sample_shift_register[LOOKAHEAD_SIZE - 1 : 0];

    logic signed [LOOKBACK_LOOKUP_TABLE_COUNT - 1 : 0][LOOKUP_TABLE_DATA_WIDTH - 1 : 0] lookback_lookup_table_results;
    logic signed [LOOKAHEAD_LOOKUP_TABLE_COUNT - 1 : 0][LOOKUP_TABLE_DATA_WIDTH - 1 : 0] lookahead_lookup_table_results;

    wire signed [LOOKUP_TABLE_DATA_WIDTH - 1 : 0] adder_block_lookback_result;
    wire signed [LOOKUP_TABLE_DATA_WIDTH - 1 : 0] adder_block_lookahead_result;

    """
        if self.configuration_combinatorial_synchronous == "combinatorial":
            content += """assign signal_estimation_output = adder_block_lookback_result + adder_block_lookahead_result;"""
        elif self.configuration_combinatorial_synchronous == "synchronous":
            content += """always_ff @(posedge clk_sample_shift_register) begin
        signal_estimation_output = adder_block_lookback_result + adder_block_lookahead_result;
    end
    
    
    """

        if self.configuration_coefficients_variable_fixed == "variable":
            content += """LookUpTableCoefficientRegister #(
            """
            if self.configuration_reduce_size == False:
                content += """.LOOKUP_TABLE_DATA_WIDTH(LOOKUP_TABLE_DATA_WIDTH),
            .LOOKBACK_LOOKUP_TABLE_ENTRIES_COUNT(LOOKBACK_LOOKUP_TABLE_ENTRIES_COUNT),
            .LOOKAHEAD_LOOKUP_TABLE_ENTRIES_COUNT(LOOKAHEAD_LOOKUP_TABLE_ENTRIES_COUNT)
        """
            content += """)
        lookup_table_coefficient_register (
            .rst(rst),
            .clk(clk),
            .enable_input(enable_lookup_table_coefficient_shift_in),
            """
            if self.configuration_reduce_size == True:
                content += """.lookback_lookahead_switch(lookup_table_coefficient_shift_in_lookback_lookahead_switch),
            """
            content += """.coefficient_in(lookup_table_coefficient),
            .lookback_coefficients(lookback_lookup_table_entries),
            .lookahead_coefficients(lookahead_lookup_table_entries)
    );
    
    """
        if self.configuration_down_sample_rate > 1:
            content += f"""ClockDivider #(
            .DOWN_SAMPLE_RATE(DOWN_SAMPLE_RATE)
        )
        clock_divider (
            .rst(rst),
            .clk(clk),
            .clk_downsample(clk_downsample),
            .clock_divider_counter()
    );

    """
        content += f"""InputDownsampleAccumulateRegister #(
            .REGISTER_LENGTH(DOWN_SAMPLE_RATE),
            .DATA_WIDTH(M_NUMBER_DIGITAL_STATES)
        )
        input_downsample_accumulate_register (
            .rst(rst),
            .clk(clk),
            .in(digital_control_input),
            .out(downsample_accumulate_output)
    );

    """
        if self.configuration_combinatorial_synchronous == "combinatorial":
            content += """LookUpTableBlock #(
            .TOTAL_INPUT_WIDTH(LOOKBACK_SIZE * M_NUMBER_DIGITAL_STATES),
            .LOOKUP_TABLE_INPUT_WIDTH(LOOKUP_TABLE_INPUT_WIDTH),
            .LOOKUP_TABLE_DATA_WIDTH(LOOKUP_TABLE_DATA_WIDTH)
        )
        lookback_lookup_table_block (
            .rst(rst),
            .input_register(lookback_register),
            .lookup_table_entries(lookback_lookup_table_entries),
            .lookup_table_results(lookback_lookup_table_results)
    );

    AdderBlockCombinatorial #(
            .INPUT_COUNT(LOOKBACK_LOOKUP_TABLE_COUNT),
            .INPUT_WIDTH(LOOKUP_TABLE_DATA_WIDTH)
        )
        adder_block_lookback (
            .rst(rst),
            .in(lookback_lookup_table_results),
            .out(adder_block_lookback_result)
    );

    LookUpTableBlock #(
            .TOTAL_INPUT_WIDTH(LOOKAHEAD_SIZE * M_NUMBER_DIGITAL_STATES),
            .LOOKUP_TABLE_INPUT_WIDTH(LOOKUP_TABLE_INPUT_WIDTH),
            .LOOKUP_TABLE_DATA_WIDTH(LOOKUP_TABLE_DATA_WIDTH)
        )
        lookahead_lookup_table_block (
            .rst(rst),
            .input_register(lookahead_register),
            .lookup_table_entries(lookahead_lookup_table_entries),
            .lookup_table_results(lookahead_lookup_table_results)
    );

    AdderBlockCombinatorial #(
            .INPUT_COUNT(LOOKAHEAD_LOOKUP_TABLE_COUNT),
            .INPUT_WIDTH(LOOKUP_TABLE_DATA_WIDTH)
        )
        adder_block_lookahead (
            .rst(rst),
            .in(lookahead_lookup_table_results),
            .out(adder_block_lookahead_result)
    );

    """
        elif self.configuration_combinatorial_synchronous == "synchronous":
            if self.configuration_reduce_size == False:
                content += """LookUpTableBlockSynchronous #(
            .TOTAL_INPUT_WIDTH(LOOKBACK_SIZE * M_NUMBER_DIGITAL_STATES),
            .LOOKUP_TABLE_INPUT_WIDTH(LOOKUP_TABLE_INPUT_WIDTH),
            .LOOKUP_TABLE_DATA_WIDTH(LOOKUP_TABLE_DATA_WIDTH)
        )
        lookback_lookup_table_block (
            .rst(rst),
            .clk(clk_downsample),
            .input_register(lookback_register),
            .lookup_table_entries(lookback_lookup_table_entries),
            .lookup_table_results(lookback_lookup_table_results)
    );

    """
            elif self.configuration_reduce_size == True:
                content += """//localparam LOOKBACK_LOOKUP_TABLE_COUNT = int'($ceil(real'(LOOKBACK_SIZE * M_NUMBER_DIGITAL_STATES) / real'(LOOKUP_TABLE_INPUT_WIDTH)));
    //logic signed [LOOKBACK_LOOKUP_TABLE_COUNT - 1 : 0][LOOKUP_TABLE_DATA_WIDTH - 1 : 0] lookback_lookup_table_results;
    
    """
                for lookback_lut_index in range(len(self.lookback_mapped_reordered_lut_entries)):
                    lut_offset: int = 0
                    for lut_index in range(len(self.lookback_mapped_reordered_bit_widths) - 1, -1, -1):
                        if lut_index == lookback_lut_index:
                            break
                        lut_offset += len(self.lookback_mapped_reordered_lut_entries[lut_index]) * self.lookback_mapped_reordered_bit_widths[lut_index][0]
                    lut_memory_contents: str = "}"
                    for entry_index in range(len(self.lookback_mapped_reordered_lut_entries[lookback_lut_index])):
                        lut_memory_contents = f", lookback_lookup_table_entries[{lut_offset + (entry_index + 1) * self.lookback_mapped_reordered_bit_widths[lookback_lut_index][0]} - 1 : {lut_offset + entry_index * self.lookback_mapped_reordered_bit_widths[lookback_lut_index][0]}]" + lut_memory_contents
                        if self.lookback_mapped_reordered_lut_entries[lookback_lut_index][len(self.lookback_mapped_reordered_lut_entries[lookback_lut_index]) - 1 - entry_index] >= 0:
                            lut_memory_contents = f", {{{self.configuration_data_width - self.lookback_mapped_reordered_bit_widths[lookback_lut_index][0]}{{1'b0}}}}" + lut_memory_contents
                        elif self.lookback_mapped_reordered_lut_entries[lookback_lut_index][len(self.lookback_mapped_reordered_lut_entries[lookback_lut_index]) - 1 - entry_index] < 0:
                            lut_memory_contents = f", {{{self.configuration_data_width - self.lookback_mapped_reordered_bit_widths[lookback_lut_index][0]}{{1'b1}}}}" + lut_memory_contents
                        #lut_memory_contents = f", {self.configuration_data_width - self.lookback_mapped_reordered_bit_widths[lookback_lut_index][0]}'h0" + lut_memory_contents
                    lut_memory_contents = "{" + lut_memory_contents.removeprefix(", ")
                    content += f"""LookUpTableSynchronous #(
            .INPUT_WIDTH(LOOKUP_TABLE_INPUT_WIDTH),
            .DATA_WIDTH(LOOKUP_TABLE_DATA_WIDTH)
        )
        lookback_lookup_table_{self.lookback_mapped_reordered_bit_widths[lookback_lut_index][1]} (
            .rst(rst),
            .clk(clk),
            .in(lookback_register_distribution[({len(self.lookback_mapped_reordered_bit_widths) - 1 - self.lookback_mapped_reordered_bit_widths[lookback_lut_index][1]} * LOOKUP_TABLE_INPUT_WIDTH) +: LOOKUP_TABLE_INPUT_WIDTH]),
            .memory({lut_memory_contents}),
            .out(lookback_lookup_table_results[{self.lookback_mapped_reordered_bit_widths[lookback_lut_index][1]}])
    );
                
    """
            content += """AdderBlockSynchronous #(
            .INPUT_COUNT(LOOKBACK_LOOKUP_TABLE_COUNT),
            .INPUT_WIDTH(LOOKUP_TABLE_DATA_WIDTH)
        )
        adder_block_lookback (
            .rst(rst),
            .clk(clk_downsample),
            .in(lookback_lookup_table_results),
            .out(adder_block_lookback_result)
    );

    """
            if self.configuration_reduce_size == False:
                content += """LookUpTableBlockSynchronous #(
            .TOTAL_INPUT_WIDTH(LOOKAHEAD_SIZE * M_NUMBER_DIGITAL_STATES),
            .LOOKUP_TABLE_INPUT_WIDTH(LOOKUP_TABLE_INPUT_WIDTH),
            .LOOKUP_TABLE_DATA_WIDTH(LOOKUP_TABLE_DATA_WIDTH)
        )
        lookahead_lookup_table_block (
            .rst(rst),
            .clk(clk_downsample),
            .input_register(lookahead_register),
            .lookup_table_entries(lookahead_lookup_table_entries),
            .lookup_table_results(lookahead_lookup_table_results)
    );

    """
            elif self.configuration_reduce_size == True:
                content += """//localparam LOOKAHEAD_LOOKUP_TABLE_COUNT = int'($ceil(real'(LOOKAHEAD_SIZE * M_NUMBER_DIGITAL_STATES) / real'(LOOKUP_TABLE_INPUT_WIDTH)));
    //logic signed [LOOKAHEAD_LOOKUP_TABLE_COUNT - 1 : 0][LOOKUP_TABLE_DATA_WIDTH - 1 : 0] lookahead_lookup_table_results;
    
    """
                for lookahead_lut_index in range(len(self.lookahead_mapped_reordered_lut_entries)):
                    lut_offset: int = 0
                    for lut_index in range(len(self.lookahead_mapped_reordered_bit_widths) - 1, -1, -1):
                        if lut_index == lookahead_lut_index:
                            break
                        lut_offset += len(self.lookahead_mapped_reordered_lut_entries[lut_index]) * self.lookahead_mapped_reordered_bit_widths[lut_index][0]
                    lut_memory_contents: str = "}"
                    for entry_index in range(len(self.lookahead_mapped_reordered_lut_entries[lookahead_lut_index])):
                        lut_memory_contents = f", lookahead_lookup_table_entries[{lut_offset + (entry_index + 1) * self.lookahead_mapped_reordered_bit_widths[lookahead_lut_index][0]} - 1 : {lut_offset + entry_index * self.lookahead_mapped_reordered_bit_widths[lookahead_lut_index][0]}]" + lut_memory_contents
                        if self.lookahead_mapped_reordered_lut_entries[lookahead_lut_index][len(self.lookahead_mapped_reordered_lut_entries[lookahead_lut_index]) - 1 - entry_index] >= 0:
                            lut_memory_contents = f", {{{self.configuration_data_width - self.lookahead_mapped_reordered_bit_widths[lookahead_lut_index][0]}{{1'b0}}}}" + lut_memory_contents
                        elif self.lookahead_mapped_reordered_lut_entries[lookahead_lut_index][len(self.lookahead_mapped_reordered_lut_entries[lookahead_lut_index]) - 1 - entry_index] < 0:
                            lut_memory_contents = f", {{{self.configuration_data_width - self.lookahead_mapped_reordered_bit_widths[lookahead_lut_index][0]}{{1'b1}}}}" + lut_memory_contents
                    lut_memory_contents = "{" + lut_memory_contents.removeprefix(", ")
                    content += f"""LookUpTableSynchronous #(
            .INPUT_WIDTH(LOOKUP_TABLE_INPUT_WIDTH),
            .DATA_WIDTH(LOOKUP_TABLE_DATA_WIDTH)
        )
        lookahead_lookup_table_{self.lookahead_mapped_reordered_bit_widths[lookahead_lut_index][1]} (
            .rst(rst),
            .clk(clk),
            .in(lookahead_register_distribution[({len(self.lookahead_mapped_reordered_bit_widths) - 1 - self.lookahead_mapped_reordered_bit_widths[lookahead_lut_index][1]} * LOOKUP_TABLE_INPUT_WIDTH) +: LOOKUP_TABLE_INPUT_WIDTH]),
            .memory({lut_memory_contents}),
            .out(lookahead_lookup_table_results[{self.lookahead_mapped_reordered_bit_widths[lookahead_lut_index][1]}])
    );
                
    """
            content += """AdderBlockSynchronous #(
            .INPUT_COUNT(LOOKAHEAD_LOOKUP_TABLE_COUNT),
            .INPUT_WIDTH(LOOKUP_TABLE_DATA_WIDTH)
        )
        adder_block_lookahead (
            .rst(rst),
            .clk(clk_downsample),
            .in(lookahead_lookup_table_results),
            .out(adder_block_lookahead_result)
    );
    
    """

        content += """ValidCounter #(
            .TOP_VALUE(LOOKBACK_SIZE + LOOKAHEAD_SIZE)
        )
        valid_counter (
            .rst(rst),
            .clk(clk),
            .valid(signal_estimation_valid_out)
    );


endmodule"""

        self.syntax_generator.single_line_no_linebreak(content, indentation = 0)
        self.syntax_generator.close()