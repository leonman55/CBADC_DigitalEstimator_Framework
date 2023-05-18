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
    configuration_reduce_size_coefficients: bool = False
    configuration_reduce_size_luts: bool = False
    configuration_reduce_size_adders: bool = False

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
        lookback_luts_result_register_total_width: int = 0
        lookahead_luts_result_register_total_width: int = 0
        if self.configuration_reduce_size_coefficients == True:
            if self.lookback_mapped_reordered_bit_widths < self.lookahead_mapped_reordered_bit_widths:
                maximum_data_width = self.lookahead_mapped_reordered_bit_widths[len(self.lookahead_mapped_reordered_bit_widths) - 1][0]
            else:
                maximum_data_width = self.lookback_mapped_reordered_bit_widths[len(self.lookback_mapped_reordered_bit_widths) - 1][0]
            for index in range(len(self.lookback_mapped_reordered_lut_entries)):
                lookback_register_total_width += len(self.lookback_mapped_reordered_lut_entries[index]) * self.lookback_mapped_reordered_bit_widths[index][0]
            for index in range(len(self.lookahead_mapped_reordered_lut_entries)):
                lookahead_register_total_width += len(self.lookahead_mapped_reordered_lut_entries[index]) * self.lookahead_mapped_reordered_bit_widths[index][0]
            for entry in self.lookback_mapped_reordered_bit_widths:
                lookback_luts_result_register_total_width += entry[0]
            for entry in self.lookahead_mapped_reordered_bit_widths:
                lookahead_luts_result_register_total_width += entry[0]
            
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
            if self.configuration_reduce_size_coefficients == True:
                content += f"""input wire lookup_table_coefficient_shift_in_lookback_lookahead_switch,
        input wire signed [{maximum_data_width} - 1 : 0] lookup_table_coefficient,
        """
            elif self.configuration_reduce_size_coefficients == False:    
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
        if self.configuration_reduce_size_coefficients == False:
            content += f"""wire signed [LOOKBACK_LOOKUP_TABLE_ENTRIES_COUNT - 1 : 0][LOOKUP_TABLE_DATA_WIDTH - 1 : 0] lookback_lookup_table_entries;
    wire signed [LOOKAHEAD_LOOKUP_TABLE_ENTRIES_COUNT - 1 : 0][LOOKUP_TABLE_DATA_WIDTH - 1 : 0] lookahead_lookup_table_entries;
    
    """
        elif self.configuration_reduce_size_coefficients == True:
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
            if self.configuration_reduce_size_coefficients == False:
                content += "assign lookback_lookup_table_entries = "
                content += ndarray_to_system_verilog_array(numpy.array(CBADC_HighLevelSimulation.convert_coefficient_matrix_to_lut_entries(self.high_level_simulation.get_fir_lookback_coefficient_matrix(), self.configuration_fir_lut_input_width))) + ";\n\t"
                content += "assign lookahead_lookup_table_entries = "
                content += ndarray_to_system_verilog_array(numpy.array(CBADC_HighLevelSimulation.convert_coefficient_matrix_to_lut_entries(self.high_level_simulation.get_fir_lookahead_coefficient_matrix(), self.configuration_fir_lut_input_width))) + ";"
                content += """
        
    """
            elif self.configuration_reduce_size_coefficients == True:
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
        if self.configuration_reduce_size_coefficients == True:
            content += """logic [(LOOKBACK_SIZE * M_NUMBER_DIGITAL_STATES) - 1 : 0] lookback_register_distribution;
    
    assign lookback_register_distribution = lookback_register;
    """
        content += """
    assign lookback_register = sample_shift_register[TOTAL_LOOKUP_REGISTER_LENGTH - 1 : LOOKAHEAD_SIZE];
    //assign lookback_register = sample_shift_register[LOOKBACK_SIZE - 1 : 0];

    logic [LOOKAHEAD_SIZE - 1 : 0][M_NUMBER_DIGITAL_STATES - 1 : 0] lookahead_register;
    """
        if self.configuration_reduce_size_coefficients == True:
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

    """
        if self.configuration_reduce_size_adders == True:
            content += f"""logic [{lookback_luts_result_register_total_width} - 1 : 0] lookback_lookup_table_results;
    logic [{lookahead_luts_result_register_total_width} - 1 : 0] lookahead_lookup_table_results;
    
    """
        elif self.configuration_reduce_size_adders == False:
            content += """logic signed [LOOKBACK_LOOKUP_TABLE_COUNT - 1 : 0][LOOKUP_TABLE_DATA_WIDTH - 1 : 0] lookback_lookup_table_results;
    logic signed [LOOKAHEAD_LOOKUP_TABLE_COUNT - 1 : 0][LOOKUP_TABLE_DATA_WIDTH - 1 : 0] lookahead_lookup_table_results;

    """
            content += """wire signed [LOOKUP_TABLE_DATA_WIDTH - 1 : 0] adder_block_lookback_result;
    wire signed [LOOKUP_TABLE_DATA_WIDTH - 1 : 0] adder_block_lookahead_result;

    """

        if self.configuration_coefficients_variable_fixed == "variable":
            content += """LookUpTableCoefficientRegister #(
            """
            if self.configuration_reduce_size_coefficients == False:
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
            if self.configuration_reduce_size_coefficients == True:
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
            if self.configuration_reduce_size_coefficients == False:
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

    """     
            elif self.configuration_reduce_size_coefficients == True:
                for lookback_lut_index in range(len(self.lookback_mapped_reordered_lut_entries)):
                    lut_offset: int = 0
                    result_offset: int = 0
                    for lut_index in range(len(self.lookback_mapped_reordered_bit_widths) - 1, -1, -1):
                        if lut_index == lookback_lut_index:
                            break
                        lut_offset += len(self.lookback_mapped_reordered_lut_entries[lut_index]) * self.lookback_mapped_reordered_bit_widths[lut_index][0]
                        result_offset += self.lookback_mapped_reordered_bit_widths[lut_index][0]
                    lut_memory_contents: str = "}"
                    for entry_index in range(len(self.lookback_mapped_reordered_lut_entries[lookback_lut_index])):
                        #if self.configuration_reduce_size_luts == True:
                        lut_memory_contents = f", lookback_lookup_table_entries[{lut_offset + (entry_index + 1) * self.lookback_mapped_reordered_bit_widths[lookback_lut_index][0]} - 1 : {lut_offset + entry_index * self.lookback_mapped_reordered_bit_widths[lookback_lut_index][0]}]" + lut_memory_contents
                        if self.configuration_reduce_size_luts == False:
                            if self.lookback_mapped_reordered_lut_entries[lookback_lut_index][len(self.lookback_mapped_reordered_lut_entries[lookback_lut_index]) - 1 - entry_index] >= 0:
                                lut_memory_contents = f", {{{self.configuration_data_width - self.lookback_mapped_reordered_bit_widths[lookback_lut_index][0]}{{1'b0}}}}" + lut_memory_contents
                            elif self.lookback_mapped_reordered_lut_entries[lookback_lut_index][len(self.lookback_mapped_reordered_lut_entries[lookback_lut_index]) - 1 - entry_index] < 0:
                                lut_memory_contents = f", {{{self.configuration_data_width - self.lookback_mapped_reordered_bit_widths[lookback_lut_index][0]}{{1'b1}}}}" + lut_memory_contents
                    lut_memory_contents = "{" + lut_memory_contents.removeprefix(", ")
                    content += f"""LookUpTable #(
            .INPUT_WIDTH(LOOKUP_TABLE_INPUT_WIDTH),
            """
                    if self.configuration_reduce_size_luts == True:
                        content += f""".DATA_WIDTH({self.lookback_mapped_reordered_bit_widths[lookback_lut_index][0]})
        """
                    elif self.configuration_reduce_size_luts == False:
                        content += f""".DATA_WIDTH({self.configuration_data_width})
        """
                    content += f""")
        lookback_lookup_table_{self.lookback_mapped_reordered_bit_widths[lookback_lut_index][1]} (
            .rst(rst),
            .in(lookback_register_distribution[({len(self.lookback_mapped_reordered_bit_widths) - 1 - self.lookback_mapped_reordered_bit_widths[lookback_lut_index][1]} * LOOKUP_TABLE_INPUT_WIDTH) +: LOOKUP_TABLE_INPUT_WIDTH]),
            .memory({lut_memory_contents}),
            """
                    if self.configuration_reduce_size_adders == True:
                        content += f""".out(lookback_lookup_table_results[{self.lookback_mapped_reordered_bit_widths[lookback_lut_index][0] + result_offset} - 1 : {result_offset}])
    """
                    elif self.configuration_reduce_size_adders == False:
                        content += f""".out(lookback_lookup_table_results[{lookback_lut_index}])
    """
                    content += f""");
                
    """
                
            if self.configuration_reduce_size_adders == False:
                content += """AdderBlockCombinatorial #(
            .INPUT_COUNT(LOOKBACK_LOOKUP_TABLE_COUNT),
            .INPUT_WIDTH(LOOKUP_TABLE_DATA_WIDTH)
        )
        adder_block_lookback (
            .rst(rst),
            .in(lookback_lookup_table_results),
            .out(adder_block_lookback_result)
    );

    """
            elif self.configuration_reduce_size_adders == True:
                lookback_lookup_table_count: int = math.ceil(self.configuration_lookback_length * self.configuration_m_number_of_digital_states / self.configuration_fir_lut_input_width)
                lookback_adder_stage_count: int = math.ceil(math.log2(lookback_lookup_table_count))
                #number_of_intermediate_lookback_results: int = 0
                #for stage_index in range(1, lookback_adder_stage_count + 1, 1):
                #    number_of_intermediate_lookback_results += math.ceil(lookback_lookup_table_count / 2**stage_index)
                lookback_intermediate_results_register_width: int = 0
                stages_output_widths: list[list[int]] = list[list[int]]()
                for stage_index in range(1, lookback_adder_stage_count + 1, 1):
                    stage_output_widths: list[int] = list[int]()
                    if stage_index == 1:
                        for adder_index in range(math.ceil(len(self.lookback_mapped_reordered_bit_widths) / 2)):
                            if 2 * adder_index != len(self.lookback_mapped_reordered_bit_widths) - 1:
                                if(self.lookback_mapped_reordered_bit_widths[2 * adder_index][0] < self.lookback_mapped_reordered_bit_widths[2 * adder_index + 1][0]):
                                    stage_output_widths.append(self.lookback_mapped_reordered_bit_widths[2 * adder_index + 1][0] + 1)
                                    lookback_intermediate_results_register_width += self.lookback_mapped_reordered_bit_widths[2 * adder_index + 1][0] + 1
                                elif(self.lookback_mapped_reordered_bit_widths[2 * adder_index][0] >= self.lookback_mapped_reordered_bit_widths[2 * adder_index + 1][0]):
                                    stage_output_widths.append(self.lookback_mapped_reordered_bit_widths[2 * adder_index][0] + 1)
                                    lookback_intermediate_results_register_width += self.lookback_mapped_reordered_bit_widths[2 * adder_index][0] + 1
                            elif 2 * adder_index == len(self.lookback_mapped_reordered_bit_widths) - 1:
                                stage_output_widths.append(self.lookback_mapped_reordered_bit_widths[len(self.lookback_mapped_reordered_bit_widths) - 1][0])
                                lookback_intermediate_results_register_width += self.lookback_mapped_reordered_bit_widths[len(self.lookback_mapped_reordered_bit_widths) - 1][0]
                    elif stage_index > 1:
                        for adder_index in range(math.ceil(len(stages_output_widths[stage_index - 2]) / 2)):
                            if 2 * adder_index != len(stages_output_widths[stage_index - 2]) - 1:
                                if(stages_output_widths[stage_index - 2][2 * adder_index] < stages_output_widths[stage_index - 2][2 * adder_index + 1]):
                                    stage_output_widths.append(stages_output_widths[stage_index - 2][2 * adder_index + 1] + 1)
                                    lookback_intermediate_results_register_width += stages_output_widths[stage_index - 2][2 * adder_index + 1] + 1
                                elif(stages_output_widths[stage_index - 2][2 * adder_index] >= stages_output_widths[stage_index - 2][2 * adder_index + 1]):
                                    stage_output_widths.append(stages_output_widths[stage_index - 2][2 * adder_index] + 1)
                                    lookback_intermediate_results_register_width += stages_output_widths[stage_index - 2][2 * adder_index] + 1
                            elif 2 * adder_index == len(stages_output_widths[stage_index - 2]) - 1:
                                stage_output_widths.append(stages_output_widths[stage_index - 2][len(stages_output_widths[stage_index - 2]) - 1])
                                lookback_intermediate_results_register_width += stages_output_widths[stage_index - 2][len(stages_output_widths[stage_index - 2]) - 1]
                    stages_output_widths.append(stage_output_widths)                       
                print(stages_output_widths)
                print(lookback_intermediate_results_register_width)
                
                content += f"""logic [{lookback_intermediate_results_register_width} - 1 : 0] lookback_adder_results;
    
    """
                lookback_total_output_offset: int = 0
                last_stage_output_offset: int = 0
                for stage_index in range(1, lookback_adder_stage_count + 1, 1):
                    if stage_index == 1:
                        input_offset: int = 0
                        stage_output_offset: int = 0
                        for adder_index in range(math.ceil(len(self.lookback_mapped_reordered_bit_widths) / 2)):
                            if 2 * adder_index != len(self.lookback_mapped_reordered_bit_widths) - 1:
                                content += f"""AdderCombinatorial #(
            .INPUT0_WIDTH({self.lookback_mapped_reordered_bit_widths[2 * adder_index][0]}),
            .INPUT1_WIDTH({self.lookback_mapped_reordered_bit_widths[2 * adder_index + 1][0]})
        )
        lookback_adder_synchronous_{stage_index - 1}_{adder_index}(
            .rst(rst),
            .input_0(lookback_lookup_table_results[{lookback_luts_result_register_total_width - input_offset} - 1 : {lookback_luts_result_register_total_width - (input_offset + self.lookback_mapped_reordered_bit_widths[2 * adder_index][0])}]),
            .input_1(lookback_lookup_table_results[{lookback_luts_result_register_total_width - (input_offset + self.lookback_mapped_reordered_bit_widths[2 * adder_index][0])} - 1 : {lookback_luts_result_register_total_width - (input_offset + self.lookback_mapped_reordered_bit_widths[2 * adder_index][0] + self.lookback_mapped_reordered_bit_widths[2 * adder_index + 1][0])}]),
            .out(lookback_adder_results[{stage_output_offset + stages_output_widths[0][adder_index]} - 1 : {stage_output_offset}])
    );
    
    """
                                input_offset += self.lookback_mapped_reordered_bit_widths[2 * adder_index][0] + self.lookback_mapped_reordered_bit_widths[2 * adder_index + 1][0]
                                stage_output_offset += stages_output_widths[0][adder_index]
                            elif 2 * adder_index == len(self.lookback_mapped_reordered_bit_widths) - 1:
                                content += f"""assign lookback_adder_results[{stage_output_offset + stages_output_widths[0][adder_index]} - 1 : {stage_output_offset}] = lookback_lookup_table_results[{lookback_luts_result_register_total_width - input_offset} - 1 : {lookback_luts_result_register_total_width - (input_offset + self.lookback_mapped_reordered_bit_widths[2 * adder_index][0])}];
    
    """
                                input_offset += self.lookback_mapped_reordered_bit_widths[2 * adder_index][0]
                                stage_output_offset += stages_output_widths[0][adder_index]
                        lookback_total_output_offset += stage_output_offset
                        last_stage_output_offset = stage_output_offset
                    elif stage_index > 1:
                        input_offset: int = 0
                        stage_output_offset: int = 0
                        for adder_index in range(math.ceil(len(stages_output_widths[stage_index - 2]) / 2)):
                            if 2 * adder_index != len(stages_output_widths[stage_index - 2]) - 1:
                                content += f"""AdderCombinatorial #(
            .INPUT0_WIDTH({stages_output_widths[stage_index - 2][2 * adder_index]}),
            .INPUT1_WIDTH({stages_output_widths[stage_index - 2][2 * adder_index + 1]})
        )
        lookback_adder_synchronous_{stage_index - 1}_{adder_index}(
            .rst(rst),
            .input_0(lookback_adder_results[{(lookback_total_output_offset - last_stage_output_offset) + input_offset + stages_output_widths[stage_index - 2][2 * adder_index]} - 1 : {(lookback_total_output_offset - last_stage_output_offset) + input_offset}]),
            .input_1(lookback_adder_results[{(lookback_total_output_offset - last_stage_output_offset) + input_offset + stages_output_widths[stage_index - 2][2 * adder_index] + stages_output_widths[stage_index - 2][2 * adder_index + 1]} - 1 : {(lookback_total_output_offset - last_stage_output_offset) + input_offset + stages_output_widths[stage_index - 2][2 * adder_index]}]),
            .out(lookback_adder_results[{lookback_total_output_offset + stage_output_offset + stages_output_widths[stage_index - 1][adder_index]} - 1 : {lookback_total_output_offset + stage_output_offset}])
    );
    
    """
                                input_offset += stages_output_widths[stage_index - 2][2 * adder_index] + stages_output_widths[stage_index - 2][2 * adder_index + 1]
                                stage_output_offset += stages_output_widths[stage_index - 1][adder_index]
                            elif 2 * adder_index == len(stages_output_widths[stage_index - 2]) - 1:
                                content += f"""assign lookback_adder_results[{lookback_total_output_offset + stage_output_offset + stages_output_widths[stage_index - 1][adder_index]} - 1 : {lookback_total_output_offset + stage_output_offset}] = lookback_adder_results[{(lookback_total_output_offset - last_stage_output_offset) + input_offset + stages_output_widths[stage_index - 2][2 * adder_index]} - 1 : {(lookback_total_output_offset - last_stage_output_offset) + input_offset}];
    
    """
                                input_offset += stages_output_widths[stage_index - 2][2 * adder_index]
                                stage_output_offset += stages_output_widths[stage_index - 1][adder_index]
                        lookback_total_output_offset += stage_output_offset
                        last_stage_output_offset = stage_output_offset
                content += f"""logic signed [{stages_output_widths[len(stages_output_widths) - 1][0]} - 1 : 0] adder_block_lookback_result;
    assign adder_block_lookback_result = lookback_adder_results[{lookback_intermediate_results_register_width} - 1 : {lookback_intermediate_results_register_width - stages_output_widths[len(stages_output_widths) - 1][0]}];
    
    """
            
            if self.configuration_reduce_size_coefficients == False:
                content += """LookUpTableBlock #(
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

    """     
            elif self.configuration_reduce_size_coefficients == True:
                for lookahead_lut_index in range(len(self.lookahead_mapped_reordered_lut_entries)):
                    lut_offset: int = 0
                    result_offset: int = 0
                    for lut_index in range(len(self.lookahead_mapped_reordered_bit_widths) - 1, -1, -1):
                        if lut_index == lookahead_lut_index:
                            break
                        lut_offset += len(self.lookahead_mapped_reordered_lut_entries[lut_index]) * self.lookahead_mapped_reordered_bit_widths[lut_index][0]
                        result_offset += self.lookback_mapped_reordered_bit_widths[lut_index][0]
                    lut_memory_contents: str = "}"
                    for entry_index in range(len(self.lookahead_mapped_reordered_lut_entries[lookahead_lut_index])):
                        #if self.configuration_reduce_size_luts == True:
                        lut_memory_contents = f", lookahead_lookup_table_entries[{lut_offset + (entry_index + 1) * self.lookahead_mapped_reordered_bit_widths[lookahead_lut_index][0]} - 1 : {lut_offset + entry_index * self.lookahead_mapped_reordered_bit_widths[lookahead_lut_index][0]}]" + lut_memory_contents
                        if self.configuration_reduce_size_luts == False:
                            if self.lookahead_mapped_reordered_lut_entries[lookahead_lut_index][len(self.lookahead_mapped_reordered_lut_entries[lookahead_lut_index]) - 1 - entry_index] >= 0:
                                lut_memory_contents = f", {{{self.configuration_data_width - self.lookahead_mapped_reordered_bit_widths[lookahead_lut_index][0]}{{1'b0}}}}" + lut_memory_contents
                            elif self.lookahead_mapped_reordered_lut_entries[lookahead_lut_index][len(self.lookahead_mapped_reordered_lut_entries[lookahead_lut_index]) - 1 - entry_index] < 0:
                                lut_memory_contents = f", {{{self.configuration_data_width - self.lookahead_mapped_reordered_bit_widths[lookahead_lut_index][0]}{{1'b1}}}}" + lut_memory_contents
                    lut_memory_contents = "{" + lut_memory_contents.removeprefix(", ")
                    content += f"""LookUpTable #(
            .INPUT_WIDTH(LOOKUP_TABLE_INPUT_WIDTH),
            """
                    if self.configuration_reduce_size_luts == True:
                        content += f""".DATA_WIDTH({self.lookahead_mapped_reordered_bit_widths[lut_index][0]})
        """
                    elif self.configuration_reduce_size_luts == False:
                        content += f""".DATA_WIDTH({self.configuration_data_width})
        """
                    content += f""")
        lookahead_lookup_table_{self.lookahead_mapped_reordered_bit_widths[lookahead_lut_index][1]} (
            .rst(rst),
            .in(lookahead_register_distribution[({len(self.lookahead_mapped_reordered_bit_widths) - 1 - self.lookahead_mapped_reordered_bit_widths[lookahead_lut_index][1]} * LOOKUP_TABLE_INPUT_WIDTH) +: LOOKUP_TABLE_INPUT_WIDTH]),
            .memory({lut_memory_contents}),
            """
                    if self.configuration_reduce_size_adders == True:
                        content += f""".out(lookahead_lookup_table_results[{self.lookahead_mapped_reordered_bit_widths[lookahead_lut_index][0] + result_offset} - 1 : {result_offset}])
    """
                    elif self.configuration_reduce_size_adders == False:
                        content += f""".out(lookahead_lookup_table_results[{lookahead_lut_index}])
    """
                    content += f""");
                
    """
            
            if self.configuration_reduce_size_adders == False:
                content += """AdderBlockCombinatorial #(
            .INPUT_COUNT(LOOKAHEAD_LOOKUP_TABLE_COUNT),
            .INPUT_WIDTH(LOOKUP_TABLE_DATA_WIDTH)
        )
        adder_block_lookahead (
            .rst(rst),
            .in(lookahead_lookup_table_results),
            .out(adder_block_lookahead_result)
    );

    """
            elif self.configuration_reduce_size_adders == True:
                lookahead_lookup_table_count: int = math.ceil(self.configuration_lookahead_length * self.configuration_m_number_of_digital_states / self.configuration_fir_lut_input_width)
                lookahead_adder_stage_count: int = math.ceil(math.log2(lookahead_lookup_table_count))
                #number_of_intermediate_lookahead_results: int = 0
                #for stage_index in range(1, lookahead_adder_stage_count + 1, 1):
                #    number_of_intermediate_lookahead_results += math.ceil(lookahead_lookup_table_count / 2**stage_index)
                lookahead_intermediate_results_register_width: int = 0
                stages_output_widths: list[list[int]] = list[list[int]]()
                for stage_index in range(1, lookahead_adder_stage_count + 1, 1):
                    stage_output_widths: list[int] = list[int]()
                    if stage_index == 1:
                        for adder_index in range(math.ceil(len(self.lookahead_mapped_reordered_bit_widths) / 2)):
                            if 2 * adder_index != len(self.lookahead_mapped_reordered_bit_widths) - 1:
                                if(self.lookahead_mapped_reordered_bit_widths[2 * adder_index][0] < self.lookahead_mapped_reordered_bit_widths[2 * adder_index + 1][0]):
                                    stage_output_widths.append(self.lookahead_mapped_reordered_bit_widths[2 * adder_index + 1][0] + 1)
                                    lookahead_intermediate_results_register_width += self.lookahead_mapped_reordered_bit_widths[2 * adder_index + 1][0] + 1
                                elif(self.lookahead_mapped_reordered_bit_widths[2 * adder_index][0] >= self.lookahead_mapped_reordered_bit_widths[2 * adder_index + 1][0]):
                                    stage_output_widths.append(self.lookahead_mapped_reordered_bit_widths[2 * adder_index][0] + 1)
                                    lookahead_intermediate_results_register_width += self.lookahead_mapped_reordered_bit_widths[2 * adder_index][0] + 1
                            elif 2 * adder_index == len(self.lookahead_mapped_reordered_bit_widths) - 1:
                                stage_output_widths.append(self.lookahead_mapped_reordered_bit_widths[len(self.lookahead_mapped_reordered_bit_widths) - 1][0])
                                lookahead_intermediate_results_register_width += self.lookahead_mapped_reordered_bit_widths[len(self.lookahead_mapped_reordered_bit_widths) - 1][0]
                    elif stage_index > 1:
                        for adder_index in range(math.ceil(len(stages_output_widths[stage_index - 2]) / 2)):
                            if 2 * adder_index != len(stages_output_widths[stage_index - 2]) - 1:
                                if(stages_output_widths[stage_index - 2][2 * adder_index] < stages_output_widths[stage_index - 2][2 * adder_index + 1]):
                                    stage_output_widths.append(stages_output_widths[stage_index - 2][2 * adder_index + 1] + 1)
                                    lookahead_intermediate_results_register_width += stages_output_widths[stage_index - 2][2 * adder_index + 1] + 1
                                elif(stages_output_widths[stage_index - 2][2 * adder_index] >= stages_output_widths[stage_index - 2][2 * adder_index + 1]):
                                    stage_output_widths.append(stages_output_widths[stage_index - 2][2 * adder_index] + 1)
                                    lookahead_intermediate_results_register_width += stages_output_widths[stage_index - 2][2 * adder_index] + 1
                            elif 2 * adder_index == len(stages_output_widths[stage_index - 2]) - 1:
                                stage_output_widths.append(stages_output_widths[stage_index - 2][len(stages_output_widths[stage_index - 2]) - 1])
                                lookahead_intermediate_results_register_width += stages_output_widths[stage_index - 2][len(stages_output_widths[stage_index - 2]) - 1]
                    stages_output_widths.append(stage_output_widths)                       
                print(stages_output_widths)
                print(lookahead_intermediate_results_register_width)
                
                content += f"""logic [{lookahead_intermediate_results_register_width} - 1 : 0] lookahead_adder_results;
    
    """
                lookahead_total_output_offset: int = 0
                last_stage_output_offset: int = 0
                for stage_index in range(1, lookahead_adder_stage_count + 1, 1):
                    if stage_index == 1:
                        input_offset: int = 0
                        stage_output_offset: int = 0
                        for adder_index in range(math.ceil(len(self.lookahead_mapped_reordered_bit_widths) / 2)):
                            if 2 * adder_index != len(self.lookahead_mapped_reordered_bit_widths) - 1:
                                content += f"""AdderCombinatorial #(
            .INPUT0_WIDTH({self.lookahead_mapped_reordered_bit_widths[2 * adder_index][0]}),
            .INPUT1_WIDTH({self.lookahead_mapped_reordered_bit_widths[2 * adder_index + 1][0]})
        )
        lookahead_adder_synchronous_{stage_index - 1}_{adder_index}(
            .rst(rst),
            .input_0(lookahead_lookup_table_results[{lookahead_luts_result_register_total_width - input_offset} - 1 : {lookahead_luts_result_register_total_width - (input_offset + self.lookahead_mapped_reordered_bit_widths[2 * adder_index][0])}]),
            .input_1(lookahead_lookup_table_results[{lookahead_luts_result_register_total_width - (input_offset + self.lookahead_mapped_reordered_bit_widths[2 * adder_index][0])} - 1 : {lookahead_luts_result_register_total_width - (input_offset + self.lookahead_mapped_reordered_bit_widths[2 * adder_index][0] + self.lookahead_mapped_reordered_bit_widths[2 * adder_index + 1][0])}]),
            .out(lookahead_adder_results[{stage_output_offset + stages_output_widths[0][adder_index]} - 1 : {stage_output_offset}])
    );
    
    """
                                input_offset += self.lookahead_mapped_reordered_bit_widths[2 * adder_index][0] + self.lookahead_mapped_reordered_bit_widths[2 * adder_index + 1][0]
                                stage_output_offset += stages_output_widths[0][adder_index]
                            elif 2 * adder_index == len(self.lookahead_mapped_reordered_bit_widths) - 1:
                                content += f""" assign lookahead_adder_results[{stage_output_offset + stages_output_widths[0][adder_index]} - 1 : {stage_output_offset}] = lookahead_lookup_table_results[{lookahead_luts_result_register_total_width - input_offset} - 1 : {lookahead_luts_result_register_total_width - (input_offset + self.lookahead_mapped_reordered_bit_widths[2 * adder_index][0])}];
    
    """
                                input_offset += self.lookahead_mapped_reordered_bit_widths[2 * adder_index][0]
                                stage_output_offset += stages_output_widths[0][adder_index]
                        lookahead_total_output_offset += stage_output_offset
                        last_stage_output_offset = stage_output_offset
                    elif stage_index > 1:
                        input_offset: int = 0
                        stage_output_offset: int = 0
                        for adder_index in range(math.ceil(len(stages_output_widths[stage_index - 2]) / 2)):
                            if 2 * adder_index != len(stages_output_widths[stage_index - 2]) - 1:
                                content += f"""AdderCombinatorial #(
            .INPUT0_WIDTH({stages_output_widths[stage_index - 2][2 * adder_index]}),
            .INPUT1_WIDTH({stages_output_widths[stage_index - 2][2 * adder_index + 1]})
        )
        lookahead_adder_synchronous_{stage_index - 1}_{adder_index}(
            .rst(rst),
            .input_0(lookahead_adder_results[{(lookahead_total_output_offset - last_stage_output_offset) + input_offset + stages_output_widths[stage_index - 2][2 * adder_index]} - 1 : {(lookahead_total_output_offset - last_stage_output_offset) + input_offset}]),
            .input_1(lookahead_adder_results[{(lookahead_total_output_offset - last_stage_output_offset) + input_offset + stages_output_widths[stage_index - 2][2 * adder_index] + stages_output_widths[stage_index - 2][2 * adder_index + 1]} - 1 : {(lookahead_total_output_offset - last_stage_output_offset) + input_offset + stages_output_widths[stage_index - 2][2 * adder_index]}]),
            .out(lookahead_adder_results[{lookahead_total_output_offset + stage_output_offset + stages_output_widths[stage_index - 1][adder_index]} - 1 : {lookahead_total_output_offset + stage_output_offset}])
    );
    
    """
                                input_offset += stages_output_widths[stage_index - 2][2 * adder_index] + stages_output_widths[stage_index - 2][2 * adder_index + 1]
                                stage_output_offset += stages_output_widths[stage_index - 1][adder_index]
                            elif 2 * adder_index == len(stages_output_widths[stage_index - 2]) - 1:
                                content += f"""assign lookahead_adder_results[{lookahead_total_output_offset + stage_output_offset + stages_output_widths[stage_index - 1][adder_index]} - 1 : {lookahead_total_output_offset + stage_output_offset}] = lookahead_adder_results[{(lookahead_total_output_offset - last_stage_output_offset) + input_offset + stages_output_widths[stage_index - 2][2 * adder_index]} - 1 : {(lookahead_total_output_offset - last_stage_output_offset) + input_offset}];
    
    """
                                input_offset += stages_output_widths[stage_index - 2][2 * adder_index]
                                stage_output_offset += stages_output_widths[stage_index - 1][adder_index]
                        lookahead_total_output_offset += stage_output_offset
                        last_stage_output_offset = stage_output_offset
                content += f"""logic signed [{stages_output_widths[len(stages_output_widths) - 1][0]} - 1 : 0] adder_block_lookahead_result;
    assign adder_block_lookahead_result = lookahead_adder_results[{lookahead_intermediate_results_register_width} - 1 : {lookahead_intermediate_results_register_width - stages_output_widths[len(stages_output_widths) - 1][0]}];
    
    """
            
        elif self.configuration_combinatorial_synchronous == "synchronous":
            if self.configuration_reduce_size_coefficients == False:
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
            elif self.configuration_reduce_size_coefficients == True:
                for lookback_lut_index in range(len(self.lookback_mapped_reordered_lut_entries)):
                    lut_offset: int = 0
                    result_offset: int = 0
                    for lut_index in range(len(self.lookback_mapped_reordered_bit_widths) - 1, -1, -1):
                        if lut_index == lookback_lut_index:
                            break
                        lut_offset += len(self.lookback_mapped_reordered_lut_entries[lut_index]) * self.lookback_mapped_reordered_bit_widths[lut_index][0]
                        result_offset += self.lookback_mapped_reordered_bit_widths[lut_index][0]
                    lut_memory_contents: str = "}"
                    for entry_index in range(len(self.lookback_mapped_reordered_lut_entries[lookback_lut_index])):
                        #if self.configuration_reduce_size_luts == True:
                        lut_memory_contents = f", lookback_lookup_table_entries[{lut_offset + (entry_index + 1) * self.lookback_mapped_reordered_bit_widths[lookback_lut_index][0]} - 1 : {lut_offset + entry_index * self.lookback_mapped_reordered_bit_widths[lookback_lut_index][0]}]" + lut_memory_contents
                        if self.configuration_reduce_size_luts == False:
                            if self.lookback_mapped_reordered_lut_entries[lookback_lut_index][len(self.lookback_mapped_reordered_lut_entries[lookback_lut_index]) - 1 - entry_index] >= 0:
                                lut_memory_contents = f", {{{self.configuration_data_width - self.lookback_mapped_reordered_bit_widths[lookback_lut_index][0]}{{1'b0}}}}" + lut_memory_contents
                            elif self.lookback_mapped_reordered_lut_entries[lookback_lut_index][len(self.lookback_mapped_reordered_lut_entries[lookback_lut_index]) - 1 - entry_index] < 0:
                                lut_memory_contents = f", {{{self.configuration_data_width - self.lookback_mapped_reordered_bit_widths[lookback_lut_index][0]}{{1'b1}}}}" + lut_memory_contents
                    lut_memory_contents = "{" + lut_memory_contents.removeprefix(", ")
                    content += f"""LookUpTableSynchronous #(
            .INPUT_WIDTH(LOOKUP_TABLE_INPUT_WIDTH),
            """
                    if self.configuration_reduce_size_luts == True:
                        content += f""".DATA_WIDTH({self.lookback_mapped_reordered_bit_widths[lookback_lut_index][0]})
        """
                    elif self.configuration_reduce_size_luts == False:
                        content += f""".DATA_WIDTH({self.configuration_data_width})
        """
                    content += f""")
        lookback_lookup_table_{self.lookback_mapped_reordered_bit_widths[lookback_lut_index][1]} (
            .rst(rst),
            .clk(clk),
            .in(lookback_register_distribution[({len(self.lookback_mapped_reordered_bit_widths) - 1 - self.lookback_mapped_reordered_bit_widths[lookback_lut_index][1]} * LOOKUP_TABLE_INPUT_WIDTH) +: LOOKUP_TABLE_INPUT_WIDTH]),
            .memory({lut_memory_contents}),
            """
                    if self.configuration_reduce_size_adders == True:
                        content += f""".out(lookback_lookup_table_results[{self.lookback_mapped_reordered_bit_widths[lookback_lut_index][0] + result_offset} - 1 : {result_offset}])
    """
                    elif self.configuration_reduce_size_adders == False:
                        content += f""".out(lookback_lookup_table_results[{lookback_lut_index}])
    """
                    content += f""");
                
    """
            if self.configuration_reduce_size_adders == False:
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
            elif self.configuration_reduce_size_adders == True:
                lookback_lookup_table_count: int = math.ceil(self.configuration_lookback_length * self.configuration_m_number_of_digital_states / self.configuration_fir_lut_input_width)
                lookback_adder_stage_count: int = math.ceil(math.log2(lookback_lookup_table_count))
                #number_of_intermediate_lookback_results: int = 0
                #for stage_index in range(1, lookback_adder_stage_count + 1, 1):
                #    number_of_intermediate_lookback_results += math.ceil(lookback_lookup_table_count / 2**stage_index)
                lookback_intermediate_results_register_width: int = 0
                stages_output_widths: list[list[int]] = list[list[int]]()
                for stage_index in range(1, lookback_adder_stage_count + 1, 1):
                    stage_output_widths: list[int] = list[int]()
                    if stage_index == 1:
                        for adder_index in range(math.ceil(len(self.lookback_mapped_reordered_bit_widths) / 2)):
                            if 2 * adder_index != len(self.lookback_mapped_reordered_bit_widths) - 1:
                                if(self.lookback_mapped_reordered_bit_widths[2 * adder_index][0] < self.lookback_mapped_reordered_bit_widths[2 * adder_index + 1][0]):
                                    stage_output_widths.append(self.lookback_mapped_reordered_bit_widths[2 * adder_index + 1][0] + 1)
                                    lookback_intermediate_results_register_width += self.lookback_mapped_reordered_bit_widths[2 * adder_index + 1][0] + 1
                                elif(self.lookback_mapped_reordered_bit_widths[2 * adder_index][0] >= self.lookback_mapped_reordered_bit_widths[2 * adder_index + 1][0]):
                                    stage_output_widths.append(self.lookback_mapped_reordered_bit_widths[2 * adder_index][0] + 1)
                                    lookback_intermediate_results_register_width += self.lookback_mapped_reordered_bit_widths[2 * adder_index][0] + 1
                            elif 2 * adder_index == len(self.lookback_mapped_reordered_bit_widths) - 1:
                                stage_output_widths.append(self.lookback_mapped_reordered_bit_widths[len(self.lookback_mapped_reordered_bit_widths) - 1][0])
                                lookback_intermediate_results_register_width += self.lookback_mapped_reordered_bit_widths[len(self.lookback_mapped_reordered_bit_widths) - 1][0]
                    elif stage_index > 1:
                        for adder_index in range(math.ceil(len(stages_output_widths[stage_index - 2]) / 2)):
                            if 2 * adder_index != len(stages_output_widths[stage_index - 2]) - 1:
                                if(stages_output_widths[stage_index - 2][2 * adder_index] < stages_output_widths[stage_index - 2][2 * adder_index + 1]):
                                    stage_output_widths.append(stages_output_widths[stage_index - 2][2 * adder_index + 1] + 1)
                                    lookback_intermediate_results_register_width += stages_output_widths[stage_index - 2][2 * adder_index + 1] + 1
                                elif(stages_output_widths[stage_index - 2][2 * adder_index] >= stages_output_widths[stage_index - 2][2 * adder_index + 1]):
                                    stage_output_widths.append(stages_output_widths[stage_index - 2][2 * adder_index] + 1)
                                    lookback_intermediate_results_register_width += stages_output_widths[stage_index - 2][2 * adder_index] + 1
                            elif 2 * adder_index == len(stages_output_widths[stage_index - 2]) - 1:
                                stage_output_widths.append(stages_output_widths[stage_index - 2][len(stages_output_widths[stage_index - 2]) - 1])
                                lookback_intermediate_results_register_width += stages_output_widths[stage_index - 2][len(stages_output_widths[stage_index - 2]) - 1]
                    stages_output_widths.append(stage_output_widths)                       
                print(stages_output_widths)
                print(lookback_intermediate_results_register_width)
                
                content += f"""logic [{lookback_intermediate_results_register_width} - 1 : 0] lookback_adder_results;
    
    """
                lookback_total_output_offset: int = 0
                last_stage_output_offset: int = 0
                for stage_index in range(1, lookback_adder_stage_count + 1, 1):
                    if stage_index == 1:
                        input_offset: int = 0
                        stage_output_offset: int = 0
                        for adder_index in range(math.ceil(len(self.lookback_mapped_reordered_bit_widths) / 2)):
                            if 2 * adder_index != len(self.lookback_mapped_reordered_bit_widths) - 1:
                                content += f"""AdderSynchronous #(
            .INPUT0_WIDTH({self.lookback_mapped_reordered_bit_widths[2 * adder_index][0]}),
            .INPUT1_WIDTH({self.lookback_mapped_reordered_bit_widths[2 * adder_index + 1][0]})
        )
        lookback_adder_synchronous_{stage_index - 1}_{adder_index}(
            .rst(rst),
            .clk(clk_downsample),
            .input_0(lookback_lookup_table_results[{lookback_luts_result_register_total_width - input_offset} - 1 : {lookback_luts_result_register_total_width - (input_offset + self.lookback_mapped_reordered_bit_widths[2 * adder_index][0])}]),
            .input_1(lookback_lookup_table_results[{lookback_luts_result_register_total_width - (input_offset + self.lookback_mapped_reordered_bit_widths[2 * adder_index][0])} - 1 : {lookback_luts_result_register_total_width - (input_offset + self.lookback_mapped_reordered_bit_widths[2 * adder_index][0] + self.lookback_mapped_reordered_bit_widths[2 * adder_index + 1][0])}]),
            .out(lookback_adder_results[{stage_output_offset + stages_output_widths[0][adder_index]} - 1 : {stage_output_offset}])
    );
    
    """
                                input_offset += self.lookback_mapped_reordered_bit_widths[2 * adder_index][0] + self.lookback_mapped_reordered_bit_widths[2 * adder_index + 1][0]
                                stage_output_offset += stages_output_widths[0][adder_index]
                            elif 2 * adder_index == len(self.lookback_mapped_reordered_bit_widths) - 1:
                                content += f"""always_ff @(posedge clk_sample_shift_register) begin
        lookback_adder_results[{stage_output_offset + stages_output_widths[0][adder_index]} - 1 : {stage_output_offset}] <= lookback_lookup_table_results[{lookback_luts_result_register_total_width - input_offset} - 1 : {lookback_luts_result_register_total_width - (input_offset + self.lookback_mapped_reordered_bit_widths[2 * adder_index][0])}];
    end
    
    """
                                input_offset += self.lookback_mapped_reordered_bit_widths[2 * adder_index][0]
                                stage_output_offset += stages_output_widths[0][adder_index]
                        lookback_total_output_offset += stage_output_offset
                        last_stage_output_offset = stage_output_offset
                    elif stage_index > 1:
                        input_offset: int = 0
                        stage_output_offset: int = 0
                        for adder_index in range(math.ceil(len(stages_output_widths[stage_index - 2]) / 2)):
                            if 2 * adder_index != len(stages_output_widths[stage_index - 2]) - 1:
                                content += f"""AdderSynchronous #(
            .INPUT0_WIDTH({stages_output_widths[stage_index - 2][2 * adder_index]}),
            .INPUT1_WIDTH({stages_output_widths[stage_index - 2][2 * adder_index + 1]})
        )
        lookback_adder_synchronous_{stage_index - 1}_{adder_index}(
            .rst(rst),
            .clk(clk_downsample),
            .input_0(lookback_adder_results[{(lookback_total_output_offset - last_stage_output_offset) + input_offset + stages_output_widths[stage_index - 2][2 * adder_index]} - 1 : {(lookback_total_output_offset - last_stage_output_offset) + input_offset}]),
            .input_1(lookback_adder_results[{(lookback_total_output_offset - last_stage_output_offset) + input_offset + stages_output_widths[stage_index - 2][2 * adder_index] + stages_output_widths[stage_index - 2][2 * adder_index + 1]} - 1 : {(lookback_total_output_offset - last_stage_output_offset) + input_offset + stages_output_widths[stage_index - 2][2 * adder_index]}]),
            .out(lookback_adder_results[{lookback_total_output_offset + stage_output_offset + stages_output_widths[stage_index - 1][adder_index]} - 1 : {lookback_total_output_offset + stage_output_offset}])
    );
    
    """
                                input_offset += stages_output_widths[stage_index - 2][2 * adder_index] + stages_output_widths[stage_index - 2][2 * adder_index + 1]
                                stage_output_offset += stages_output_widths[stage_index - 1][adder_index]
                            elif 2 * adder_index == len(stages_output_widths[stage_index - 2]) - 1:
                                content += f"""always_ff @(posedge clk_sample_shift_register) begin
        lookback_adder_results[{lookback_total_output_offset + stage_output_offset + stages_output_widths[stage_index - 1][adder_index]} - 1 : {lookback_total_output_offset + stage_output_offset}] <= lookback_adder_results[{(lookback_total_output_offset - last_stage_output_offset) + input_offset + stages_output_widths[stage_index - 2][2 * adder_index]} - 1 : {(lookback_total_output_offset - last_stage_output_offset) + input_offset}];
    end
    
    """
                                input_offset += stages_output_widths[stage_index - 2][2 * adder_index]
                                stage_output_offset += stages_output_widths[stage_index - 1][adder_index]
                        lookback_total_output_offset += stage_output_offset
                        last_stage_output_offset = stage_output_offset
                content += f"""logic signed [{stages_output_widths[len(stages_output_widths) - 1][0]} - 1 : 0] adder_block_lookback_result;
    assign adder_block_lookback_result = lookback_adder_results[{lookback_intermediate_results_register_width} - 1 : {lookback_intermediate_results_register_width - stages_output_widths[len(stages_output_widths) - 1][0]}];
    
    """

            if self.configuration_reduce_size_coefficients == False:
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
            elif self.configuration_reduce_size_coefficients == True:
                for lookahead_lut_index in range(len(self.lookahead_mapped_reordered_lut_entries)):
                    lut_offset: int = 0
                    result_offset: int = 0
                    for lut_index in range(len(self.lookahead_mapped_reordered_bit_widths) - 1, -1, -1):
                        if lut_index == lookahead_lut_index:
                            break
                        lut_offset += len(self.lookahead_mapped_reordered_lut_entries[lut_index]) * self.lookahead_mapped_reordered_bit_widths[lut_index][0]
                        result_offset += self.lookback_mapped_reordered_bit_widths[lut_index][0]
                    lut_memory_contents: str = "}"
                    for entry_index in range(len(self.lookahead_mapped_reordered_lut_entries[lookahead_lut_index])):
                        #if self.configuration_reduce_size_luts == True:
                        lut_memory_contents = f", lookahead_lookup_table_entries[{lut_offset + (entry_index + 1) * self.lookahead_mapped_reordered_bit_widths[lookahead_lut_index][0]} - 1 : {lut_offset + entry_index * self.lookahead_mapped_reordered_bit_widths[lookahead_lut_index][0]}]" + lut_memory_contents
                        if self.configuration_reduce_size_luts == False:
                            if self.lookahead_mapped_reordered_lut_entries[lookahead_lut_index][len(self.lookahead_mapped_reordered_lut_entries[lookahead_lut_index]) - 1 - entry_index] >= 0:
                                lut_memory_contents = f", {{{self.configuration_data_width - self.lookahead_mapped_reordered_bit_widths[lookahead_lut_index][0]}{{1'b0}}}}" + lut_memory_contents
                            elif self.lookahead_mapped_reordered_lut_entries[lookahead_lut_index][len(self.lookahead_mapped_reordered_lut_entries[lookahead_lut_index]) - 1 - entry_index] < 0:
                                lut_memory_contents = f", {{{self.configuration_data_width - self.lookahead_mapped_reordered_bit_widths[lookahead_lut_index][0]}{{1'b1}}}}" + lut_memory_contents
                    lut_memory_contents = "{" + lut_memory_contents.removeprefix(", ")
                    content += f"""LookUpTableSynchronous #(
            .INPUT_WIDTH(LOOKUP_TABLE_INPUT_WIDTH),
            """
                    if self.configuration_reduce_size_luts == True:
                        content += f""".DATA_WIDTH({self.lookahead_mapped_reordered_bit_widths[lut_index][0]})
        """
                    elif self.configuration_reduce_size_luts == False:
                        content += f""".DATA_WIDTH({self.configuration_data_width})
        """
                    content += f""")
        lookahead_lookup_table_{self.lookahead_mapped_reordered_bit_widths[lookahead_lut_index][1]} (
            .rst(rst),
            .clk(clk),
            .in(lookahead_register_distribution[({len(self.lookahead_mapped_reordered_bit_widths) - 1 - self.lookahead_mapped_reordered_bit_widths[lookahead_lut_index][1]} * LOOKUP_TABLE_INPUT_WIDTH) +: LOOKUP_TABLE_INPUT_WIDTH]),
            .memory({lut_memory_contents}),
            """
                    if self.configuration_reduce_size_adders == True:
                        content += f""".out(lookahead_lookup_table_results[{self.lookahead_mapped_reordered_bit_widths[lookahead_lut_index][0] + result_offset} - 1 : {result_offset}])
    """
                    elif self.configuration_reduce_size_adders == False:
                        content += f""".out(lookahead_lookup_table_results[{lookahead_lut_index}])
    """
                    content += f""");
                
    """
            if self.configuration_reduce_size_adders == False:
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
            elif self.configuration_reduce_size_adders == True:
                lookahead_lookup_table_count: int = math.ceil(self.configuration_lookahead_length * self.configuration_m_number_of_digital_states / self.configuration_fir_lut_input_width)
                lookahead_adder_stage_count: int = math.ceil(math.log2(lookahead_lookup_table_count))
                #number_of_intermediate_lookahead_results: int = 0
                #for stage_index in range(1, lookahead_adder_stage_count + 1, 1):
                #    number_of_intermediate_lookahead_results += math.ceil(lookahead_lookup_table_count / 2**stage_index)
                lookahead_intermediate_results_register_width: int = 0
                stages_output_widths: list[list[int]] = list[list[int]]()
                for stage_index in range(1, lookahead_adder_stage_count + 1, 1):
                    stage_output_widths: list[int] = list[int]()
                    if stage_index == 1:
                        for adder_index in range(math.ceil(len(self.lookahead_mapped_reordered_bit_widths) / 2)):
                            if 2 * adder_index != len(self.lookahead_mapped_reordered_bit_widths) - 1:
                                if(self.lookahead_mapped_reordered_bit_widths[2 * adder_index][0] < self.lookahead_mapped_reordered_bit_widths[2 * adder_index + 1][0]):
                                    stage_output_widths.append(self.lookahead_mapped_reordered_bit_widths[2 * adder_index + 1][0] + 1)
                                    lookahead_intermediate_results_register_width += self.lookahead_mapped_reordered_bit_widths[2 * adder_index + 1][0] + 1
                                elif(self.lookahead_mapped_reordered_bit_widths[2 * adder_index][0] >= self.lookahead_mapped_reordered_bit_widths[2 * adder_index + 1][0]):
                                    stage_output_widths.append(self.lookahead_mapped_reordered_bit_widths[2 * adder_index][0] + 1)
                                    lookahead_intermediate_results_register_width += self.lookahead_mapped_reordered_bit_widths[2 * adder_index][0] + 1
                            elif 2 * adder_index == len(self.lookahead_mapped_reordered_bit_widths) - 1:
                                stage_output_widths.append(self.lookahead_mapped_reordered_bit_widths[len(self.lookahead_mapped_reordered_bit_widths) - 1][0])
                                lookahead_intermediate_results_register_width += self.lookahead_mapped_reordered_bit_widths[len(self.lookahead_mapped_reordered_bit_widths) - 1][0]
                    elif stage_index > 1:
                        for adder_index in range(math.ceil(len(stages_output_widths[stage_index - 2]) / 2)):
                            if 2 * adder_index != len(stages_output_widths[stage_index - 2]) - 1:
                                if(stages_output_widths[stage_index - 2][2 * adder_index] < stages_output_widths[stage_index - 2][2 * adder_index + 1]):
                                    stage_output_widths.append(stages_output_widths[stage_index - 2][2 * adder_index + 1] + 1)
                                    lookahead_intermediate_results_register_width += stages_output_widths[stage_index - 2][2 * adder_index + 1] + 1
                                elif(stages_output_widths[stage_index - 2][2 * adder_index] >= stages_output_widths[stage_index - 2][2 * adder_index + 1]):
                                    stage_output_widths.append(stages_output_widths[stage_index - 2][2 * adder_index] + 1)
                                    lookahead_intermediate_results_register_width += stages_output_widths[stage_index - 2][2 * adder_index] + 1
                            elif 2 * adder_index == len(stages_output_widths[stage_index - 2]) - 1:
                                stage_output_widths.append(stages_output_widths[stage_index - 2][len(stages_output_widths[stage_index - 2]) - 1])
                                lookahead_intermediate_results_register_width += stages_output_widths[stage_index - 2][len(stages_output_widths[stage_index - 2]) - 1]
                    stages_output_widths.append(stage_output_widths)                       
                print(stages_output_widths)
                print(lookahead_intermediate_results_register_width)
                
                content += f"""logic [{lookahead_intermediate_results_register_width} - 1 : 0] lookahead_adder_results;
    
    """
                lookahead_total_output_offset: int = 0
                last_stage_output_offset: int = 0
                for stage_index in range(1, lookahead_adder_stage_count + 1, 1):
                    if stage_index == 1:
                        input_offset: int = 0
                        stage_output_offset: int = 0
                        for adder_index in range(math.ceil(len(self.lookahead_mapped_reordered_bit_widths) / 2)):
                            if 2 * adder_index != len(self.lookahead_mapped_reordered_bit_widths) - 1:
                                content += f"""AdderSynchronous #(
            .INPUT0_WIDTH({self.lookahead_mapped_reordered_bit_widths[2 * adder_index][0]}),
            .INPUT1_WIDTH({self.lookahead_mapped_reordered_bit_widths[2 * adder_index + 1][0]})
        )
        lookahead_adder_synchronous_{stage_index - 1}_{adder_index}(
            .rst(rst),
            .clk(clk_downsample),
            .input_0(lookahead_lookup_table_results[{lookahead_luts_result_register_total_width - input_offset} - 1 : {lookahead_luts_result_register_total_width - (input_offset + self.lookahead_mapped_reordered_bit_widths[2 * adder_index][0])}]),
            .input_1(lookahead_lookup_table_results[{lookahead_luts_result_register_total_width - (input_offset + self.lookahead_mapped_reordered_bit_widths[2 * adder_index][0])} - 1 : {lookahead_luts_result_register_total_width - (input_offset + self.lookahead_mapped_reordered_bit_widths[2 * adder_index][0] + self.lookahead_mapped_reordered_bit_widths[2 * adder_index + 1][0])}]),
            .out(lookahead_adder_results[{stage_output_offset + stages_output_widths[0][adder_index]} - 1 : {stage_output_offset}])
    );
    
    """
                                input_offset += self.lookahead_mapped_reordered_bit_widths[2 * adder_index][0] + self.lookahead_mapped_reordered_bit_widths[2 * adder_index + 1][0]
                                stage_output_offset += stages_output_widths[0][adder_index]
                            elif 2 * adder_index == len(self.lookahead_mapped_reordered_bit_widths) - 1:
                                content += f"""always_ff @(posedge clk_sample_shift_register) begin
        lookahead_adder_results[{stage_output_offset + stages_output_widths[0][adder_index]} - 1 : {stage_output_offset}] <= lookahead_lookup_table_results[{lookahead_luts_result_register_total_width - input_offset} - 1 : {lookahead_luts_result_register_total_width - (input_offset + self.lookahead_mapped_reordered_bit_widths[2 * adder_index][0])}];
    end
    
    """
                                input_offset += self.lookahead_mapped_reordered_bit_widths[2 * adder_index][0]
                                stage_output_offset += stages_output_widths[0][adder_index]
                        lookahead_total_output_offset += stage_output_offset
                        last_stage_output_offset = stage_output_offset
                    elif stage_index > 1:
                        input_offset: int = 0
                        stage_output_offset: int = 0
                        for adder_index in range(math.ceil(len(stages_output_widths[stage_index - 2]) / 2)):
                            if 2 * adder_index != len(stages_output_widths[stage_index - 2]) - 1:
                                content += f"""AdderSynchronous #(
            .INPUT0_WIDTH({stages_output_widths[stage_index - 2][2 * adder_index]}),
            .INPUT1_WIDTH({stages_output_widths[stage_index - 2][2 * adder_index + 1]})
        )
        lookahead_adder_synchronous_{stage_index - 1}_{adder_index}(
            .rst(rst),
            .clk(clk_downsample),
            .input_0(lookahead_adder_results[{(lookahead_total_output_offset - last_stage_output_offset) + input_offset + stages_output_widths[stage_index - 2][2 * adder_index]} - 1 : {(lookahead_total_output_offset - last_stage_output_offset) + input_offset}]),
            .input_1(lookahead_adder_results[{(lookahead_total_output_offset - last_stage_output_offset) + input_offset + stages_output_widths[stage_index - 2][2 * adder_index] + stages_output_widths[stage_index - 2][2 * adder_index + 1]} - 1 : {(lookahead_total_output_offset - last_stage_output_offset) + input_offset + stages_output_widths[stage_index - 2][2 * adder_index]}]),
            .out(lookahead_adder_results[{lookahead_total_output_offset + stage_output_offset + stages_output_widths[stage_index - 1][adder_index]} - 1 : {lookahead_total_output_offset + stage_output_offset}])
    );
    
    """
                                input_offset += stages_output_widths[stage_index - 2][2 * adder_index] + stages_output_widths[stage_index - 2][2 * adder_index + 1]
                                stage_output_offset += stages_output_widths[stage_index - 1][adder_index]
                            elif 2 * adder_index == len(stages_output_widths[stage_index - 2]) - 1:
                                content += f"""always_ff @(posedge clk_sample_shift_register) begin
        lookahead_adder_results[{lookahead_total_output_offset + stage_output_offset + stages_output_widths[stage_index - 1][adder_index]} - 1 : {lookahead_total_output_offset + stage_output_offset}] <= lookahead_adder_results[{(lookahead_total_output_offset - last_stage_output_offset) + input_offset + stages_output_widths[stage_index - 2][2 * adder_index]} - 1 : {(lookahead_total_output_offset - last_stage_output_offset) + input_offset}];
    end
    
    """
                                input_offset += stages_output_widths[stage_index - 2][2 * adder_index]
                                stage_output_offset += stages_output_widths[stage_index - 1][adder_index]
                        lookahead_total_output_offset += stage_output_offset
                        last_stage_output_offset = stage_output_offset
                content += f"""logic signed [{stages_output_widths[len(stages_output_widths) - 1][0]} - 1 : 0] adder_block_lookahead_result;
    assign adder_block_lookahead_result = lookahead_adder_results[{lookahead_intermediate_results_register_width} - 1 : {lookahead_intermediate_results_register_width - stages_output_widths[len(stages_output_widths) - 1][0]}];
    
    """
    
        if self.configuration_combinatorial_synchronous == "combinatorial":
            content += """assign signal_estimation_output = adder_block_lookback_result + adder_block_lookahead_result;
    
    """
        elif self.configuration_combinatorial_synchronous == "synchronous":
            content += """always_ff @(posedge clk_sample_shift_register) begin
        signal_estimation_output = adder_block_lookback_result + adder_block_lookahead_result;
    end
    
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