from tkinter.messagebox import NO
import SystemVerilogModule
import SystemVerilogPort
import SystemVerilogPortDirection
import SystemVerilogPortType
import SystemVerilogDimension
import DigitalEstimatorModules.TestModule
import DigitalEstimatorModules.SimpleAdder
import DigitalEstimatorModules.LookUpTable
import SystemVerilogSignalSign
from SystemVerilogSyntaxGenerator import decimal_number, get_parameter_value, connect_port_array, ndarray_to_system_verilog_array, set_parameter_value
import CBADC_HighLevelSimulation
import cbadc
import numpy


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

    high_level_simulation: CBADC_HighLevelSimulation.DigitalEstimatorParameterGenerator

    def __init__(self, path: str, name: str):
        super().__init__(path, name)

    def generate(self):
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
        input wire signed [LOOKUP_TABLE_DATA_WIDTH - 1 : 0] lookup_table_coefficient,
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
    wire signed [LOOKBACK_LOOKUP_TABLE_ENTRIES_COUNT - 1 : 0][LOOKUP_TABLE_DATA_WIDTH - 1 : 0] lookback_lookup_table_entries;
    wire signed [LOOKAHEAD_LOOKUP_TABLE_ENTRIES_COUNT - 1 : 0][LOOKUP_TABLE_DATA_WIDTH - 1 : 0] lookahead_lookup_table_entries;
    
    """
        if self.configuration_down_sample_rate > 1:
            content += f"""assign clk_sample_shift_register = clk_downsample;
    
    """
        else:
            content += f"""assign clk_sample_shift_register = clk;
    
    """
        if self.configuration_coefficients_variable_fixed == "fixed":
            content += "assign lookback_lookup_table_entries = "
            content += ndarray_to_system_verilog_array(numpy.array(CBADC_HighLevelSimulation.convert_coefficient_matrix_to_lut_entries(self.high_level_simulation.get_fir_lookback_coefficient_matrix(), self.configuration_fir_lut_input_width))) + ";\n\t\t"
            content += "assign lookahead_lookup_table_entries = "
            content += ndarray_to_system_verilog_array(numpy.array(CBADC_HighLevelSimulation.convert_coefficient_matrix_to_lut_entries(self.high_level_simulation.get_fir_lookahead_coefficient_matrix(), self.configuration_fir_lut_input_width))) + ";\n"
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

    assign lookback_register = sample_shift_register[TOTAL_LOOKUP_REGISTER_LENGTH - 1 : LOOKAHEAD_SIZE];
    //assign lookback_register = sample_shift_register[LOOKBACK_SIZE - 1 : 0];

    logic [LOOKAHEAD_SIZE - 1 : 0][M_NUMBER_DIGITAL_STATES - 1 : 0] lookahead_register;

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
            .LOOKUP_TABLE_DATA_WIDTH(LOOKUP_TABLE_DATA_WIDTH),
            .LOOKBACK_LOOKUP_TABLE_ENTRIES_COUNT(LOOKBACK_LOOKUP_TABLE_ENTRIES_COUNT),
            .LOOKAHEAD_LOOKUP_TABLE_ENTRIES_COUNT(LOOKAHEAD_LOOKUP_TABLE_ENTRIES_COUNT)
        )
        lookup_table_coefficient_register (
            .rst(rst),
            .clk(clk),
            .enable_input(enable_lookup_table_coefficient_shift_in),
            .coefficient_in(lookup_table_coefficient),
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

    AdderBlockSynchronous #(
            .INPUT_COUNT(LOOKBACK_LOOKUP_TABLE_COUNT),
            .INPUT_WIDTH(LOOKUP_TABLE_DATA_WIDTH)
        )
        adder_block_lookback (
            .rst(rst),
            .clk(clk_downsample),
            .in(lookback_lookup_table_results),
            .out(adder_block_lookback_result)
    );

    LookUpTableBlockSynchronous #(
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

    AdderBlockSynchronous #(
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