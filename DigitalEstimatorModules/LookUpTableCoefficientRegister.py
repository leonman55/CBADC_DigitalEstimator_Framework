import SystemVerilogModule


class LookUpTableCoefficientRegister(SystemVerilogModule.SystemVerilogModule):
    configuration_lookup_table_data_width: int = 31
    configuration_lookback_lookup_table_entries_count: int = 512
    configuration_lookahead_lookup_table_entries_count: int = 512
    configuration_reduce_size: bool = False

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
        if self.configuration_reduce_size == False:
            content: str = f"""module LookUpTableCoefficientRegister #(
        parameter LOOKUP_TABLE_DATA_WIDTH = {self.configuration_lookup_table_data_width},
        parameter LOOKBACK_LOOKUP_TABLE_ENTRIES_COUNT = {self.configuration_lookback_lookup_table_entries_count},
        parameter LOOKAHEAD_LOOKUP_TABLE_ENTRIES_COUNT = {self.configuration_lookahead_lookup_table_entries_count}
    ) (
        input wire rst,
        input wire clk,
        input wire enable_input,
        input wire [LOOKUP_TABLE_DATA_WIDTH - 1 : 0] coefficient_in,
        output logic [LOOKBACK_LOOKUP_TABLE_ENTRIES_COUNT - 1 : 0][LOOKUP_TABLE_DATA_WIDTH - 1 : 0] lookback_coefficients,
        output logic [LOOKAHEAD_LOOKUP_TABLE_ENTRIES_COUNT - 1 : 0][LOOKUP_TABLE_DATA_WIDTH - 1 : 0] lookahead_coefficients
);

    always_ff @(posedge clk) begin
        if(rst) begin
            lookback_coefficients <= {{LOOKBACK_LOOKUP_TABLE_ENTRIES_COUNT * LOOKUP_TABLE_DATA_WIDTH{{1'b0}}}};
            lookahead_coefficients <= {{LOOKAHEAD_LOOKUP_TABLE_ENTRIES_COUNT * LOOKUP_TABLE_DATA_WIDTH{{1'b0}}}};
        end
        else begin
            if(enable_input) begin
                lookback_coefficients <= {{lookback_coefficients[LOOKBACK_LOOKUP_TABLE_ENTRIES_COUNT - 2 : 0], coefficient_in}};
                lookahead_coefficients <= {{lookahead_coefficients[LOOKAHEAD_LOOKUP_TABLE_ENTRIES_COUNT - 2 : 0], lookback_coefficients[LOOKBACK_LOOKUP_TABLE_ENTRIES_COUNT - 1]}};
            end
        end
    end


endmodule"""

        elif self.configuration_reduce_size == True:
            if self.lookback_mapped_reordered_bit_widths < self.lookahead_mapped_reordered_bit_widths:
                maximum_data_width = self.lookahead_mapped_reordered_bit_widths[len(self.lookahead_mapped_reordered_bit_widths) - 1][0]
            else:
                maximum_data_width = self.lookback_mapped_reordered_bit_widths[len(self.lookback_mapped_reordered_bit_widths) - 1][0]
            for index in range(len(self.lookback_mapped_reordered_lut_entries)):
                lookback_register_total_width += len(self.lookback_mapped_reordered_lut_entries[index]) * self.lookback_mapped_reordered_bit_widths[index][0]
            for index in range(len(self.lookahead_mapped_reordered_lut_entries)):
                lookahead_register_total_width += len(self.lookahead_mapped_reordered_lut_entries[index]) * self.lookahead_mapped_reordered_bit_widths[index][0]
            content: str = f"""module LookUpTableCoefficientRegister #(
        parameter COEFFICIENT_INPUT_DATA_WIDTH = {maximum_data_width},
        parameter LOOKBACK_REGISTER_TOTAL_WIDTH = {lookback_register_total_width},
        parameter LOOKAHEAD_REGISTER_TOTAL_WIDTH = {lookahead_register_total_width}
    ) (
        input wire rst,
        input wire clk,
        input wire enable_input,
        input wire lookback_lookahead_switch,
        input wire [COEFFICIENT_INPUT_DATA_WIDTH - 1 : 0] coefficient_in,
        output logic [LOOKBACK_REGISTER_TOTAL_WIDTH - 1 : 0] lookback_coefficients,
        output logic [LOOKAHEAD_REGISTER_TOTAL_WIDTH - 1 : 0] lookahead_coefficients
);

    always_ff @(posedge clk) begin
        if(rst) begin
            lookback_coefficients <= {{LOOKBACK_REGISTER_TOTAL_WIDTH{{1'b0}}}};
            lookahead_coefficients <= {{LOOKAHEAD_REGISTER_TOTAL_WIDTH{{1'b0}}}};
        end
        else begin
            if(enable_input) begin
                if(lookback_lookahead_switch == 1'b0) begin"""
            lookback_coefficient_offset_target_top: int = lookback_register_total_width - 1
            lookback_coefficient_offset_target_bottom: int = lookback_coefficient_offset_target_top - self.lookback_mapped_reordered_bit_widths[len(self.lookback_mapped_reordered_bit_widths) - 1][0] + 1
            lookback_coefficient_offset_source_top: int = 0
            lookback_coefficient_offset_source_bottom: int = 0
            for lut_index in range(len(self.lookback_mapped_reordered_lut_entries)):
                for entry_index in range(len(self.lookback_mapped_reordered_lut_entries[len(self.lookback_mapped_reordered_lut_entries) - 1 - lut_index])):
                    if lut_index == 0 and entry_index == 0:
                        content += f"""
                    lookback_coefficients[{lookback_register_total_width - 1 - lookback_coefficient_offset_target_bottom} : {lookback_register_total_width - 1 - lookback_coefficient_offset_target_top}] <= coefficient_in[{self.lookback_mapped_reordered_bit_widths[len(self.lookback_mapped_reordered_bit_widths) - 1 - lut_index][0] - 1} : 0];"""
                    elif entry_index == 0:
                        #lookback_coefficient_offset_source_top = lookback_coefficient_offset_target_bottom + self.lookback_mapped_reordered_bit_widths[len(self.lookback_mapped_reordered_bit_widths) - 1 - lut_index][0] - 1
                        #lookback_coefficient_offset_source_bottom = lookback_coefficient_offset_target_bottom
                        lookback_coefficient_offset_source_top = lookback_coefficient_offset_target_top
                        lookback_coefficient_offset_source_bottom = lookback_coefficient_offset_target_top - self.lookback_mapped_reordered_bit_widths[len(self.lookback_mapped_reordered_bit_widths) - 1 - lut_index][0] + 1
                        lookback_coefficient_offset_target_top = lookback_coefficient_offset_source_top - self.lookback_mapped_reordered_bit_widths[len(self.lookback_mapped_reordered_bit_widths) - lut_index][0]
                        lookback_coefficient_offset_target_bottom = lookback_coefficient_offset_source_bottom - self.lookback_mapped_reordered_bit_widths[len(self.lookback_mapped_reordered_bit_widths) - lut_index][0]
                        content += f"""
                    lookback_coefficients[{lookback_register_total_width - 1 - lookback_coefficient_offset_target_bottom} : {lookback_register_total_width - 1 - lookback_coefficient_offset_target_top}] <= lookback_coefficients[{lookback_register_total_width - 1 - lookback_coefficient_offset_source_bottom} : {lookback_register_total_width - 1 - lookback_coefficient_offset_source_top}];"""
                    else:
                        lookback_coefficient_offset_source_top = lookback_coefficient_offset_target_top
                        lookback_coefficient_offset_source_bottom = lookback_coefficient_offset_target_bottom
                        lookback_coefficient_offset_target_top -= self.lookback_mapped_reordered_bit_widths[len(self.lookback_mapped_reordered_bit_widths) - 1 - lut_index][0]
                        lookback_coefficient_offset_target_bottom -= self.lookback_mapped_reordered_bit_widths[len(self.lookback_mapped_reordered_bit_widths) - 1 - lut_index][0]
                        content += f"""
                    lookback_coefficients[{lookback_register_total_width - 1 - lookback_coefficient_offset_target_bottom} : {lookback_register_total_width - 1 - lookback_coefficient_offset_target_top}] <= lookback_coefficients[{lookback_register_total_width - 1 - lookback_coefficient_offset_source_bottom} : {lookback_register_total_width - 1 - lookback_coefficient_offset_source_top}];"""
            content += """
                end
                else begin"""
            lookahead_coefficient_offset_target_top: int = lookahead_register_total_width - 1
            lookahead_coefficient_offset_target_bottom: int = lookahead_coefficient_offset_target_top - self.lookahead_mapped_reordered_bit_widths[len(self.lookahead_mapped_reordered_bit_widths) - 1][0] + 1
            lookahead_coefficient_offset_source_top: int = 0
            lookahead_coefficient_offset_source_bottom: int = 0
            for lut_index in range(len(self.lookahead_mapped_reordered_lut_entries)):
                for entry_index in range(len(self.lookahead_mapped_reordered_lut_entries[len(self.lookahead_mapped_reordered_lut_entries) - 1 - lut_index])):
                    if lut_index == 0 and entry_index == 0:
                        content += f"""
                    lookahead_coefficients[{lookahead_register_total_width - 1 - lookahead_coefficient_offset_target_bottom} : {lookahead_register_total_width - 1 - lookahead_coefficient_offset_target_top}] <= coefficient_in[{self.lookahead_mapped_reordered_bit_widths[len(self.lookahead_mapped_reordered_bit_widths) - 1 - lut_index][0] - 1} : 0];"""
                    elif entry_index == 0:
                        #lookahead_coefficient_offset_source_top = lookahead_coefficient_offset_target_bottom + self.lookahead_mapped_reordered_bit_widths[len(self.lookahead_mapped_reordered_bit_widths) - 1 - lut_index][0] - 1
                        #lookahead_coefficient_offset_source_bottom = lookahead_coefficient_offset_target_bottom
                        lookahead_coefficient_offset_source_top = lookahead_coefficient_offset_target_top
                        lookahead_coefficient_offset_source_bottom = lookahead_coefficient_offset_target_top - self.lookback_mapped_reordered_bit_widths[len(self.lookahead_mapped_reordered_bit_widths) - 1 - lut_index][0] + 1
                        lookahead_coefficient_offset_target_top = lookahead_coefficient_offset_source_top - self.lookahead_mapped_reordered_bit_widths[len(self.lookahead_mapped_reordered_bit_widths) - lut_index][0]
                        lookahead_coefficient_offset_target_bottom = lookahead_coefficient_offset_source_bottom - self.lookahead_mapped_reordered_bit_widths[len(self.lookahead_mapped_reordered_bit_widths) - lut_index][0]
                        content += f"""
                    lookahead_coefficients[{lookahead_register_total_width - 1 - lookahead_coefficient_offset_target_bottom} : {lookahead_register_total_width - 1 - lookahead_coefficient_offset_target_top}] <= lookahead_coefficients[{lookahead_register_total_width - 1 - lookahead_coefficient_offset_source_bottom} : {lookahead_register_total_width - 1 - lookahead_coefficient_offset_source_top}];"""
                    else:
                        lookahead_coefficient_offset_source_top = lookahead_coefficient_offset_target_top
                        lookahead_coefficient_offset_source_bottom = lookahead_coefficient_offset_target_bottom
                        lookahead_coefficient_offset_target_top -= self.lookahead_mapped_reordered_bit_widths[len(self.lookahead_mapped_reordered_bit_widths) - 1 - lut_index][0]
                        lookahead_coefficient_offset_target_bottom -= self.lookahead_mapped_reordered_bit_widths[len(self.lookahead_mapped_reordered_bit_widths) - 1 - lut_index][0]
                        content += f"""
                    lookahead_coefficients[{lookahead_register_total_width - 1 - lookahead_coefficient_offset_target_bottom} : {lookahead_register_total_width - 1 - lookahead_coefficient_offset_target_top}] <= lookahead_coefficients[{lookahead_register_total_width - 1 - lookahead_coefficient_offset_source_bottom} : {lookahead_register_total_width - 1 - lookahead_coefficient_offset_source_top}];"""
            content += """
                end
            end
        end
    end


endmodule"""

        self.syntax_generator.single_line_no_linebreak(content, indentation = 0)
        self.syntax_generator.close()