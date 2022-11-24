import SystemVerilogModule


class LookUpTableCoefficientRegister(SystemVerilogModule.SystemVerilogModule):
    configuration_lookup_table_data_width: int = 31
    configuration_lookback_lookup_table_entries_count: int = 512
    configuration_lookahead_lookup_table_entries_count: int = 512

    def __init__(self, path: str, name: str):
        super().__init__(path, name)

    def generate(self):
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

        self.syntax_generator.single_line_no_linebreak(content, indentation = 0)
        self.syntax_generator.close()