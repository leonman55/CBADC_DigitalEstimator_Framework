import SystemVerilogModule


class LookUpTableBlockAssertions(SystemVerilogModule.SystemVerilogModule):
    configuration_total_input_width: int = 4
    configuration_look_up_table_input_width: int = 4
    configuration_look_up_table_data_width: int = 31


    def __init__(self, path: str, name: str):
        super().__init__(path, name)

    def generate(self):
        content: str = f"""module LookUpTableBlockAssertions #(
        parameter TOTAL_INPUT_WIDTH = {self.configuration_total_input_width},
        parameter LOOKUP_TABLE_INPUT_WIDTH = {self.configuration_look_up_table_input_width},
        localparam LOOKUP_TABLE_COUNT = int'($ceil(real'(TOTAL_INPUT_WIDTH) / real'(LOOKUP_TABLE_INPUT_WIDTH))),
        localparam LOOKUP_TABLE_ENTRIES_COUNT = int'(TOTAL_INPUT_WIDTH / LOOKUP_TABLE_INPUT_WIDTH) * (2**LOOKUP_TABLE_INPUT_WIDTH) + ((TOTAL_INPUT_WIDTH % LOOKUP_TABLE_INPUT_WIDTH) == 0 ? 0 : (2**(TOTAL_INPUT_WIDTH % LOOKUP_TABLE_INPUT_WIDTH))),
        parameter LOOKUP_TABLE_DATA_WIDTH = {self.configuration_look_up_table_data_width}
    ) (
        input wire rst,
        input wire [TOTAL_INPUT_WIDTH - 1 : 0] input_register,
        input wire [LOOKUP_TABLE_ENTRIES_COUNT - 1 : 0][LOOKUP_TABLE_DATA_WIDTH - 1 : 0] lookup_table_entries,
        input wire [LOOKUP_TABLE_COUNT - 1 : 0][LOOKUP_TABLE_DATA_WIDTH - 1 : 0] lookup_table_results
);




endmodule"""

        self.syntax_generator.single_line_no_linebreak(content, indentation = 0)
        self.syntax_generator.close()