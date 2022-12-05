import SystemVerilogModule


class LookUpTableBlock(SystemVerilogModule.SystemVerilogModule):

    def __init__(self, path: str, name: str):
        super().__init__(path, name)

    def generate(self):
        content: str = """module LookUpTableBlock #(
        parameter TOTAL_INPUT_WIDTH = 4,
        parameter LOOKUP_TABLE_INPUT_WIDTH = 4,
        localparam LOOKUP_TABLE_COUNT = int'($ceil(real'(TOTAL_INPUT_WIDTH) / real'(LOOKUP_TABLE_INPUT_WIDTH))),
        localparam LOOKUP_TABLE_ENTRIES_COUNT = int'(TOTAL_INPUT_WIDTH / LOOKUP_TABLE_INPUT_WIDTH) * (2**LOOKUP_TABLE_INPUT_WIDTH) + ((TOTAL_INPUT_WIDTH % LOOKUP_TABLE_INPUT_WIDTH) == 0 ? 0 : (2**(TOTAL_INPUT_WIDTH % LOOKUP_TABLE_INPUT_WIDTH))),
        parameter LOOKUP_TABLE_DATA_WIDTH = 1
    ) (
        input wire rst,
        input wire [TOTAL_INPUT_WIDTH - 1 : 0] input_register,
        input wire [LOOKUP_TABLE_ENTRIES_COUNT - 1 : 0][LOOKUP_TABLE_DATA_WIDTH - 1 : 0] lookup_table_entries,
        output logic [LOOKUP_TABLE_COUNT - 1 : 0][LOOKUP_TABLE_DATA_WIDTH - 1 : 0] lookup_table_results
);

    generate
        for(genvar lookup_table_number = 0; lookup_table_number < LOOKUP_TABLE_COUNT; lookup_table_number++) begin
            if((TOTAL_INPUT_WIDTH - (lookup_table_number * LOOKUP_TABLE_INPUT_WIDTH)) >= LOOKUP_TABLE_INPUT_WIDTH) begin
                LookUpTable #(
                        .INPUT_WIDTH(LOOKUP_TABLE_INPUT_WIDTH),
                        .DATA_WIDTH(LOOKUP_TABLE_DATA_WIDTH)
                    )
                    lookup_table (
                        .rst(rst),
                        .in(input_register[(lookup_table_number * LOOKUP_TABLE_INPUT_WIDTH) +: LOOKUP_TABLE_INPUT_WIDTH]),
                        .memory(lookup_table_entries[(lookup_table_number * (2**LOOKUP_TABLE_INPUT_WIDTH)) +: 2**LOOKUP_TABLE_INPUT_WIDTH]),
                        .out(lookup_table_results[lookup_table_number])
                );
            end
            else begin
                LookUpTable #(
                        //.INPUT_WIDTH(LOOKUP_TABLE_INPUT_WIDTH),
                        .INPUT_WIDTH(TOTAL_INPUT_WIDTH - (lookup_table_number * LOOKUP_TABLE_INPUT_WIDTH)),
                        .DATA_WIDTH(LOOKUP_TABLE_DATA_WIDTH)
                    )
                    lookup_table (
                        .rst(rst),
                        .in(input_register[(lookup_table_number * (LOOKUP_TABLE_INPUT_WIDTH)) +: (TOTAL_INPUT_WIDTH - (lookup_table_number * LOOKUP_TABLE_INPUT_WIDTH))]),
                        .memory(lookup_table_entries[(lookup_table_number * (2**LOOKUP_TABLE_INPUT_WIDTH)) +: (2**(TOTAL_INPUT_WIDTH % LOOKUP_TABLE_INPUT_WIDTH))]),
                        .out(lookup_table_results[lookup_table_number])
                );
            end
        end
    endgenerate


endmodule"""

        self.syntax_generator.single_line_no_linebreak(content, indentation = 0)
        self.syntax_generator.close()