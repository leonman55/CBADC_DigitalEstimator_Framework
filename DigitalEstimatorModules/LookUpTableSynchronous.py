import SystemVerilogModule


class LookUpTableSynchronous(SystemVerilogModule.SystemVerilogModule):

    def __init__(self, path: str, name: str):
        super().__init__(path, name)

    def generate(self):
        content: str = """module LookUpTableSynchronous #(
        parameter INPUT_WIDTH = 4,
        parameter DATA_WIDTH = 64
    ) (
        input wire rst,
        input wire clk,
        input wire [INPUT_WIDTH - 1 : 0] in,
        input wire signed [2**INPUT_WIDTH - 1 : 0][DATA_WIDTH - 1 : 0] memory,
        output logic signed [DATA_WIDTH - 1 : 0] out
);

    always_ff @(posedge clk) begin
        if(rst) begin
            out <= {DATA_WIDTH{1'b0}};
        end
        else begin
            out <= memory[in];
        end
    end

endmodule"""

        self.syntax_generator.single_line_no_linebreak(content, indentation = 0)
        self.syntax_generator.close()