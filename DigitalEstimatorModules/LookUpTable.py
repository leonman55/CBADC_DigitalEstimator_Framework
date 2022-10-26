import SystemVerilogModule


class LookUpTable(SystemVerilogModule.SystemVerilogModule):
    content: str = """module LookUpTable #(
        parameter INPUT_WIDTH = 4,
        parameter DATA_WIDTH = 64
    ) (
        input wire [INPUT_WIDTH - 1 : 0] in,
        input wire [2**INPUT_WIDTH - 1 : 0][DATA_WIDTH - 1 : 0] memory,
        output [DATA_WIDTH - 1 : 0] out
);

    assign out = memory[in];

endmodule"""

    def __init__(self, path: str, name: str):
        super().__init__(path, name)

    def generate(self):
        self.syntax_generator.single_line_no_linebreak(self.content, indentation = 0)
        self.syntax_generator.close()