import math

import SystemVerilogModule


class AdderBlockCombinatorialAssertions(SystemVerilogModule.SystemVerilogModule):
    configuration_input_count: int = 2
    configuration_adder_input_width: int = 31
    stage_count: int = math.ceil(math.log2(float(configuration_input_count)))


    def __init__(self, path: str, name: str):
        super().__init__(path, name)
        self.stage_count = math.ceil(math.log2(float(self.configuration_input_count)))

    def generate(self):
        content: str = f"""module AdderBlockCombinatorialAssertions #(
    	parameter INPUT_COUNT = 2,
        parameter INPUT_WIDTH = 1,
        localparam STAGE_COUNT = int'($ceil($clog2(INPUT_COUNT)))
    ) (
        input wire rst,
        input wire [INPUT_COUNT - 1 : 0][INPUT_WIDTH - 1 : 0] in,
        input reg [INPUT_WIDTH - 1 : 0] out
);

    


endmodule"""

        self.syntax_generator.single_line_no_linebreak(content, indentation = 0)
        self.syntax_generator.close()