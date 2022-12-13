import SystemVerilogModule


class LookUpTableSynchronousAssertions(SystemVerilogModule.SystemVerilogModule):
    configuration_input_width: int = 4
    configuration_data_width: int = 31


    def __init__(self, path: str, name: str):
        super().__init__(path, name)

    def generate(self):
        content: str = f"""module LookUpTableSynchronousAssertions #(
        parameter INPUT_WIDTH = {self.configuration_input_width},
        parameter DATA_WIDTH = {self.configuration_data_width}
    ) (
        input wire rst,
        input wire clk,
        input wire [INPUT_WIDTH - 1 : 0] in,
        input wire [2**INPUT_WIDTH - 1 : 0][DATA_WIDTH - 1 : 0] memory,
        input wire [DATA_WIDTH - 1 : 0] out
);

    property Reset;
        @(negedge rst) out == {{DATA_WIDTH{{1'b0}}}};
    endproperty

	property CheckResult;
        @(posedge clk) !$isunknown($past(in)) |-> out == memory[$past(in)];
    endproperty

    assert property (disable iff(rst) CheckResult)
        else $display("FAIL: LUT output incorrect!");

    assert property (Reset)
        //$display("PASS: LUT reset executed as expected.");
        else $display("FAIL: Reset unsuccessful!");
        
        
endmodule"""

        self.syntax_generator.single_line_no_linebreak(content, indentation = 0)
        self.syntax_generator.close()