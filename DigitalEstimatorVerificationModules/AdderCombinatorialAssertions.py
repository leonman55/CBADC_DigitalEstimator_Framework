import SystemVerilogModule


class AdderCombinatorialAssertions(SystemVerilogModule.SystemVerilogModule):
    configuration_adder_input_width: int = 31


    def __init__(self, path: str, name: str):
        super().__init__(path, name)

    def generate(self):
        content: str = f"""module AdderCombinatorialAssertions #(
		parameter INPUT_WIDTH = {self.configuration_adder_input_width},
		localparam OUTPUT_WIDTH = INPUT_WIDTH
	) (
		input rst,
		input [INPUT_WIDTH - 1 : 0] input_0,
		input [INPUT_WIDTH - 1 : 0] input_1,
		input [OUTPUT_WIDTH - 1 : 0] out
);

    property Reset;
        @(negedge rst) out == {{INPUT_WIDTH{{1'b0}}}};
    endproperty

	property CheckResult;
        @(input_0, input_1) out == input_0 + input_1;
    endproperty

    assert property (disable iff(rst || $isunknown(input_0) || $isunknown(input_1)) CheckResult)
        else $display("FAIL: Result incorrect!");

    assert property (Reset)
        //$display("PASS: Reset executed as expected.");
        else $display("FAIL: Reset unsuccessful!");


endmodule"""

        self.syntax_generator.single_line_no_linebreak(content, indentation = 0)
        self.syntax_generator.close()