import SystemVerilogModule


class AdderCombinatorialReducedSizeAssertions(SystemVerilogModule.SystemVerilogModule):
    configuration_input0_width: int = 0
    configuration_input1_width: int = 0

    def __init__(self, path: str, name: str):
        super().__init__(path, name)

    def generate(self):
        content: str = f"""module AdderCombinatorialAssertions #(
		parameter INPUT0_WIDTH = {self.configuration_input0_width},
        parameter INPUT1_WIDTH = {self.configuration_input1_width},
		localparam OUTPUT_WIDTH = (INPUT0_WIDTH < INPUT1_WIDTH) ? INPUT1_WIDTH + 1 : INPUT0_WIDTH + 1
	) (
		input wire rst,
		input wire signed [INPUT0_WIDTH - 1 : 0] input_0,
		input wire signed [INPUT1_WIDTH - 1 : 0] input_1,
		input wire signed [OUTPUT_WIDTH - 1 : 0] out
);

    property Reset;
        @(negedge rst) out == {{OUTPUT_WIDTH{{1'b0}}}};
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