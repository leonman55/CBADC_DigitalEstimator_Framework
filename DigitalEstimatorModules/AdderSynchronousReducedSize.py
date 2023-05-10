import SystemVerilogModule


class AdderSynchronousReducedSize(SystemVerilogModule.SystemVerilogModule):
    configuration_input0_width: int = 32
    configuration_input1_width: int = 32

    def __init__(self, path: str, name: str):
        super().__init__(path, name)

    def generate(self):
        content: str = f"""module AdderSynchronous #(
		parameter INPUT0_WIDTH = {self.configuration_input0_width},
        parameter INPUT1_WIDTH = {self.configuration_input1_width},
		localparam OUTPUT_WIDTH = (INPUT0_WIDTH < INPUT1_WIDTH) ? INPUT1_WIDTH + 1 : INPUT0_WIDTH + 1
	) (
		input wire rst,
		input wire clk,
		input wire signed [INPUT0_WIDTH - 1 : 0] input_0,
		input wire signed [INPUT1_WIDTH - 1 : 0] input_1,
		output logic signed [OUTPUT_WIDTH - 1 : 0] out
);

	always_ff @(posedge clk) begin
        if(rst == 1) begin
            out <= {{OUTPUT_WIDTH{{1'b0}}}};
        end
		else begin
            out <= input_0 + input_1;
        end
	end


endmodule"""

        self.syntax_generator.single_line_no_linebreak(content, indentation = 0)
        self.syntax_generator.close()