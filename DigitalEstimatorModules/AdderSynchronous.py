import SystemVerilogModule


class AdderSynchronous(SystemVerilogModule.SystemVerilogModule):

    def __init__(self, path: str, name: str):
        super().__init__(path, name)

    def generate(self):
        content: str = f"""module AdderSynchronous #(
		parameter INPUT_WIDTH = 32,
		localparam OUTPUT_WIDTH = INPUT_WIDTH
	) (
		input wire rst,
		input wire clk,
		input wire [INPUT_WIDTH - 1 : 0] input_0,
		input wire [INPUT_WIDTH - 1 : 0] input_1,
		output logic [OUTPUT_WIDTH - 1 : 0] out
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