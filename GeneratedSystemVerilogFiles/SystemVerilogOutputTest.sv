module Main #(
	parameter width = 1,
	parameter length = 5,
	parameter ALU's = 7
	) (
	input clk,
	input rst
);

logic [63:0] alu_input_0 [1:0];
logic [63:0] alu_input_1 [1:0];

assign alu_input_0 = alu_input_1;

always_comb begin
	alu_input_0 = alu_input_1;
end

always_ff @(posedge clk, negedge rst) begin
	alu_input_0 <= alu_input_1;
end

endmodule
