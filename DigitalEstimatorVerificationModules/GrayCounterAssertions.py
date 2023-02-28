import SystemVerilogModule


class GrayCounterAssertions(SystemVerilogModule.SystemVerilogModule):
    configuration_bit_size: int = 31
    configuration_top_value: int = 15


    def __init__(self, path: str, name: str):
        super().__init__(path, name)

    def generate(self):
        content: str = f"""module GrayCounterAssertions #(
        parameter BIT_SIZE = 2,
        parameter TOP_VALUE = 15
    ) (
        input wire rst,
        input wire clk,
        input wire [BIT_SIZE - 1 : 0] counter
);

    function automatic integer GrayCodeToBinary(input [BIT_SIZE - 1 : 0] gray_code);
        integer binary = 0;
        for(integer i = BIT_SIZE + 1; i >= 0; i--) begin
            binary = binary | ((((1 << (i + 1)) & binary) >> 1) ^ ((1 << i) & gray_code));
        end
        return binary;
    endfunction

    property Reset;
        @(negedge rst) counter == 1'b0;
    endproperty

	property CheckCounting;
        @(posedge clk) GrayCodeToBinary($past(rst)) == 0 && !$isunknown($past(counter)) |-> (GrayCodeToBinary($past(counter)) + 1) % (TOP_VALUE + 1) == GrayCodeToBinary(counter);
    endproperty

    assert property (Reset)
        //$display("PASS: Gray counter reset executed as expected.");
        else $display("FAIL: Gray counter reset unsuccessful!");

    assert property (disable iff(rst) CheckCounting)
        else $display("FAIL: Gray counter is not counting properly! &past(counter): %d	counter: %d", GrayCodeToBinary($sampled($past(counter))), GrayCodeToBinary($sampled(counter)));
 
        
endmodule"""

        self.syntax_generator.single_line_no_linebreak(content, indentation = 0)
        self.syntax_generator.close()