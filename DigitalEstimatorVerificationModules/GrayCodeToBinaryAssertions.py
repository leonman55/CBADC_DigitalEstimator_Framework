import SystemVerilogModule


class GrayCodeToBinaryAssertions(SystemVerilogModule.SystemVerilogModule):
    configuration_bit_size: int = 31


    def __init__(self, path: str, name: str):
        super().__init__(path, name)

    def generate(self):
        content: str = f"""module GrayCodeToBinaryAssertions #(
        parameter BIT_SIZE = 2
    ) (
        input wire rst,
        input wire [BIT_SIZE - 1 : 0] gray_code,
        input wire [BIT_SIZE - 1 : 0] binary
);

    property Reset;
        @(negedge rst) binary == 0;
    endproperty

    property CheckValidOutput;
        @(gray_code) !$isunknown(gray_code) |-> !$isunknown(binary);
    endproperty

    assert property (Reset)
        //$display("PASS: GrayCodeToBinary reset executed as expected.");
        else $display("FAIL: GrayCodeToBinary reset unsuccessful!");

    assert property (disable iff(rst) CheckValidOutput)
        else $display("FAIL: A defined input Gray code should lead to a defined output binary code!");

        
endmodule"""

        self.syntax_generator.single_line_no_linebreak(content, indentation = 0)
        self.syntax_generator.close()