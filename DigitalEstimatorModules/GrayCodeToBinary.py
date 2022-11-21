import SystemVerilogModule


class GrayCodeToBinary(SystemVerilogModule.SystemVerilogModule):
    configuration_bit_width: int = 4


    def __init__(self, path: str, name: str):
        super().__init__(path, name)

    def generate(self):
        content: str = f"""module GrayCodeToBinary #(
        parameter BIT_SIZE = {self.configuration_bit_width}
    ) (
        input logic rst,
        input logic [BIT_SIZE - 1 : 0] gray_code,
        output logic [BIT_SIZE - 1 : 0] binary
);
    
    always @(rst, gray_code) begin
        if(rst) begin
            binary = 0;
        end
        binary = 0;
        for(integer i = BIT_SIZE; i >= 0; i--) begin
            binary = binary | ((((1 << (i + 1)) & binary) >> 1) ^ ((1 << i) & gray_code));
        end
    end
    
endmodule"""

        self.syntax_generator.single_line_no_linebreak(content, indentation = 0)
        self.syntax_generator.close()