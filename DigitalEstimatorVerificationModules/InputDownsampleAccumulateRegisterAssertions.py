import SystemVerilogModule


class InputDownsampleAccumulateRegisterAssertions(SystemVerilogModule.SystemVerilogModule):
    configuration_register_length: int = 2
    configuration_data_width: int = 31


    def __init__(self, path: str, name: str):
        super().__init__(path, name)

    def generate(self):
        content: str = f"""module InputDownsampleAccumulateRegisterAssertions #(
        parameter REGISTER_LENGTH = 2,
        parameter DATA_WIDTH = 4
    ) (
        input wire rst,
        input wire clk,
        input wire [DATA_WIDTH - 1 : 0] in,
        input wire [REGISTER_LENGTH - 1 : 0][DATA_WIDTH - 1 : 0] out
);

    property Reset;
        @(negedge rst) out == 1'b0;
    endproperty

	property CheckShifting;
        @(posedge clk) out[0] == in |-> ##(REGISTER_LENGTH - 1) out[REGISTER_LENGTH - 1] == $past(in, REGISTER_LENGTH - 1);
    endproperty

    assert property (Reset)
        //$display("PASS: Input downsample accumulate register reset executed as expected.");
        else $display("FAIL: Input downsample accumulate register reset unsuccessful!");

    assert property (disable iff(rst) CheckShifting)
        else $display("FAIL: Input downsample accumulate register is not shifting properly!");
 
        
endmodule"""

        self.syntax_generator.single_line_no_linebreak(content, indentation = 0)
        self.syntax_generator.close()