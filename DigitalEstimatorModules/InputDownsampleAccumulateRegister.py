import SystemVerilogModule


class InputDownsampleAccumulateRegister(SystemVerilogModule.SystemVerilogModule):
    configuration_down_sample_rate: int = 1
    configuration_data_width: int = 4


    def __init__(self, path: str, name: str):
        super().__init__(path, name)

    def generate(self):
        content: str = f"""module InputDownsampleAccumulateRegister #(
        parameter REGISTER_LENGTH = {self.configuration_down_sample_rate},
        parameter DATA_WIDTH = {self.configuration_data_width}
    ) (
        input wire rst,
        input wire clk,
        input wire [DATA_WIDTH - 1 : 0] in,
        output reg [REGISTER_LENGTH - 1 : 0][DATA_WIDTH - 1 : 0] out
);

    always_ff @(posedge clk) begin
        if(rst == 1'b1) begin
            out <= {{REGISTER_LENGTH * DATA_WIDTH - 1{{1'b0}}}};
        end
        else begin
            if(REGISTER_LENGTH == 1) begin
                out <= in;
            end
            else begin
                out <= {{out[REGISTER_LENGTH - 2 : 0], in}};
            end
        end
    end


endmodule"""

        self.syntax_generator.single_line_no_linebreak(content, indentation = 0)
        self.syntax_generator.close()