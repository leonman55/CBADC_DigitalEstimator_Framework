import SystemVerilogModule


class GrayCounter(SystemVerilogModule.SystemVerilogModule):
    configuration_counter_bit_width: int = 4
    configuration_counter_top_value: int = 15
    configuration_clock_edge: str = "posedge"


    def __init__(self, path: str, name: str):
        super().__init__(path, name)

    def generate(self):
        content: str = f"""module GrayCounter #(
        parameter BIT_SIZE = {self.configuration_counter_bit_width},
        parameter TOP_VALUE = {self.configuration_counter_top_value}
    ) (
        input wire rst,
        input wire clk,
        output logic [BIT_SIZE - 1 : 0] counter
);
	
	logic tmp [BIT_SIZE : 0];
	logic least_significant_one [BIT_SIZE : 0];
	logic tmp_msb;
	
    always_comb begin
        if(rst == 1) begin
            least_significant_one[0] = 1;
            for(logic [BIT_SIZE : 0] i = 1; i <= BIT_SIZE; i = i + 1) begin
				least_significant_one[i] = 0;
            end
            counter = 0;
            tmp_msb = 0;
        end
        else begin
            least_significant_one[0] = 1;
            for(logic [BIT_SIZE : 0] i = 1; i < BIT_SIZE; i = i + 1) begin
                least_significant_one[i] = least_significant_one[i-1] & ~tmp[i-1];
            end
            tmp_msb = tmp[BIT_SIZE] | tmp[BIT_SIZE - 1];
            for(logic [BIT_SIZE : 0] i = 0; i < BIT_SIZE; i = i + 1) begin
                counter[i] = tmp[i + 1];
            end
        end
    end
    
	"""
        if self.configuration_clock_edge == "posedge":
            content += "always @(posedge clk) begin"
        elif self.configuration_clock_edge == "negedge":
            content += "always @(negedge clk) begin"
        elif self.configuration_clock_edge == "both":
            content += "always @(clk) begin"
        content += """
        if(rst == 1) begin
            tmp[0] <= 1;
            for(logic [BIT_SIZE : 0] i = 1; i <= BIT_SIZE; i = i + 1) begin
                tmp[i] <= 0;
            end
        end
        else begin
            if(counter == (TOP_VALUE ^ (TOP_VALUE >> 1)))begin
                tmp[0] <= 1;
                for(logic [BIT_SIZE : 0] i = 1; i <= BIT_SIZE; i = i + 1) begin
                    tmp[i] <= 0;
                end
            end
            else begin
                tmp[0] <= ~tmp[0];
                for(logic [BIT_SIZE : 0] i = 1; i < BIT_SIZE; i = i + 1) begin
                    tmp[i] <= tmp[i] ^ (tmp[i-1] & least_significant_one[i-1]);
                end
                tmp[BIT_SIZE] <= tmp[BIT_SIZE] ^ (tmp_msb & least_significant_one[BIT_SIZE - 1]);
            end
        end
    end
	
endmodule"""

        self.syntax_generator.single_line_no_linebreak(content, indentation = 0)
        self.syntax_generator.close()