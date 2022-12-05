import SystemVerilogModule


class ValidCounter(SystemVerilogModule.SystemVerilogModule):
    configuration_top_value: int = 1024

    def __init__(self, path: str, name: str):
        super().__init__(path, name)

    def generate(self):
        content: str = f"""module ValidCounter #(
        parameter TOP_VALUE = {self.configuration_top_value},
        localparam BIT_SIZE = int'($ceil(real'($clog2(TOP_VALUE + 1))))
    ) (
        input wire rst,
        input wire clk,
        output logic valid
);
	
    logic [BIT_SIZE - 1 : 0] counter;
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
    
	always_ff @(posedge clk) begin
        if(rst == 1) begin
            tmp[0] <= 1;
            for(logic [BIT_SIZE : 0] i = 1; i <= BIT_SIZE; i = i + 1) begin
                tmp[i] <= 0;
            end
        end
        else begin
            if(counter == (TOP_VALUE ^ (TOP_VALUE >> 1))) begin
                
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

    always_comb begin
        if(rst == 1) begin
            valid = 1'b0;
        end
        else begin
            if(counter == (TOP_VALUE ^ (TOP_VALUE >> 1))) begin
                valid <= 1'b1;
            end
            else begin
                valid <= 1'b0;
            end
        end
    end
	
endmodule"""

        self.syntax_generator.single_line_no_linebreak(content, indentation = 0)
        self.syntax_generator.close()