import SystemVerilogModule


class ClockDivider(SystemVerilogModule.SystemVerilogModule):
    configuration_down_sample_rate: int = 1
    configuration_counter_type: str = "binary"


    def __init__(self, path: str, name: str):
        super().__init__(path, name)

    def generate(self):
        content: str = ""
        if self.configuration_counter_type == "binary":
            content: str = f"""module ClockDivider #(
        parameter DOWN_SAMPLE_RATE = {self.configuration_down_sample_rate},
        localparam EDGE_COUNTER_TOP_VALUE = (2 * DOWN_SAMPLE_RATE) - 1,
        localparam CLOCK_COUNTER_OUTPUT_WIDTH = (DOWN_SAMPLE_RATE == 1) ? 1 : int'($ceil($clog2(2 * (DOWN_SAMPLE_RATE - 1))))
    ) (
        input wire rst,
        input wire clk,
        output reg clk_downsample,
        output reg [CLOCK_COUNTER_OUTPUT_WIDTH - 1 : 0] clock_divider_counter
);

    logic reset_executed;
    logic [CLOCK_COUNTER_OUTPUT_WIDTH : 0] edge_counter;

    assign clock_divider_counter = edge_counter >> 1;

    always_ff @(clk) begin
        if(rst == 1'b1) begin
            reset_executed <= 1;
            edge_counter <= {{CLOCK_COUNTER_OUTPUT_WIDTH{{1'b0}}}};
        end
        else begin
            if(edge_counter == EDGE_COUNTER_TOP_VALUE) begin
                edge_counter <= {{CLOCK_COUNTER_OUTPUT_WIDTH{{1'b0}}}};
            end
            else begin
                if(reset_executed == 1'b1 && clk == 1'b1 || reset_executed == 1'b0) begin
                    reset_executed <= 1'b0;
                    edge_counter <= edge_counter + 1;
                end
            end
        end
    end

    always_comb begin
        if(rst) begin
            clk_downsample = 1'b0;
        end
        else begin
            if(edge_counter < ((EDGE_COUNTER_TOP_VALUE + 1) / 2)) begin
                clk_downsample = 1'b0;
            end
            else begin
                clk_downsample = 1'b1;
            end
        end
    end


endmodule"""
        elif self.configuration_counter_type == "gray":
            content = f"""module ClockDivider #(
        parameter DOWN_SAMPLE_RATE = {self.configuration_down_sample_rate},
        localparam EDGE_COUNTER_TOP_VALUE = ((2 * DOWN_SAMPLE_RATE) - 1) ^ (((2 * DOWN_SAMPLE_RATE) - 1) >> 1),
        localparam CLOCK_COUNTER_OUTPUT_WIDTH = (DOWN_SAMPLE_RATE == 1) ? 1 : int'($ceil($clog2(DOWN_SAMPLE_RATE - 1)))
    ) (
        input wire rst,
        input wire clk,
        output reg clk_downsample,
        output reg [CLOCK_COUNTER_OUTPUT_WIDTH - 1 : 0] clock_divider_counter
);

    logic [CLOCK_COUNTER_OUTPUT_WIDTH : 0] edge_counter;
    logic [CLOCK_COUNTER_OUTPUT_WIDTH : 0] edge_counter_binary;

    assign clock_divider_counter = edge_counter_binary;

    always_ff @(posedge clk) begin
        if(rst) begin
            clk_downsample <= 1'b0;
        end
        else begin
            if(edge_counter == ((DOWN_SAMPLE_RATE - 1) ^ ((DOWN_SAMPLE_RATE - 1) >> 1))) begin
                clk_downsample <= 1'b0;
            end
            else if(edge_counter == (((DOWN_SAMPLE_RATE / 2) - 1) ^ (((DOWN_SAMPLE_RATE / 2) - 1) >> 1))) begin
                clk_downsample <= 1'b1;
            end
        end
    end


    GrayCounter #(
            .BIT_SIZE(CLOCK_COUNTER_OUTPUT_WIDTH + 1),
            .TOP_VALUE(DOWN_SAMPLE_RATE - 1)
        )
        gray_counter (
            .rst(rst),
            .clk(clk),
            .counter(edge_counter)
    );

    GrayCodeToBinary #(
            .BIT_SIZE(CLOCK_COUNTER_OUTPUT_WIDTH + 1)
        )
        gray_code_to_binary (
            .rst(rst),
            .gray_code(edge_counter),
            .binary(edge_counter_binary)
    );


endmodule"""

        self.syntax_generator.single_line_no_linebreak(content, indentation = 0)
        self.syntax_generator.close()