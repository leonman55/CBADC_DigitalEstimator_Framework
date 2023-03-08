import SystemVerilogModule


class ClockDividerAssertions(SystemVerilogModule.SystemVerilogModule):
    configuration_down_sample_rate: int = 5
    configuration_counter_type: str = "binary"


    def __init__(self, path: str, name: str):
        super().__init__(path, name)

    def generate(self):
        content: str = ""
        if self.configuration_counter_type == "binary":
            content = f"""module ClockDividerAssertions #(
        parameter DOWN_SAMPLE_RATE = 1,
        localparam EDGE_COUNTER_TOP_VALUE = DOWN_SAMPLE_RATE - 1,
        localparam CLOCK_COUNTER_OUTPUT_WIDTH = (DOWN_SAMPLE_RATE == 1) ? 1 : int'($ceil($clog2(DOWN_SAMPLE_RATE - 1)))
    ) (
        input wire rst,
        input wire clk,
        input wire clk_downsample,
        input wire [CLOCK_COUNTER_OUTPUT_WIDTH : 0] edge_counter,
        input wire [CLOCK_COUNTER_OUTPUT_WIDTH - 1 : 0] clock_divider_counter
);

    property Reset;
        @(negedge rst) clk_downsample == 1'b0 && clock_divider_counter == 0;
    endproperty

	property CheckCounting;
        @(posedge clk) $past(rst) == 0 && !$isunknown($past(edge_counter)) |-> ($past(edge_counter) + 1) % (EDGE_COUNTER_TOP_VALUE + 1) == edge_counter;
    endproperty

    property CheckDownsampledClock;
        @(posedge clk) (edge_counter < ((EDGE_COUNTER_TOP_VALUE + 1) / 2)) ? clk_downsample == 1'b0 : clk_downsample == 1'b1;
    endproperty

    assert property (Reset)
        //$display("PASS: ClockDivider reset executed as expected.");
        else $display("FAIL: ClockDivider reset unsuccessful!");

    assert property (disable iff(rst) CheckCounting)
        else $display("FAIL: ClockDivider is not counting properly! &past(edge_counter): %d\tedge_counter: %d", $past(edge_counter), edge_counter);

    assert property (disable iff(rst) CheckDownsampledClock)
        else $display("FAIL: Downsampled clock switched wrong! edge_counter: %d\tclk_downsample: %d", edge_counter, clk_downsample);
 
        
endmodule"""
        elif self.configuration_counter_type == "gray":
            content = """module ClockDividerAssertions #(
        parameter DOWN_SAMPLE_RATE = 1,
        localparam EDGE_COUNTER_TOP_VALUE = DOWN_SAMPLE_RATE - 1,
        localparam CLOCK_COUNTER_OUTPUT_WIDTH = (DOWN_SAMPLE_RATE == 1) ? 1 : int'($ceil($clog2(DOWN_SAMPLE_RATE - 1)))
    ) (
        input wire rst,
        input wire clk,
        input wire clk_downsample,
        input wire [CLOCK_COUNTER_OUTPUT_WIDTH : 0] edge_counter,
        input wire [CLOCK_COUNTER_OUTPUT_WIDTH - 1 : 0] clock_divider_counter
);

    function automatic integer GrayCodeToBinary(input [CLOCK_COUNTER_OUTPUT_WIDTH : 0] gray_code);
        integer binary = 0;
        for(integer i = CLOCK_COUNTER_OUTPUT_WIDTH + 1; i >= 0; i--) begin
            binary = binary | ((((1 << (i + 1)) & binary) >> 1) ^ ((1 << i) & gray_code));
        end
        return binary;
    endfunction

    property Reset;
        @(negedge rst) clk_downsample == 1'b0 && clock_divider_counter == 0;
    endproperty

	property CheckCounting;
        @(posedge clk) GrayCodeToBinary($past(rst)) == 0 && !$isunknown($past(edge_counter)) |-> (GrayCodeToBinary($past(edge_counter)) + 1) % (EDGE_COUNTER_TOP_VALUE + 1) == GrayCodeToBinary(edge_counter);
    endproperty

    property CheckDownsampledClock;
        @(posedge clk) (GrayCodeToBinary(edge_counter) < ((EDGE_COUNTER_TOP_VALUE + 1) / 2)) ? clk_downsample == 1'b0 : clk_downsample == 1'b1;
    endproperty

    assert property (Reset)
        //$display("PASS: ClockDivider reset executed as expected.");
        else $display("FAIL: ClockDivider reset unsuccessful!");

    assert property (disable iff(rst) CheckCounting)
        else $display("FAIL: ClockDivider is not counting properly! &past(edge_counter): %d\tedge_counter: %d", GrayCodeToBinary($sampled($past(edge_counter))), GrayCodeToBinary($sampled(edge_counter)));

    assert property (disable iff(rst) CheckDownsampledClock)
        else $display("FAIL: Downsampled clock switched wrong! edge_counter: %d\tclk_downsample: %d", edge_counter, clk_downsample);
 
        
endmodule"""

        self.syntax_generator.single_line_no_linebreak(content, indentation = 0)
        self.syntax_generator.close()