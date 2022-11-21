import SystemVerilogModule


class AdderBlockCombinatorial(SystemVerilogModule.SystemVerilogModule):

    def __init__(self, path: str, name: str):
        super().__init__(path, name)

    def generate(self):
        content: str = """module AdderBlockCombinatorial #(
    	parameter INPUT_COUNT = 2,
        parameter INPUT_WIDTH = 1,
        localparam STAGE_COUNT = int'($ceil($clog2(INPUT_COUNT)))
    ) (
        input wire rst,
        input wire [INPUT_COUNT - 1 : 0][INPUT_WIDTH - 1 : 0] in,
        //output reg [INPUT_WIDTH + STAGE_COUNT - 1 : 0] out
        output reg [INPUT_WIDTH - 1 : 0] out
);

    function automatic int GetTotalNumberOfIntermediateResults();
        int count_intermediate_results = 0;
        for(int stage_index = 1; stage_index <= STAGE_COUNT; stage_index++) begin
            count_intermediate_results = count_intermediate_results + int'($ceil(real'(INPUT_COUNT) / (2**stage_index)));
        end
        return count_intermediate_results;
    endfunction

    function automatic int GetAdderInputOffset(int stage_number);
        int adder_input_offset = 0;
        if(stage_number < 3) begin
            return adder_input_offset;
        end
        else begin
            for(int stage = 1; stage <= stage_number - 2; stage ++) begin
                adder_input_offset = adder_input_offset + $ceil(real'(INPUT_COUNT) / (2**stage));
            end
            return adder_input_offset;
        end
    endfunction

    function automatic int GetAdderResultOffset(int stage_number);
        int adder_result_offset = 0;
        if(stage_number < 2) begin
            return adder_result_offset;
        end
        else begin
            for(int stage = 1; stage <= stage_number - 1; stage ++) begin
                adder_result_offset = adder_result_offset + $ceil(real'(INPUT_COUNT) / (2**stage));
            end
            return adder_result_offset;
        end
    endfunction

    function automatic int GetNumberOfAddersInStage(int stage);
        return int'($ceil(real'(INPUT_COUNT) / (2**stage)));
    endfunction

    generate
        localparam count_intermediate_results = GetTotalNumberOfIntermediateResults();
        //wire [count_intermediate_results - 1 : 0][INPUT_WIDTH + STAGE_COUNT - 1 : 0] adder_out;
        logic [count_intermediate_results - 1 : 0][INPUT_WIDTH - 1 : 0] adder_out;
        for(genvar stage_index = 1; stage_index <= STAGE_COUNT; stage_index++) begin
            localparam adder_input_offset = GetAdderInputOffset(stage_index);
            localparam adder_out_offset = GetAdderResultOffset(stage_index);
            for(genvar element_index = 0; element_index < GetNumberOfAddersInStage(stage_index); element_index++) begin
                if((2 * element_index) != int'($ceil(real'(INPUT_COUNT) / (2**(stage_index - 1)))) - 1 || stage_index == STAGE_COUNT) begin
                    AdderCombinatorial #(
                        //.INPUT_WIDTH(INPUT_WIDTH + stage_index - 1)
                        .INPUT_WIDTH(INPUT_WIDTH)
                        )
                        adder (
                            .rst(rst),
                            .input_0(stage_index - 1 ? adder_out[2 * element_index + adder_input_offset] : in[2 * element_index]),
                            .input_1(stage_index - 1 ? adder_out[2 * element_index + 1 + adder_input_offset] : in[2 * element_index + 1]),
                            .out(adder_out[element_index + adder_out_offset])
                    );
                end
                else begin
                    assign adder_out[element_index + adder_out_offset] = stage_index - 1 ? adder_out[adder_out_offset - 1] : in[2 * element_index];
                end
            end
        end
        assign out = adder_out[count_intermediate_results - 1];
    endgenerate


    /*initial begin
        for(int stage_index = 1; stage_index <= STAGE_COUNT; stage_index++) begin
            $display("Number of adders in stage: %d", GetNumberOfAddersInStage(stage_index));
        end
    end*/


endmodule"""

        self.syntax_generator.single_line_no_linebreak(content, indentation = 0)
        self.syntax_generator.close()