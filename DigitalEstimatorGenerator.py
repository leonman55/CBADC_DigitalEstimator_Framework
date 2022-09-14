import os
import platform

import FileGenerator
import SystemVerilogSignal
import SystemVerilogSyntaxGenerator
import SystemVerilogPort
import SystemVerilogPortDirection
import SystemVerilogPortType
import SystemVerilogClockEdge


def main():
    #print(os.getcwd())
    #output = FileGenerator.FileGenerator()
    path: str = ""
    if platform.system() == "Linux":
        #output.set_path("./GeneratedSystemVerilogFiles/")
        path = "./GeneratedSystemVerilogFiles/"
    elif platform.system() == "Windows":
        #output.set_path(".\\GeneratedSystemVerilogFiles\\")
        path = ".\\GeneratedSystemVerilogFiles\\"
    name = "SystemVerilogOutputTest.sv"

    #output.write_line("Test line 1.")
    #output.write_line_linebreak("Test line 2.")
    #output.write_line_linebreak("Test line 3.")

    main_test_syntax = SystemVerilogSyntaxGenerator.SystemVerilogSyntaxGenerator(path, name)
    parameters: dict = {"width": "1", "length": "5", "ALU's": "7"}
    ports: list = list()
    clk: SystemVerilogPort.SystemVerilogPort = SystemVerilogPort.SystemVerilogPort("clk", SystemVerilogPortDirection.Input(), SystemVerilogPortType.NoType(), -1, -1)
    rst: SystemVerilogPort.SystemVerilogPort = SystemVerilogPort.SystemVerilogPort("rst", SystemVerilogPortDirection.Input(), SystemVerilogPortType.NoType(), -1, -1)
    ports.append(clk)
    ports.append(rst)
    main_test_syntax.module_head("Main", parameters, ports)
    alu_input_0: SystemVerilogSignal.SystemVerilogSignal = main_test_syntax.signal("alu_input_0", SystemVerilogPortType.Logic(), 63, 0, 1, 0)
    alu_input_1: SystemVerilogSignal.SystemVerilogSignal = main_test_syntax.signal("alu_input_1", SystemVerilogPortType.Logic(), 63, 0, 1, 0)
    main_test_syntax.blank_line()
    main_test_syntax.assign(alu_input_0, alu_input_1)
    main_test_syntax.blank_line()
    main_test_syntax.always_combinatorial()
    main_test_syntax.assign(alu_input_0, alu_input_1)
    main_test_syntax.end_always()
    main_test_syntax.blank_line()
    main_test_syntax.always_synchronous({clk: SystemVerilogClockEdge.SystemVerilogPosedge(), rst: SystemVerilogClockEdge.SystemVerilogNegedge()})
    main_test_syntax.assign(alu_input_0, alu_input_1)
    main_test_syntax.end_always()
    main_test_syntax.blank_line()
    main_test_syntax.end_module()
    main_test_syntax.close()

if __name__ == '__main__':
    main()
