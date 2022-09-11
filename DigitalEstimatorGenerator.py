import os
import platform

import FileGenerator
import SystemVerilogSyntaxGenerator
import SystemVerilogPort
import SystemVerilogPortDirection
import SystemVerilogPortType


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
    ports.append(SystemVerilogPort.SystemVerilogPort("clk", SystemVerilogPortDirection.Input(), SystemVerilogPortType.NoType(), -1, -1))
    ports.append(SystemVerilogPort.SystemVerilogPort("rst", SystemVerilogPortDirection.Input(), SystemVerilogPortType.NoType(), -1, -1))
    main_test_syntax.module_head("Main", parameters, ports)
    main_test_syntax.close()

if __name__ == '__main__':
    main()
