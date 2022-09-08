import os
import platform

import FileGenerator
import SystemVerilogSyntaxGenerator


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
    main_test_syntax.module_head("Main", {"width": "1", "length": "5", "ALU's": "7"}, None)
    main_test_syntax.close()

if __name__ == '__main__':
    main()
