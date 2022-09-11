import FileGenerator
import SystemVerilogPort


class SystemVerilogSyntaxGenerator:
    output: FileGenerator.FileGenerator = None
    indentation_level: int = 0

    def __init__(self, path: str, name: str):
        self.output = FileGenerator.FileGenerator()
        self.output.set_path(path)
        self.output.set_name(name)
        self.output.open_output_file()

    def close(self):
        self.output.close_output_file()

    def get_indentation(self) -> str:
        indentation: str = ""
        for level in range(self.indentation_level):
            indentation += "\t"
        return indentation

    def single_line_no_linebreak(self, line: str, indentation: bool = 1):
        if indentation == 1:
            self.output.write_line_no_linebreak(self.get_indentation() + line)
        else:
            self.output.write_line_no_linebreak(line)

    def single_line_linebreak(self, line: str, indentation: bool = 1):
        if indentation == 1:
            self.output.write_line_linebreak(self.get_indentation() + line)
        else:
            self.output.write_line_linebreak(line)

    def module_head(self, module_name: str, parameter_list: dict = None, port_list: list = None):
        self.single_line_no_linebreak("module " + module_name + " ")
        if parameter_list is not None and len(parameter_list) != 0:
            self.parameter_list(parameter_list)
            self.single_line_no_linebreak(" ", bool(0))
        self.port_list(port_list)

    def parameter_list(self, parameter_list: dict):
        if parameter_list is None or len(parameter_list) == 0:
            return
        self.single_line_linebreak("#(", bool(0))
        self.indentation_level += 1
        count_parameter: int = 0
        for parameter in parameter_list.keys():
            count_parameter += 1
            if count_parameter == len(parameter_list):
                self.single_line_linebreak("parameter " + parameter + " = " + parameter_list.get(parameter))
            else:
                self.single_line_linebreak("parameter " + parameter + " = " + parameter_list.get(parameter) + ",")
        #self.indentation_level -= 1
        self.single_line_no_linebreak(")")

    def port_list(self, port_list: list):
        if port_list is None or len(port_list) == 0:
            return
        self.single_line_linebreak("(", bool(0))
        #self.indentation_level += 1
        count_ports: int = 0
        port: SystemVerilogPort.SystemVerilogPort = None
        for port in port_list:
            count_ports += 1
            if count_ports == len(port_list):
                if port.port_msb < 0 or port.port_lsb < 0:
                    self.single_line_linebreak(port.port_direction.direction + " " + port.port_type.type + port.port_name)
                else:
                    self.single_line_linebreak(port.port_direction.direction + " " + port.port_type.type + " [" + str(port.port_msb) + ":" + str(port.port_lsb) + "] " + port.port_name)
            else:
                if port.port_msb < 0 or port.port_lsb < 0:
                    self.single_line_linebreak(port.port_direction.direction + " " + port.port_type.type + port.port_name + ",")
                else:
                    self.single_line_linebreak(port.port_direction.direction + " " + port.port_type.type + " [" + str(port.port_msb) + ":" + str(port.port_lsb) + "] " + port.port_name + ",")
        self.indentation_level -= 1
        self.single_line_linebreak(");")