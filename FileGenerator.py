from io import FileIO
from io import BufferedWriter


class FileGenerator:
    file_path: str = None
    file_name: str = None
    output_file: FileIO = None

    def set_path(self, file_path: str):
        self.file_path = file_path

    def set_name(self, file_name: str):
        self.file_name = file_name

    def open_output_file(self):
        self.output_file = open(self.file_path + self.file_name, "w")

    def write_line(self, line_string: str):
        self.output_file.write(line_string)

    def write_line_no_linebreak(self, line_string: str):
        self.write_line(line_string)

    def write_line_linebreak(self, line_string: str):
        self.write_line(line_string + "\n")

    def close_output_file(self):
        self.output_file.close()
