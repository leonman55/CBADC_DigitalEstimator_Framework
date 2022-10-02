import SystemVerilogModule
import SystemVerilogPort
import SystemVerilogPortDirection
import SystemVerilogPortType
import SystemVerilogDimension
import DigitalEstimatorModules.TestModule
import DigitalEstimatorModules.SimpleAdder
import SystemVerilogSignalSign
from SystemVerilogSyntaxGenerator import decimal_number, get_parameter_value, connect_port_array
import CBADC_HighLevelSimulation
import cbadc


class DigitalEstimatorWrapper(SystemVerilogModule.SystemVerilogModule):
    parameter_alu_input_width: dict[str, str] = {"ALU_INPUT_WIDTH": "32"}

    clk: SystemVerilogPort.SystemVerilogPort = None
    rst: SystemVerilogPort.SystemVerilogPort = None
    adder_input: list[SystemVerilogPort.SystemVerilogPort] = list[SystemVerilogPort.SystemVerilogPort]()
    adder_output: SystemVerilogPort.SystemVerilogPort = None

    def __init__(self, path: str, name: str):
        super().__init__(path, name)

    def generate(self):
        self.parameter_alu_input_width = self.add_parameter(self.parameter_alu_input_width)

        self.clk = self.add_port("clk", SystemVerilogPortDirection.Input(), SystemVerilogPortType.NoType(), -1, -1)
        self.rst = self.add_port("rst", SystemVerilogPortDirection.Input(), SystemVerilogPortType.NoType(), -1, -1)

        #test_module: DigitalEstimatorModules.TestModule.TestModule = DigitalEstimatorModules.TestModule.TestModule(self.path, "TestModule")
        #test_module.parameter_width = {"alu_width": "32"}
        #test_module.parameter_length = {"alu_input_channels": "2"}
        #test_module.parameter_alus = {"ALUS": "1"}
        #test_module.generate()
        #test_module_port_connections: dict[SystemVerilogPort.SystemVerilogPort, SystemVerilogPort.SystemVerilogPort] = dict[SystemVerilogPort.SystemVerilogPort, SystemVerilogPort.SystemVerilogPort]()
        #test_module_port_connections[test_module.clk] = self.clk
        #test_module_port_connections[test_module.rst] = self.rst

        #self.add_submodule(test_module, test_module_port_connections)

        simple_adder: DigitalEstimatorModules.SimpleAdder = DigitalEstimatorModules.SimpleAdder.SimpleAdder(self.path, "SimpleAdder")
        simple_adder.generate()
        simple_adder_port_connections: dict[SystemVerilogPort.SystemVerilogPort, SystemVerilogPort.SystemVerilogPort] = dict[SystemVerilogPort.SystemVerilogPort, SystemVerilogPort.SystemVerilogPort]()
        simple_adder_port_connections[simple_adder.clk] = self.clk
        simple_adder_port_connections[simple_adder.rst] = self.rst
        adder_input_0: SystemVerilogPort.SystemVerilogPort = self.add_port("adder_input_0", SystemVerilogPortDirection.Input(), SystemVerilogPortType.NoType(), int(get_parameter_value(self.parameter_alu_input_width)) - 1, 0)
        self.adder_input.append(adder_input_0)
        adder_input_1: SystemVerilogPort.SystemVerilogPort = self.add_port("adder_input_1", SystemVerilogPortDirection.Input(), SystemVerilogPortType.NoType(), int(get_parameter_value(self.parameter_alu_input_width)) - 1, 0)
        self.adder_input.append(adder_input_1)
        simple_adder_port_connections.update(connect_port_array(simple_adder.alu_input, self.adder_input))
        self.adder_output = self.add_port("adder_output", SystemVerilogPortDirection.Output(), SystemVerilogPortType.NoType(), int(get_parameter_value(self.parameter_alu_input_width)) - 1, 0)
        simple_adder_port_connections[simple_adder.alu_output] = self.adder_output

        self.add_submodule(simple_adder, simple_adder_port_connections)

        digital_estimator_fir: cbadc.digital_estimator.FIRFilter = CBADC_HighLevelSimulation.simulate_digital_estimator_fir()

        self.syntax_generator.timescale()
        self.syntax_generator.module_head(self.name, self.parameter_list, self.port_list)
        #count_modules: int = 0
        #module: SystemVerilogModule.SystemVerilogModule = None
        #for module in self.submodules.keys():
            #self.syntax_generator.module_instance(module.name, module.name + "_" + str(count_modules), module.parameter_list, module.port_list, self.submodules.get(module))
            #count_modules += 1

        dimensions: list[SystemVerilogDimension.SystemVerilogDimension] = list[SystemVerilogDimension.SystemVerilogDimension]()
        test_dimension: SystemVerilogDimension.SystemVerilogDimension = SystemVerilogDimension.SystemVerilogDimension(63, 0)
        dimensions.append(test_dimension)
        values: str = "{"
        for index in range(64):
            if index == 63:
                values += decimal_number(index) + "};"
            else:
                values += decimal_number(index) + ", "
        h_matrix = self.syntax_generator.local_parameter("h", SystemVerilogPortType.Logic(), SystemVerilogSignalSign.Signed(), 63, 0, dimensions, values)

        self.syntax_generator.instantiate_submodules(self.submodules)
        self.syntax_generator.end_module()
        self.syntax_generator.close()
        