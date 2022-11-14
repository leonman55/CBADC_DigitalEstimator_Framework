from asyncio.subprocess import PIPE
from genericpath import isfile
import os
import platform
import subprocess
from pathlib import Path
from stat import *

import FileGenerator
import SystemVerilogSignal
# import SystemVerilogSyntaxGenerator
import SystemVerilogPort
import SystemVerilogPortDirection
import SystemVerilogPortType
import SystemVerilogClockEdge
import SystemVerilogModule
import DigitalEstimatorModules.TestModule
import DigitalEstimatorModules.DigitalEstimatorWrapper
import DigitalEstimatorVerificationModules.DigitalEstimatorTestbench
import DigitalEstimatorModules.DigitalEstimatorWrapper
import DigitalEstimatorModules.LookUpTable
import DigitalEstimatorModules.LookUpTableBlock
import DigitalEstimatorModules.AdderCombinatorial
import DigitalEstimatorModules.AdderBlockCombinatorial
import CBADC_HighLevelSimulation


class DigitalEstimatorGenerator():
    path: str = "../df/sim/SystemVerilogFiles"

    configuration_number_of_timesteps_in_clock_cycle: int = 10
    configuration_n_number_of_analog_states: int = 4
    configuration_m_number_of_digital_states: int = configuration_n_number_of_analog_states
    configuration_beta: float = 6250.0
    configuration_rho: float = -1e-2
    configuration_kappa: float = -1.0
    configuration_eta2: float = 1e7
    configuration_lookback_length: int = 512
    configuration_lookahead_length: int = 4
    configuration_fir_data_width: int = 31
    configuration_fir_lut_input_width: int = 4
    configuration_simulation_length: int = 2 << 12
    configuration_offset: int = 0.1

    def generate(self):
        digital_estimator_testbench: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorVerificationModules.DigitalEstimatorTestbench.DigitalEstimatorTestbench(self.path, "DigitalEstimatorTestbench")
        digital_estimator_testbench.configuration_number_of_timesteps_in_clock_cycle = self.configuration_number_of_timesteps_in_clock_cycle
        digital_estimator_testbench.configuration_rho = self.configuration_rho
        digital_estimator_testbench.configuration_beta = self.configuration_beta
        digital_estimator_testbench.configuration_eta2 = self.configuration_eta2
        digital_estimator_testbench.configuration_kappa = self.configuration_kappa
        digital_estimator_testbench.configuration_lookback_length = self.configuration_lookback_length
        digital_estimator_testbench.configuration_lookahead_length = self.configuration_lookahead_length
        digital_estimator_testbench.configuration_n_number_of_analog_states = self.configuration_n_number_of_analog_states
        digital_estimator_testbench.configuration_m_number_of_digital_states = self.configuration_m_number_of_digital_states
        digital_estimator_testbench.configuration_fir_data_width = self.configuration_fir_data_width
        digital_estimator_testbench.configuration_fir_lut_input_width = self.configuration_fir_lut_input_width
        digital_estimator_testbench.configuration_simulation_length = self.configuration_simulation_length
        digital_estimator_testbench.generate()

        digital_estimator: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorModules.DigitalEstimatorWrapper.DigitalEstimatorWrapper(self.path, "DigitalEstimator")
        digital_estimator.generate()
        look_up_table: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorModules.LookUpTable.LookUpTable(self.path, "LookUpTable")
        look_up_table.generate()
        look_up_table_block: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorModules.LookUpTableBlock.LookUpTableBlock(self.path, "LookUpTableBlock")
        look_up_table_block.generate()
        adder_combinatorial: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorModules.AdderCombinatorial.AdderCombinatorial(self.path, "AdderCombinatorial")
        adder_combinatorial.generate()
        adder_block_combinatorial: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorModules.AdderBlockCombinatorial.AdderBlockCombinatorial(self.path, "AdderBlockCombinatorial")
        adder_block_combinatorial.generate()

        high_level_simulation: CBADC_HighLevelSimulation.DigitalEstimatorParameterGenerator = CBADC_HighLevelSimulation.DigitalEstimatorParameterGenerator(
            self.path,
            self.configuration_n_number_of_analog_states,
            self.configuration_m_number_of_digital_states,
            self.configuration_beta,
            self.configuration_rho,
            self.configuration_kappa,
            self.configuration_eta2,
            self.configuration_lookback_length,
            self.configuration_lookahead_length,
            self.configuration_fir_data_width,
            size = self.configuration_simulation_length,
            offset = self.configuration_offset
        )
        high_level_simulation.write_control_signal_to_csv_file(high_level_simulation.simulate_analog_system())
        high_level_simulation.simulate_digital_estimator_fir()
        high_level_simulation.simulate_fir_filter(number_format_float = True)
        #high_level_simulation.simulate_fir_filter(number_format_float = False)

        simulation_settings: list[str] = list[str]()
        simulation_settings.append("source ~/pro/acmos2/virtuoso/setup_user")
        simulation_settings.append("source ~/pro/fall2022/bash/setup_user")
        simulation_settings.append("cd " + self.path)
        simulation_settings.append("xrun -f xrun_options")
        self.write_xrun_simulation_file("sim.sh", simulation_settings)

        options: list[str] = list[str]()
        #options.append("-gui")
        #options.append("-q")
        options.append("-access +rwc")
        options.append("-top " + digital_estimator_testbench.name)
        options.append("*.sv")
        self.write_xrun_options_file("xrun_options", options)

        #sim_xrun: subprocess.CompletedProcess = subprocess.run(sim_folder + "sim.sh", shell = True)
        sim_xrun = subprocess.Popen(["./sim.sh"], cwd = self.path, stdout = PIPE, text = True, shell = True)

        high_level_simulation.write_digital_estimation_fir_to_csv_file(high_level_simulation.simulate_digital_estimator_fir())

        sim_xrun.wait()
        pass_count: int = 0
        fail_count: int = 0
        while True:
            line: str = sim_xrun.stdout.readline().removesuffix("\n")
            if line == "":
                break
            else:
                if line.startswith("PASS") == True:
                    pass_count += 1
                elif line.startswith("FAIL") == True:
                    fail_count += 1
                print(line)
        
        print("\n\nStatistics:")
        if fail_count == 0:
            print("All checked assertions met!")
        else:
            print(f"{pass_count} out of {pass_count + fail_count} visited assertions were met.")
            print(f"{fail_count} assertions failed!")

        high_level_simulation.compare_simulation_system_verilog_to_high_level()

        high_level_simulation.plot_results()


    def write_xrun_simulation_file(self, name: str, settings: list[str]):
        simulation_file: FileGenerator.FileGenerator = FileGenerator.FileGenerator()
        simulation_file.set_path(self.path)
        simulation_file.set_name(name)
        simulation_file.open_output_file()
        for line in settings:
            simulation_file.write_line_linebreak(line)
        simulation_file.close_output_file()
        Path(self.path + "/" + name).chmod(S_IRWXU)


    def write_xrun_options_file(self, name: str, options: list[str]):
        options_file: FileGenerator.FileGenerator = FileGenerator.FileGenerator()
        options_file.set_path(self.path)
        options_file.set_name(name)
        options_file.open_output_file()
        for line in options:
            options_file.write_line_linebreak(line)
        options_file.close_output_file()


if __name__ == '__main__':
    digital_estimator_generator: DigitalEstimatorGenerator = DigitalEstimatorGenerator()
    digital_estimator_generator.generate()