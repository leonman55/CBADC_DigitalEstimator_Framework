from asyncio.subprocess import PIPE
from genericpath import isfile
import os
import platform
import subprocess

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


def main():
    configuration_number_of_timesteps_in_clock_cycle: int = 10
    configuration_n_number_of_analog_states: int = 5
    configuration_m_number_of_digital_states: int = configuration_n_number_of_analog_states
    configuration_beta: float = 6250.0
    configuration_rho: float = -1e-2
    configuration_kappa: float = -1.0
    configuration_eta2: float = 1e7
    configuration_lookback_length: int = 32
    configuration_lookahead_length: int = 32
    configuration_fir_data_width: int = 16
    configuration_fir_lut_input_width: int = 4
    configuration_sumulation_length: int = 2 << 14

    path: str = ""
    if platform.system() == "Linux":
        #path = "./GeneratedSystemVerilogFiles/"
        path = "../df/sim/SystemVerilogFiles/"
        sim_folder = "../df/sim/"
    elif platform.system() == "Windows":
        #path = ".\\GeneratedSystemVerilogFiles\\"
        path = "..\\df\\sim\\SystemVerilogFiles\\"
        sim_folder = "..\\df\\sim\\"

    digital_estimator_testbench: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorVerificationModules.DigitalEstimatorTestbench.DigitalEstimatorTestbench(path, "DigitalEstimatorTestbench")
    digital_estimator_testbench.configuration_number_of_timesteps_in_clock_cycle = configuration_number_of_timesteps_in_clock_cycle
    digital_estimator_testbench.configuration_rho = configuration_rho
    digital_estimator_testbench.configuration_beta = configuration_beta
    digital_estimator_testbench.configuration_eta2 = configuration_eta2
    digital_estimator_testbench.configuration_kappa = configuration_kappa
    digital_estimator_testbench.configuration_lookback_length = configuration_lookback_length
    digital_estimator_testbench.configuration_lookahead_length = configuration_lookahead_length
    digital_estimator_testbench.configuration_n_number_of_analog_states = configuration_n_number_of_analog_states
    digital_estimator_testbench.configuration_m_number_of_digital_states = configuration_m_number_of_digital_states
    digital_estimator_testbench.configuration_fir_data_width = configuration_fir_data_width
    digital_estimator_testbench.configuration_fir_lut_input_width = configuration_fir_lut_input_width
    digital_estimator_testbench.configuration_simulation_length = configuration_sumulation_length
    digital_estimator_testbench.generate()

    digital_estimator: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorModules.DigitalEstimatorWrapper.DigitalEstimatorWrapper(path, "DigitalEstimator")
    digital_estimator.generate()
    look_up_table: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorModules.LookUpTable.LookUpTable(path, "LookUpTable")
    look_up_table.generate()
    look_up_table_block: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorModules.LookUpTableBlock.LookUpTableBlock(path, "LookUpTableBlock")
    look_up_table_block.generate()
    adder_combinatorial: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorModules.AdderCombinatorial.AdderCombinatorial(path, "AdderCombinatorial")
    adder_combinatorial.generate()
    adder_block_combinatorial: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorModules.AdderBlockCombinatorial.AdderBlockCombinatorial(path, "AdderBlockCombinatorial")
    adder_block_combinatorial.generate()

    high_level_simulation: CBADC_HighLevelSimulation.DigitalEstimatorParameterGenerator = CBADC_HighLevelSimulation.DigitalEstimatorParameterGenerator(
        configuration_n_number_of_analog_states,
        configuration_m_number_of_digital_states,
        configuration_beta,
        configuration_rho,
        configuration_kappa,
        configuration_eta2,
        configuration_lookback_length,
        configuration_lookahead_length,
        configuration_fir_data_width,
        size = configuration_sumulation_length
    )
    high_level_simulation.write_control_signal_to_csv_file(high_level_simulation.simulate_analog_system())

    options: list[str] = list[str]()
    options.append("-top " + digital_estimator_testbench.name)
    write_xrun_options_file(path, "xrun_options", options)
    #sim_xrun: subprocess.CompletedProcess = subprocess.run(sim_folder + "sim.sh", shell = True)
    sim_xrun = subprocess.Popen(["./sim.sh"], cwd = sim_folder, stdout = PIPE, text = True, shell = True)
    sim_xrun.wait()
    """pass_count: int = 0
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
        print(f"{fail_count} assertions failed!")"""


def write_xrun_options_file(path: str, name: str, options: list[str]):
    options_file: FileGenerator.FileGenerator = FileGenerator.FileGenerator()
    options_file.set_path(path)
    options_file.set_name(name)
    options_file.open_output_file()
    for line in options:
        options_file.write_line_linebreak(line)
    options_file.close_output_file()


if __name__ == '__main__':
    main()
