from asyncio.subprocess import PIPE
from genericpath import isfile
import os
import platform
import subprocess
from pathlib import Path
from stat import *
import math
import numpy as np
import shutil

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
import DigitalEstimatorModules.ClockDivider
import DigitalEstimatorModules.InputDownsampleAccumulateRegister
import DigitalEstimatorModules.GrayCounter
import DigitalEstimatorModules.GrayCodeToBinary
import DigitalEstimatorModules.LookUpTableCoefficientRegister
import DigitalEstimatorModules.ValidCounter
import DigitalEstimatorModules.AdderSynchronous
import DigitalEstimatorModules.AdderBlockSynchronous
import DigitalEstimatorModules.LookUpTableSynchronous
import DigitalEstimatorModules.LookUpTableBlockSynchronous
import DigitalEstimatorVerificationModules.AdderCombinatorialAssertions
import DigitalEstimatorVerificationModules.AdderSynchronousAssertions
import DigitalEstimatorVerificationModules.AdderBlockCombinatorialAssertions
import DigitalEstimatorVerificationModules.AdderBlockSynchronousAssertions
import DigitalEstimatorVerificationModules.LookUpTableAssertions
import DigitalEstimatorVerificationModules.LookUpTableSynchronousAssertions
import DigitalEstimatorVerificationModules.LookUpTableBlockAssertions
import DigitalEstimatorVerificationModules.LookUpTableBlockSynchronousAssertions
import DigitalEstimatorVerificationModules.ClockDividerAssertions
import DigitalEstimatorVerificationModules.GrayCodeToBinaryAssertions
import DigitalEstimatorVerificationModules.GrayCounterAssertions
import DigitalEstimatorVerificationModules.InputDownsampleAccumulateRegisterAssertions
import CBADC_HighLevelSimulation


class DigitalEstimatorGenerator():
    """This is the main class for constructing Digital Estimation filter
        implementations in SystemVerilog.

    With this class Digital Estimation filter implementations in SystemVerilog
        can be constructed. The class is configurable so that it can be easily
        matched with the desired analog system and digital control configuration.
    It not only includes the construction of the desired filter, it also provides
        test automation.

    Attributes
    ----------
    path: str
        The base path, where the generated SystemVerilog files are placed, the
            simulation is carried out, etc.
    configuration_number_of_timesteps_in_clock_cycle: int
        Can be used to alter the RTL simulation timing setting.
    configuration_n_number_of_analog_states: int
        Sets the number of analog states
    configuration_m_number_of_digital_states: int
        Sets the number of digital states
    configuration_lookback_length: int
        Sets the size of the lookback batch for the estimation calculation
    configuration_lookahead_length: int
        Sets the size of the lookahead batch for the estimation calculation
    configuration_fir_data_width: int
        Sets the data width of the filter coefficients and resulting from that
            the size of datapath of the adders.
    configuration_fir_lut_input_width: int
        Sets the size of the LUT select input in bits.
    configuration_simulation_length: int
        Sets the simulation length of the cbadc Python high level simulation and
            the RTL simulation. Note that internally the simulation steps
            carried out will be multiplied by the oversampling rate, leading to
            a larger and more time consuming simulation.
    configuration_over_sample_rate: int
        The oversampling rate of the digital control unit.
    configuration_down_sample_rate: int
        The downsample rate of the digital estimation filter.
    configuration_counter_type: str
        Sets the type of the counter used in the downsample clock generator.
        Available options:
        "binary": Normal binary numbers used.
        "gray": Gray code numbers used.
    configuration_combinatorial_synchronous: str
        Sets if the filter is constructed as a combinatorial or synchronous
            implementation.
    configuration_required_snr_db: float
        Gives a lower bound for the SNR, which should be surpassed.
    high_level_simulation: CBADC_HighLevelSimulation.DigitalEstimatorParameterGenerator
        The instance of the cbadc simulator used to simulate the design and
            generate the filter coefficients.
    """

    path: str = "../df/sim/SystemVerilogFiles3"
    path_synthesis: str = "../df/src/SystemVerilogFiles3"

    configuration_number_of_timesteps_in_clock_cycle: int = 10
    configuration_n_number_of_analog_states: int = 2
    configuration_m_number_of_digital_states: int = configuration_n_number_of_analog_states
    configuration_lookback_length: int = 128
    configuration_lookahead_length: int = 128
    configuration_fir_data_width: int = 21
    configuration_fir_lut_input_width: int = 4
    configuration_simulation_length: int = 1 << 12
    configuration_over_sample_rate: int = 32
    configuration_down_sample_rate: int = configuration_over_sample_rate
    configuration_counter_type: str = "gray"
    configuration_combinatorial_synchronous: str = "synchronous"
    configuration_required_snr_db: float = 55
    configuration_coefficients_variable_fixed: str = "variable"

    high_level_simulation: CBADC_HighLevelSimulation.DigitalEstimatorParameterGenerator = None

    top_module_name: str = "DigitalEstimator"

    module_list: list[SystemVerilogModule.SystemVerilogModule] = list[SystemVerilogModule.SystemVerilogModule]()


    def generate(self) -> int:
        """Generates the SystemVerilog files of the digital estimator
            according to the specifications.

        This should be called after setting all wanted configuration options.
        It generates all SystemVerilog modules, the input stimuli and the
            scripts needed for simulation.
        """
        self.configuration_m_number_of_digital_states = self.configuration_n_number_of_analog_states
        self.configuration_down_sample_rate = self.configuration_over_sample_rate

        directory: str = self.path
        Path(directory).mkdir(parents = True, exist_ok = True)
        assert Path.exists(Path(directory))

        self.high_level_simulation = CBADC_HighLevelSimulation.DigitalEstimatorParameterGenerator(
            path = self.path,
            n_number_of_analog_states = self.configuration_n_number_of_analog_states,
            m_number_of_digital_states = self.configuration_m_number_of_digital_states,
            k1 = self.configuration_lookback_length,
            k2 = self.configuration_lookahead_length,
            data_width = self.configuration_fir_data_width,
            size = self.configuration_simulation_length,
            OSR = self.configuration_over_sample_rate,
            down_sample_rate = self.configuration_down_sample_rate
        )
        self.high_level_simulation.write_control_signal_to_csv_file(self.high_level_simulation.simulate_analog_system())

        self.top_module_name = "DigitalEstimator_" + str(self.configuration_n_number_of_analog_states) + "_" + str(self.configuration_over_sample_rate) + "_" + str(self.configuration_fir_data_width) + "_" + str(self.configuration_lookback_length) + "_" + str(self.configuration_lookahead_length) + "_" + str(self.configuration_fir_lut_input_width)

        digital_estimator_testbench: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorVerificationModules.DigitalEstimatorTestbench.DigitalEstimatorTestbench(self.path, "DigitalEstimatorTestbench")
        digital_estimator_testbench.configuration_number_of_timesteps_in_clock_cycle = self.configuration_number_of_timesteps_in_clock_cycle
        digital_estimator_testbench.configuration_lookback_length = self.configuration_lookback_length
        digital_estimator_testbench.configuration_lookahead_length = self.configuration_lookahead_length
        digital_estimator_testbench.configuration_n_number_of_analog_states = self.configuration_n_number_of_analog_states
        digital_estimator_testbench.configuration_m_number_of_digital_states = self.configuration_m_number_of_digital_states
        digital_estimator_testbench.configuration_fir_data_width = self.configuration_fir_data_width
        digital_estimator_testbench.configuration_fir_lut_input_width = self.configuration_fir_lut_input_width
        digital_estimator_testbench.configuration_simulation_length = self.configuration_simulation_length
        digital_estimator_testbench.configuration_down_sample_rate = self.configuration_down_sample_rate
        digital_estimator_testbench.configuration_over_sample_rate = self.configuration_over_sample_rate
        digital_estimator_testbench.high_level_simulation = self.high_level_simulation
        digital_estimator_testbench.configuration_downsample_clock_counter_type = self.configuration_counter_type
        digital_estimator_testbench.configuration_combinatorial_synchronous = self.configuration_combinatorial_synchronous
        digital_estimator_testbench.top_module_name = self.top_module_name
        digital_estimator_testbench.generate()

        digital_estimator: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorModules.DigitalEstimatorWrapper.DigitalEstimatorWrapper(self.path, "DigitalEstimator")
        digital_estimator.configuration_n_number_of_analog_states = self.configuration_n_number_of_analog_states
        digital_estimator.configuration_m_number_of_digital_states = self.configuration_m_number_of_digital_states
        digital_estimator.configuration_fir_lut_input_width = self.configuration_fir_lut_input_width
        digital_estimator.configuration_data_width = self.configuration_fir_data_width
        digital_estimator.configuration_lookback_length = self.configuration_lookback_length
        digital_estimator.configuration_lookahead_length = self.configuration_lookahead_length
        digital_estimator.configuration_down_sample_rate = self.configuration_down_sample_rate
        digital_estimator.configuration_combinatorial_synchronous = self.configuration_combinatorial_synchronous
        digital_estimator.module_name = self.top_module_name
        digital_estimator.generate()
        self.module_list.append(digital_estimator)

        if self.configuration_combinatorial_synchronous == "combinatorial":
            look_up_table: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorModules.LookUpTable.LookUpTable(self.path, "LookUpTable")
            look_up_table.generate()
            self.module_list.append(look_up_table)

            look_up_table_block: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorModules.LookUpTableBlock.LookUpTableBlock(self.path, "LookUpTableBlock")
            look_up_table_block.generate()
            self.module_list.append(look_up_table_block)

            adder_combinatorial: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorModules.AdderCombinatorial.AdderCombinatorial(self.path, "AdderCombinatorial")
            adder_combinatorial.generate()
            self.module_list.append(adder_combinatorial)

            adder_block_combinatorial: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorModules.AdderBlockCombinatorial.AdderBlockCombinatorial(self.path, "AdderBlockCombinatorial")
            adder_block_combinatorial.generate()
            self.module_list.append(adder_block_combinatorial)
        elif self.configuration_combinatorial_synchronous == "synchronous":
            adder_synchronous: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorModules.AdderSynchronous.AdderSynchronous(self.path, "AdderSynchronous")
            adder_synchronous.generate()
            self.module_list.append(adder_synchronous)

            adder_block_synchronous: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorModules.AdderBlockSynchronous.AdderBlockSynchronous(self.path, "AdderBlockSynchronous")
            adder_block_synchronous.generate()
            self.module_list.append(adder_block_synchronous)

            look_up_table_synchronous: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorModules.LookUpTableSynchronous.LookUpTableSynchronous(self.path, "LookUpTableSynchronous")
            look_up_table_synchronous.generate()
            self.module_list.append(look_up_table_synchronous)

            look_up_table_block_synchronous: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorModules.LookUpTableBlockSynchronous.LookUpTableBlockSynchronous(self.path, "LookUpTableBlockSynchronous")
            look_up_table_block_synchronous.generate()
            self.module_list.append(look_up_table_block_synchronous)

        if self.configuration_down_sample_rate > 1:
            input_downsample_accumulate_register: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorModules.InputDownsampleAccumulateRegister.InputDownsampleAccumulateRegister(self.path, "InputDownsampleAccumulateRegister")
            input_downsample_accumulate_register.configuration_down_sample_rate = self.configuration_down_sample_rate
            input_downsample_accumulate_register.configuration_data_width = self.configuration_m_number_of_digital_states
            input_downsample_accumulate_register.generate()
            self.module_list.append(input_downsample_accumulate_register)

            clock_divider: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorModules.ClockDivider.ClockDivider(self.path, "ClockDivider")
            clock_divider.configuration_down_sample_rate = self.configuration_down_sample_rate
            clock_divider.configuration_counter_type = self.configuration_counter_type
            clock_divider.generate()
            self.module_list.append(clock_divider)

            if self.configuration_counter_type == "gray":
                gray_counter: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorModules.GrayCounter.GrayCounter(self.path, "GrayCounter")
                gray_counter.configuration_counter_bit_width = math.ceil(math.log2(float(self.configuration_down_sample_rate * 2.0)))
                gray_counter.configuration_clock_edge = "both"
                gray_counter.generate()
                self.module_list.append(gray_counter)

                gray_code_to_binary: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorModules.GrayCodeToBinary.GrayCodeToBinary(self.path, "GrayCodeToBinary")
                gray_code_to_binary.configuration_bit_width = math.ceil(math.log2(float(self.configuration_down_sample_rate * 2.0)))
                gray_code_to_binary.generate()
                self.module_list.append(gray_code_to_binary)

        lookup_table_coefficient_register: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorModules.LookUpTableCoefficientRegister.LookUpTableCoefficientRegister(self.path, "LookUpTableCoefficientRegister")
        lookup_table_coefficient_register.configuration_lookup_table_data_width = self.configuration_fir_data_width
        lookup_table_coefficient_register.generate()
        self.module_list.append(lookup_table_coefficient_register)

        valid_counter: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorModules.ValidCounter.ValidCounter(self.path, "ValidCounter")
        valid_counter.configuration_top_value = self.configuration_lookback_length + self.configuration_lookahead_length
        valid_counter.generate()
        self.module_list.append(valid_counter)

        if self.configuration_combinatorial_synchronous == "combinatorial":
            adder_combinatorial_assertions: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorVerificationModules.AdderCombinatorialAssertions.AdderCombinatorialAssertions(self.path, "AdderCombinatorialAssertions")
            adder_combinatorial_assertions.configuration_adder_input_width = self.configuration_fir_lut_input_width
            adder_combinatorial_assertions.generate()

            adder_block_combinatorial_assertions: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorVerificationModules.AdderBlockCombinatorialAssertions.AdderBlockCombinatorialAssertions(self.path, "AdderBlockCombinatorialAssertions")
            adder_block_combinatorial_assertions.generate()

            look_up_table_assertions: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorVerificationModules.LookUpTableAssertions.LookUpTableAssertions(self.path, "LookUpTableAssertions")
            look_up_table_assertions.configuration_input_width = self.configuration_fir_lut_input_width
            look_up_table_assertions.configuration_data_width = self.configuration_fir_data_width
            look_up_table_assertions.generate()

            look_up_table_block_assertions: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorVerificationModules.LookUpTableBlockAssertions.LookUpTableBlockAssertions(self.path, "LookUpTableBlockAssertions")
            look_up_table_block_assertions.generate()
        else:
            adder_synchronous_assertions: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorVerificationModules.AdderSynchronousAssertions.AdderSynchronousAssertions(self.path, "AdderSynchronousAssertions")
            adder_synchronous_assertions.configuration_adder_input_width = self.configuration_fir_lut_input_width
            adder_synchronous_assertions.generate()

            adder_block_synchronous_assertions: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorVerificationModules.AdderBlockSynchronousAssertions.AdderBlockSynchronousAssertions(self.path, "AdderBlockSynchronousAssertions")
            adder_block_synchronous_assertions.generate()

            look_up_table_synchronous_assertions: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorVerificationModules.LookUpTableSynchronousAssertions.LookUpTableSynchronousAssertions(self.path, "LookUpTableSynchronousAssertions")
            look_up_table_synchronous_assertions.configuration_input_width = self.configuration_fir_lut_input_width
            look_up_table_synchronous_assertions.configuration_data_width = self.configuration_fir_data_width
            look_up_table_synchronous_assertions.generate()

            look_up_table_block_synchronous_assertions: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorVerificationModules.LookUpTableBlockSynchronousAssertions.LookUpTableBlockSynchronousAssertions(self.path, "LookUpTableBlockSynchronousAssertions")
            look_up_table_block_synchronous_assertions.generate()

        if self.configuration_down_sample_rate > 1:
            input_downsample_accumulate_register_assertions: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorVerificationModules.InputDownsampleAccumulateRegisterAssertions.InputDownsampleAccumulateRegisterAssertions(self.path, "InputDownsampleAccumulateRegisterAssertions")
            input_downsample_accumulate_register_assertions.generate()

            clock_divider_assertions: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorVerificationModules.ClockDividerAssertions.ClockDividerAssertions(self.path, "ClockDividerAssertions")
            clock_divider_assertions.configuration_down_sample_rate = self.configuration_down_sample_rate
            clock_divider_assertions.configuration_counter_type = self.configuration_counter_type
            clock_divider_assertions.generate()

            if self.configuration_counter_type == "gray":
                gray_code_to_binary_assertions: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorVerificationModules.GrayCodeToBinaryAssertions.GrayCodeToBinaryAssertions(self.path, "GrayCodeToBinaryAssertions")
                gray_code_to_binary_assertions.configuration_bit_size = math.ceil(math.log2(self.configuration_down_sample_rate * 2))
                gray_code_to_binary_assertions.generate()

                gray_counter_assertions: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorVerificationModules.GrayCounterAssertions.GrayCounterAssertions(self.path, "GrayCounterAssertions")
                gray_counter_assertions.generate()


        simulation_settings: list[str] = list[str]()
        simulation_settings.append("source ~/pro/acmos2/virtuoso/setup_user")
        simulation_settings.append("source ~/pro/fall2022/bash/setup_user")
        #simulation_settings.append("cd " + self.path)
        simulation_settings.append("xrun -f xrun_options")
        self.write_xrun_simulation_file("sim.sh", simulation_settings)

        simulation_gui_settings: list[str] = list[str]()
        simulation_gui_settings.append("source ~/pro/acmos2/virtuoso/setup_user")
        simulation_gui_settings.append("source ~/pro/fall2022/bash/setup_user")
        simulation_gui_settings.append("xrun -gui -f xrun_options")
        self.write_xrun_simulation_file("sim_gui.sh", simulation_gui_settings)

        options: list[str] = list[str]()
        #options.append("-gui")
        #options.append("-q")
        options.append("-access +rwc")
        options.append("-top " + digital_estimator_testbench.name)
        options.append("*.sv")
        self.write_xrun_options_file("xrun_options", options)

    def simulate(self):
        """Starts the simulation of the configured system and checks the outputs.

        This includes the cbadc Python high level simulation.
        A self programmed simulation, which was used to check the SystemVerilog implementation.
        The RTL simulation.
        The checks include checking the output logs of the RTL simulation,
        comparing the high level simulation against the RTL simulation
        and the achieved performance of the system.
        """
        high_level_fir_simulator =  self.high_level_simulation.simulate_digital_estimator_fir()
        #self.high_level_simulation.write_digital_estimation_fir_to_csv_file(self.high_level_simulation.simulate_digital_estimator_fir())
        self.high_level_simulation.write_digital_estimation_fir_to_csv_file(high_level_fir_simulator)

        #self.high_level_simulation.simulate_fir_filter_self_programmed(number_format = "both")
        #self.high_level_simulation.simulate_fir_filter_self_programmed(number_format = "float")
        #self.high_level_simulation.simulate_fir_filter_self_programmed(number_format = "integer")

        #sim_xrun: subprocess.CompletedProcess = subprocess.run(sim_folder + "sim.sh", shell = True)
        #sim_xrun = subprocess.Popen(["./sim.sh"], cwd = self.path, stdout = PIPE, text = True, shell = True)
        sim_xrun = subprocess.Popen(["./sim.sh"], cwd = self.path, text = True, shell = True)

        sim_xrun.wait()
        xrun_return: int = self.check_simulation_log()

        system_verilog_simulation_equals_high_level_simulation: bool = self.high_level_simulation.compare_simulation_system_verilog_to_high_level()

        self.high_level_simulation.plot_results()

        system_verilog_snr_db_check: int = self.check_rtl_snr()

        if xrun_return != 0:
            return 1, "Errors during RTL simulation!"
        elif system_verilog_simulation_equals_high_level_simulation == False:
            return 2, "SystemVerilog RTL simulation results deviated from high level Python simulation!"
        elif system_verilog_snr_db_check != 0:
            return 3, "Signal-to-Noise-Ratio (SNR) is too low!"
        else:
            return 0, "Test passed."

    def write_synthesis_scripts(self):
        try:
            #current_working_directory = Path.cwd()
            directory: str = self.path_synthesis
            Path(directory).mkdir(parents = True, exist_ok = True)
            assert Path.exists(Path(directory))

            with open(self.path_synthesis + "/sources.txt", mode = "w") as sources_file:
                for module in self.module_list:
                    module_filename: str = module.name + ".sv"
                    sources_file.write(module_filename + "\n")
                    source: str = self.path + "/" + module_filename
                    destination: str = self.path_synthesis + "/" + module_filename
                    shutil.copyfile(source, destination)

            script_lines: list[str] = list[str]()
            with open("../df/synth_template", mode = "r") as synthesize_script_old:
                script_lines = synthesize_script_old.readlines()
                for index in range(len(script_lines)):
                    if script_lines[index].find("DESIGN_TOP_MODULE=") != -1:
                        script_lines.pop(index)
                        script_lines.insert(index, "DESIGN_TOP_MODULE=\"" + self.top_module_name + "\"\n")
                        break
            with open(directory + "/synth", mode = "w") as synthesize_script:
                synthesize_script.writelines(script_lines)
            Path(directory + "/synth").chmod(S_IRWXU)

            shutil.copyfile("../df/config_syn_template.tcl", self.path_synthesis + "/config_syn.tcl")

            script_settings_lines: list[str] = list[str]()
            with open("../df/tcl/synth_template.tcl", mode = "r") as synthesize_settings_old:
                script_settings_lines = synthesize_settings_old.readlines()
                for index in range(len(script_settings_lines)):
                    if script_settings_lines[index].find("source ") != -1:
                        script_settings_lines.pop(index)
                        script_settings_lines.insert(index, "source ../../../" + self.path_synthesis + "/config_syn.tcl\n")
                        break
            with open(directory + "/synth.tcl", mode = "w") as synthesize_settings:
                synthesize_settings.writelines(script_settings_lines)
        except:
            pass

    def synthesize(self):
        """Starts the simulation of the configured system and checks the outputs.

        This includes the cbadc Python high level simulation.
        A self programmed simulation, which was used to check the SystemVerilog implementation.
        The RTL simulation.
        The checks include checking the output logs of the RTL simulation,
        comparing the high level simulation against the RTL simulation
        and the achieved performance of the system.
        """
        synth_genus = subprocess.Popen(["./synth"], cwd = self.path_synthesis, text = True, shell = True)
        synth_genus.wait()

    def write_xrun_simulation_file(self, name: str, settings: list[str]):
        """Generates the script for the RTL simulation.
        """
        simulation_file: FileGenerator.FileGenerator = FileGenerator.FileGenerator()
        simulation_file.set_path(self.path)
        simulation_file.set_name(name)
        simulation_file.open_output_file()
        for line in settings:
            simulation_file.write_line_linebreak(line)
        simulation_file.close_output_file()
        Path(self.path + "/" + name).chmod(S_IRWXU)


    def write_xrun_options_file(self, name: str, options: list[str]):
        """Generates the options file used for the simulation.
        """
        options_file: FileGenerator.FileGenerator = FileGenerator.FileGenerator()
        options_file.set_path(self.path)
        options_file.set_name(name)
        options_file.open_output_file()
        for line in options:
            options_file.write_line_linebreak(line)
        options_file.close_output_file()

    def check_simulation_log(self) -> int:
        """Checks the simulation log for errors.
        """
        pass_count: int = 0
        fail_count: int = 0
        with open(self.path + "/xrun.log", "r") as xrun_log_file:
            for line in xrun_log_file.readlines():
                line: str = line.removesuffix("\n")
                if line == "":
                    continue
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
        return fail_count

    def check_rtl_snr(self) -> int:
        """Checks the achieved SNR of the system for the set limit.
        """
        with open(self.path + "/digital_estimation_snr_db.csv") as digital_estimation_snr_db_csv_file:
            snr_db: float = float(digital_estimation_snr_db_csv_file.readline())
            if snr_db < self.configuration_required_snr_db or math.isnan(snr_db):
                return 1
            else:
                return 0


if __name__ == '__main__':
    """Main function for testing the implementation.

    This is mainly used for debugging purposes.
    Generating digital estimation filters should normally be done with the pytest framework.
    """
    digital_estimator_generator: DigitalEstimatorGenerator = DigitalEstimatorGenerator()
    digital_estimator_generator.generate()
    simulation_result: tuple[int, str] = digital_estimator_generator.simulate()
    if simulation_result[0] == 0:
        digital_estimator_generator.write_synthesis_scripts()
        #digital_estimator_generator.synthesize()