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
import gzip
import statistics

import FileGenerator
import SystemVerilogModule
import DigitalEstimatorModules.TestModule
import DigitalEstimatorModules.DigitalEstimatorWrapper
import DigitalEstimatorVerificationModules.DigitalEstimatorTestbench
import DigitalEstimatorModules.DigitalEstimatorWrapper
import DigitalEstimatorModules.LookUpTable
import DigitalEstimatorModules.LookUpTableBlock
import DigitalEstimatorModules.AdderCombinatorial
import DigitalEstimatorModules.AdderCombinatorialReducedSize
import DigitalEstimatorModules.AdderBlockCombinatorial
import DigitalEstimatorModules.ClockDivider
import DigitalEstimatorModules.InputDownsampleAccumulateRegister
import DigitalEstimatorModules.GrayCounter
import DigitalEstimatorModules.GrayCodeToBinary
import DigitalEstimatorModules.LookUpTableCoefficientRegister
import DigitalEstimatorModules.ValidCounter
import DigitalEstimatorModules.AdderSynchronous
import DigitalEstimatorModules.AdderSynchronousReducedSize
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

    #path: str = "../df/sim/SystemVerilogFiles"
    path: str = "/local_work/leonma/sim/SystemVerilogFiles24"
    #path_synthesis: str = "../df/src/SystemVerilogFiles"
    path_synthesis: str = "/local_work/leonma/src/SystemVerilogFiles24"

    scripts_base_folder = "../df/scripts"


    configuration_number_of_timesteps_in_clock_cycle: int = 10
    configuration_analog_bandwidth: int = 20 * 1e6
    configuration_n_number_of_analog_states: int = 3
    #configuration_n_number_of_analog_states: int = 7
    configuration_m_number_of_digital_states: int = configuration_n_number_of_analog_states
    configuration_lookback_length: int = 128
    #configuration_lookback_length: int = 256
    configuration_lookahead_length: int = 128
    #configuration_lookahead_length: int = 256
    configuration_fir_data_width: int = 22
    #configuration_fir_data_width: int = 25
    configuration_fir_lut_input_width: int = 4
    configuration_simulation_length: int = 1 << 12
    configuration_over_sample_rate: int = 23
    #configuration_over_sample_rate: int = 8
    configuration_down_sample_rate: int = configuration_over_sample_rate
    configuration_counter_type: str = "binary"
    configuration_combinatorial_synchronous: str = "combinatorial"
    configuration_required_snr_db: float = 55
    configuration_coefficients_variable_fixed: str = "variable"
    configuration_reduce_size_coefficients: bool = True
    configuration_reduce_size_luts: bool = True
    configuration_reduce_size_adders: bool = True

    high_level_simulation: CBADC_HighLevelSimulation.DigitalEstimatorParameterGenerator = None

    top_module_name: str = "DigitalEstimator"

    module_list: list[SystemVerilogModule.SystemVerilogModule] = list[SystemVerilogModule.SystemVerilogModule]()
    simulation_module_list: list[SystemVerilogModule.SystemVerilogModule] = list[SystemVerilogModule.SystemVerilogModule]()


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
            bandwidth = self.configuration_analog_bandwidth,
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

        self.top_module_name = "DigitalEstimator_" + str(self.configuration_n_number_of_analog_states) + "_" + str(self.configuration_over_sample_rate) + "_" + str(self.configuration_fir_data_width) + "_" + str(self.configuration_lookback_length) + "_" + str(self.configuration_lookahead_length) + "_" + str(self.configuration_fir_lut_input_width) + "_" + str(self.configuration_coefficients_variable_fixed)
        
        lookback_lut_entries = None
        lookback_lut_entries_mapped = None
        lookback_lut_entries_bit_mapping = None
        lookback_lut_entries_max_widths = None
        lookback_lut_entries_max_widths_average = None
        lookback_lut_entries_max_widths_sorted = None
        lookback_lut_entries_mapped_reordered = None
        lookahead_lut_entries = None
        lookahead_lut_entries_mapped = None
        lookahead_lut_entries_bit_mapping = None
        lookahead_lut_entries_max_widths = None
        lookahead_lut_entries_max_widths_average = None
        lookahead_lut_entries_max_widths_sorted = None
        lookahead_lut_entries_mapped_reordered = None
        
        self.module_list = list[SystemVerilogModule.SystemVerilogModule]()
        self.simulation_module_list = list[SystemVerilogModule.SystemVerilogModule]()

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
        digital_estimator_testbench.configuration_coefficients_variable_fixed = self.configuration_coefficients_variable_fixed
        digital_estimator_testbench.configuration_mapped_simulation = False
        digital_estimator_testbench.configuration_placedandrouted_simulation = False
        digital_estimator_testbench.top_module_name = self.top_module_name
        digital_estimator_testbench.configuration_reduce_size_coefficients = self.configuration_reduce_size_coefficients
        digital_estimator_testbench.configuration_reduce_size_luts = self.configuration_reduce_size_luts
        digital_estimator_testbench.configuration_reduce_size_adders = self.configuration_reduce_size_adders
        digital_estimator_testbench.generate()
        self.simulation_module_list.append(digital_estimator_testbench)
        
        if self.configuration_reduce_size_coefficients == True:
            lookback_lut_entries = CBADC_HighLevelSimulation.convert_coefficient_matrix_to_lut_entries(self.high_level_simulation.fir_hb_matrix, self.configuration_fir_lut_input_width)
            lookback_lut_entries_mapped = CBADC_HighLevelSimulation.map_lut_entries_to_luts(lut_entries = lookback_lut_entries, lut_input_width = self.configuration_fir_lut_input_width)
            lookback_lut_entries_bit_mapping: tuple[list[list[int]], int] = CBADC_HighLevelSimulation.get_lut_entry_bit_mapping(lut_entry_matrix = lookback_lut_entries, lut_input_width = self.configuration_fir_lut_input_width)
            lookback_lut_entries_max_widths = CBADC_HighLevelSimulation.get_maximum_bitwidth_per_lut(lookback_lut_entries_bit_mapping[0])
            lookback_lut_entries_max_widths_average = statistics.mean(lookback_lut_entries_max_widths)
            lookback_lut_entries_max_widths_sorted: list[tuple[int, int]] = CBADC_HighLevelSimulation.sort_luts_by_size(lookback_lut_entries_max_widths)
            lookback_lut_entries_mapped_reordered: list[list[int]] = CBADC_HighLevelSimulation.reorder_lut_entries(lookback_lut_entries_mapped, lookback_lut_entries_max_widths_sorted)
            print("LUT entries: \n", lookback_lut_entries)
            print("Mapped LUT entries: \n", lookback_lut_entries_mapped)
            print("LUT entry bit mapping: \n", lookback_lut_entries_bit_mapping[0])
            print("Maximum widths per LUT: \n", lookback_lut_entries_max_widths)
            print("Average bit width: ", lookback_lut_entries_bit_mapping[1])
            print("Possible savings on registers with average bit width: ", str(100.0 * (self.configuration_fir_data_width - lookback_lut_entries_bit_mapping[1]) / self.configuration_fir_data_width), "%")
            print("Average bit width: ", lookback_lut_entries_max_widths_average)
            print("Possible savings on registers with average maximum bit width: ", str(100.0 * (self.configuration_fir_data_width - lookback_lut_entries_max_widths_average) / self.configuration_fir_data_width), "%")
            print("Sorted maximum widhts per LUT: \n", lookback_lut_entries_max_widths_sorted)
            print("Reordered mapped LUT entries: \n", lookback_lut_entries_mapped_reordered)
            lookahead_lut_entries = CBADC_HighLevelSimulation.convert_coefficient_matrix_to_lut_entries(self.high_level_simulation.fir_hf_matrix, self.configuration_fir_lut_input_width)
            lookahead_lut_entries_mapped = CBADC_HighLevelSimulation.map_lut_entries_to_luts(lut_entries = lookahead_lut_entries, lut_input_width = self.configuration_fir_lut_input_width)
            lookahead_lut_entries_bit_mapping: tuple[list[list[int]], int] = CBADC_HighLevelSimulation.get_lut_entry_bit_mapping(lut_entry_matrix = lookahead_lut_entries, lut_input_width = self.configuration_fir_lut_input_width)
            lookahead_lut_entries_max_widths = CBADC_HighLevelSimulation.get_maximum_bitwidth_per_lut(lookahead_lut_entries_bit_mapping[0])
            lookahead_lut_entries_max_widths_average = statistics.mean(lookahead_lut_entries_max_widths)
            lookahead_lut_entries_max_widths_sorted: list[tuple[int, int]] = CBADC_HighLevelSimulation.sort_luts_by_size(lookahead_lut_entries_max_widths)
            lookahead_lut_entries_mapped_reordered: list[list[int]] = CBADC_HighLevelSimulation.reorder_lut_entries(lookahead_lut_entries_mapped, lookahead_lut_entries_max_widths_sorted)
            print("LUT entries: \n", lookahead_lut_entries)
            print("Mapped LUT entries: \n", lookahead_lut_entries_mapped)
            print("LUT entry bit mapping: \n", lookahead_lut_entries_bit_mapping[0])
            print("Maximum widths per LUT: \n", lookahead_lut_entries_max_widths)
            print("Average bit width: ", lookahead_lut_entries_bit_mapping[1])
            print("Possible savings on registers with average bit width: ", str(100.0 * (self.configuration_fir_data_width - lookahead_lut_entries_bit_mapping[1]) / self.configuration_fir_data_width), "%")
            print("Average bit width: ", lookahead_lut_entries_max_widths_average)
            print("Possible savings on registers with average maximum bit width: ", str(100.0 * (self.configuration_fir_data_width - lookahead_lut_entries_max_widths_average) / self.configuration_fir_data_width), "%")
            print("Sorted maximum widhts per LUT: \n", lookahead_lut_entries_max_widths_sorted)
            print("Reordered mapped LUT entries: \n", lookahead_lut_entries_mapped_reordered)

        digital_estimator: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorModules.DigitalEstimatorWrapper.DigitalEstimatorWrapper(self.path, "DigitalEstimator")
        digital_estimator.configuration_n_number_of_analog_states = self.configuration_n_number_of_analog_states
        digital_estimator.configuration_m_number_of_digital_states = self.configuration_m_number_of_digital_states
        digital_estimator.configuration_fir_lut_input_width = self.configuration_fir_lut_input_width
        digital_estimator.configuration_data_width = self.configuration_fir_data_width
        digital_estimator.configuration_lookback_length = self.configuration_lookback_length
        digital_estimator.configuration_lookahead_length = self.configuration_lookahead_length
        digital_estimator.configuration_down_sample_rate = self.configuration_down_sample_rate
        digital_estimator.configuration_combinatorial_synchronous = self.configuration_combinatorial_synchronous
        digital_estimator.configuration_coefficients_variable_fixed = self.configuration_coefficients_variable_fixed
        digital_estimator.configuration_reduce_size_coefficients = self.configuration_reduce_size_coefficients
        digital_estimator.configuration_reduce_size_luts = self.configuration_reduce_size_luts
        digital_estimator.configuration_reduce_size_adders = self.configuration_reduce_size_adders
        digital_estimator.high_level_simulation = self.high_level_simulation
        if self.configuration_reduce_size_coefficients == True:
            digital_estimator.lookback_mapped_reordered_lut_entries = lookback_lut_entries_mapped_reordered
            digital_estimator.lookback_mapped_reordered_bit_widths = lookback_lut_entries_max_widths_sorted
            digital_estimator.lookahead_mapped_reordered_lut_entries = lookahead_lut_entries_mapped_reordered
            digital_estimator.lookahead_mapped_reordered_bit_widths = lookahead_lut_entries_max_widths_sorted
        digital_estimator.module_name = self.top_module_name
        digital_estimator.generate()
        self.module_list.append(digital_estimator)

        if self.configuration_combinatorial_synchronous == "combinatorial":
            if self.configuration_reduce_size_adders == False:
                adder_combinatorial: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorModules.AdderCombinatorial.AdderCombinatorial(self.path, "AdderCombinatorial")
                adder_combinatorial.generate()
                self.module_list.append(adder_combinatorial)

                adder_block_combinatorial: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorModules.AdderBlockCombinatorial.AdderBlockCombinatorial(self.path, "AdderBlockCombinatorial")
                adder_block_combinatorial.generate()
                self.module_list.append(adder_block_combinatorial)
            elif self.configuration_reduce_size_adders == True:
                adder_combinatorial_reduced_size: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorModules.AdderCombinatorialReducedSize.AdderCombinatorialReducedSize(self.path, "AdderCombinatorial")
                adder_combinatorial_reduced_size.generate()
                self.module_list.append(adder_combinatorial_reduced_size)
                
            look_up_table: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorModules.LookUpTable.LookUpTable(self.path, "LookUpTable")
            look_up_table.generate()
            self.module_list.append(look_up_table)

            if self.configuration_reduce_size_luts == False:
                look_up_table_block: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorModules.LookUpTableBlock.LookUpTableBlock(self.path, "LookUpTableBlock")
                look_up_table_block.generate()
                self.module_list.append(look_up_table_block)

        elif self.configuration_combinatorial_synchronous == "synchronous":
            if self.configuration_reduce_size_adders == False:
                adder_synchronous: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorModules.AdderSynchronous.AdderSynchronous(self.path, "AdderSynchronous")
                adder_synchronous.generate()
                self.module_list.append(adder_synchronous)
                
                adder_block_synchronous: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorModules.AdderBlockSynchronous.AdderBlockSynchronous(self.path, "AdderBlockSynchronous")
                adder_block_synchronous.generate()
                self.module_list.append(adder_block_synchronous)
            elif self.configuration_reduce_size_adders == True:
                adder_synchronous_reduced_size: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorModules.AdderSynchronousReducedSize.AdderSynchronousReducedSize(self.path, "AdderSynchronous")
                adder_synchronous_reduced_size.generate()
                self.module_list.append(adder_synchronous_reduced_size)

            look_up_table_synchronous: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorModules.LookUpTableSynchronous.LookUpTableSynchronous(self.path, "LookUpTableSynchronous")
            look_up_table_synchronous.generate()
            self.module_list.append(look_up_table_synchronous)

            if self.configuration_reduce_size_luts == False:
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
                gray_counter.configuration_clock_edge = "posedge"
                gray_counter.generate()
                self.module_list.append(gray_counter)

                gray_code_to_binary: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorModules.GrayCodeToBinary.GrayCodeToBinary(self.path, "GrayCodeToBinary")
                gray_code_to_binary.configuration_bit_width = math.ceil(math.log2(float(self.configuration_down_sample_rate * 2.0)))
                gray_code_to_binary.generate()
                self.module_list.append(gray_code_to_binary)

        if self.configuration_coefficients_variable_fixed == "variable":
            lookup_table_coefficient_register: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorModules.LookUpTableCoefficientRegister.LookUpTableCoefficientRegister(self.path, "LookUpTableCoefficientRegister")
            lookup_table_coefficient_register.configuration_lookup_table_data_width = self.configuration_fir_data_width
            lookup_table_coefficient_register.configuration_lookback_lookup_table_entries_count = self.configuration_lookback_length
            lookup_table_coefficient_register.configuration_lookahead_lookup_table_entries_count = self.configuration_lookahead_length
            lookup_table_coefficient_register.configuration_reduce_size_coefficients = self.configuration_reduce_size_coefficients
            if self.configuration_reduce_size_coefficients == True:
                lookup_table_coefficient_register.lookback_mapped_reordered_lut_entries = lookback_lut_entries_mapped_reordered
                lookup_table_coefficient_register.lookback_mapped_reordered_bit_widths = lookback_lut_entries_max_widths_sorted
                lookup_table_coefficient_register.lookahead_mapped_reordered_lut_entries = lookahead_lut_entries_mapped_reordered
                lookup_table_coefficient_register.lookahead_mapped_reordered_bit_widths = lookahead_lut_entries_max_widths_sorted
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
            self.simulation_module_list.append(adder_combinatorial_assertions)

            adder_block_combinatorial_assertions: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorVerificationModules.AdderBlockCombinatorialAssertions.AdderBlockCombinatorialAssertions(self.path, "AdderBlockCombinatorialAssertions")
            adder_block_combinatorial_assertions.generate()
            self.simulation_module_list.append(adder_block_combinatorial_assertions)

            look_up_table_assertions: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorVerificationModules.LookUpTableAssertions.LookUpTableAssertions(self.path, "LookUpTableAssertions")
            look_up_table_assertions.configuration_input_width = self.configuration_fir_lut_input_width
            look_up_table_assertions.configuration_data_width = self.configuration_fir_data_width
            look_up_table_assertions.generate()
            self.simulation_module_list.append(look_up_table_assertions)

            look_up_table_block_assertions: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorVerificationModules.LookUpTableBlockAssertions.LookUpTableBlockAssertions(self.path, "LookUpTableBlockAssertions")
            look_up_table_block_assertions.generate()
            self.simulation_module_list.append(look_up_table_block_assertions)
        else:
            if self.configuration_reduce_size_adders == False:
                adder_synchronous_assertions: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorVerificationModules.AdderSynchronousAssertions.AdderSynchronousAssertions(self.path, "AdderSynchronousAssertions")
                adder_synchronous_assertions.configuration_adder_input_width = self.configuration_fir_lut_input_width
                adder_synchronous_assertions.generate()
                self.simulation_module_list.append(adder_synchronous_assertions)

            if self.configuration_reduce_size_adders == False:
                adder_block_synchronous_assertions: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorVerificationModules.AdderBlockSynchronousAssertions.AdderBlockSynchronousAssertions(self.path, "AdderBlockSynchronousAssertions")
                adder_block_synchronous_assertions.generate()
                self.simulation_module_list.append(adder_block_synchronous_assertions)

            look_up_table_synchronous_assertions: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorVerificationModules.LookUpTableSynchronousAssertions.LookUpTableSynchronousAssertions(self.path, "LookUpTableSynchronousAssertions")
            look_up_table_synchronous_assertions.configuration_input_width = self.configuration_fir_lut_input_width
            look_up_table_synchronous_assertions.configuration_data_width = self.configuration_fir_data_width
            look_up_table_synchronous_assertions.generate()
            self.simulation_module_list.append(look_up_table_synchronous_assertions)

            if self.configuration_reduce_size_coefficients == False:
                look_up_table_block_synchronous_assertions: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorVerificationModules.LookUpTableBlockSynchronousAssertions.LookUpTableBlockSynchronousAssertions(self.path, "LookUpTableBlockSynchronousAssertions")
                look_up_table_block_synchronous_assertions.generate()
                self.simulation_module_list.append(look_up_table_block_synchronous_assertions)

        if self.configuration_down_sample_rate > 1:
            input_downsample_accumulate_register_assertions: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorVerificationModules.InputDownsampleAccumulateRegisterAssertions.InputDownsampleAccumulateRegisterAssertions(self.path, "InputDownsampleAccumulateRegisterAssertions")
            input_downsample_accumulate_register_assertions.generate()
            self.simulation_module_list.append(input_downsample_accumulate_register_assertions)

            clock_divider_assertions: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorVerificationModules.ClockDividerAssertions.ClockDividerAssertions(self.path, "ClockDividerAssertions")
            clock_divider_assertions.configuration_down_sample_rate = self.configuration_down_sample_rate
            clock_divider_assertions.configuration_counter_type = self.configuration_counter_type
            clock_divider_assertions.generate()
            self.simulation_module_list.append(clock_divider_assertions)

            if self.configuration_counter_type == "gray":
                gray_code_to_binary_assertions: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorVerificationModules.GrayCodeToBinaryAssertions.GrayCodeToBinaryAssertions(self.path, "GrayCodeToBinaryAssertions")
                gray_code_to_binary_assertions.configuration_bit_size = math.ceil(math.log2(self.configuration_down_sample_rate * 2))
                gray_code_to_binary_assertions.generate()
                self.simulation_module_list.append(gray_code_to_binary_assertions)

                gray_counter_assertions: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorVerificationModules.GrayCounterAssertions.GrayCounterAssertions(self.path, "GrayCounterAssertions")
                gray_counter_assertions.generate()
                self.simulation_module_list.append(gray_counter_assertions)


        simulation_settings: list[str] = list[str]()
        simulation_settings.append("source ~/pro/acmos2/virtuoso/setup_user")
        simulation_settings.append("source ~/pro/fall2022/bash/setup_user")
        #simulation_settings.append("cd " + self.path)
        simulation_settings.append("xrun -f xrun_options")
        simulation_settings.append("rm -R xcelium.d/")
        self.write_xrun_simulation_file("sim.sh", simulation_settings)

        simulation_gui_settings: list[str] = list[str]()
        simulation_gui_settings.append("source ~/pro/acmos2/virtuoso/setup_user")
        simulation_gui_settings.append("source ~/pro/fall2022/bash/setup_user")
        simulation_gui_settings.append("xrun -gui -f xrun_options")
        simulation_gui_settings.append("rm -R xcelium.d/")
        self.write_xrun_simulation_file("sim_gui.sh", simulation_gui_settings)

        options: list[str] = list[str]()
        #options.append("-gui")
        #options.append("-q")
        options.append("-access +rwc")
        options.append("-top " + digital_estimator_testbench.name)
        #options.append("*.sv")
        for module in self.module_list:
            options.append(module.name + ".sv")
        for simulation_module in self.simulation_module_list:
            options.append(simulation_module.name + ".sv")
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

        os.rename(self.path + "/digital_estimation.csv", self.path + "/digital_estimation.xrun.csv")

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
        
    def copy_design_files_for_synthesis(self):
        directory: str = self.path_synthesis
        Path(directory).mkdir(parents = True, exist_ok = True)
        assert Path.exists(Path(directory))
        for module in self.module_list:
            module_filename: str = module.name + ".sv"
            source: str = self.path + "/" + module_filename
            destination: str = self.path_synthesis + "/" + module_filename
            shutil.copyfile(source, destination)
            
    def calculate_clock_periode_ns(self) -> float:
        number_of_significant_digits: int = 2
        clock_periode: float = 10**9 / (self.configuration_analog_bandwidth * 2 * self.configuration_over_sample_rate)
        clock_periode = float(math.trunc(clock_periode * 10**number_of_significant_digits)) / 10**number_of_significant_digits
        return clock_periode

    def write_synthesis_scripts_genus(self):
        try:
            Path(self.path_synthesis).mkdir(parents = True, exist_ok = True)
            assert Path.exists(Path(self.path_synthesis))

            with open(self.path_synthesis + "/sources.txt", mode = "w") as sources_file:
                for module in self.module_list:
                    module_filename: str = module.name + ".sv"
                    sources_file.write(module_filename + "\n")
                    #source: str = self.path + "/" + module_filename
                    #destination: str = self.path_synthesis + "/" + module_filename
                    #shutil.copyfile(source, destination)

            script_lines: list[str] = list[str]()
            with open(self.scripts_base_folder + "/synth_template", mode = "r") as synthesize_script_old:
                script_lines = synthesize_script_old.readlines()
                for index in range(len(script_lines)):
                    if script_lines[index].find("DESIGN_TOP_MODULE=") != -1:
                        script_lines.pop(index)
                        script_lines.insert(index, "DESIGN_TOP_MODULE=\"" + self.top_module_name + "\"\n")
                        break
            with open(self.path_synthesis + "/synth", mode = "w") as synthesize_script:
                synthesize_script.writelines(script_lines)
            Path(self.path_synthesis + "/synth").chmod(S_IRWXU)

            #shutil.copyfile(self.scripts_base_folder + "/config_syn_template.tcl", self.path_synthesis + "/config_syn.tcl")
            script_settings_lines: list[str] = list[str]()
            with open(self.scripts_base_folder + "/config_syn_template.tcl", mode = "r") as synthesize_configuration_old:
                script_settings_lines = synthesize_configuration_old.readlines()
                for index in range(len(script_settings_lines)):
                    if script_settings_lines[index].find("set TIMING_CLOCK_PERIOD") != -1:
                        script_settings_lines.pop(index)
                        script_settings_lines.insert(index, "set TIMING_CLOCK_PERIOD " + str(self.calculate_clock_periode_ns()) + "\n")
                        break
            with open(self.path_synthesis + "/config_syn.tcl", mode = "w") as synthesize_configuration:
                synthesize_configuration.writelines(script_settings_lines)

            script_settings_lines: list[str] = list[str]()
            with open(self.scripts_base_folder + "/synth_template.tcl", mode = "r") as synthesize_settings_old:
                script_settings_lines = synthesize_settings_old.readlines()
                for index in range(len(script_settings_lines)):
                    if script_settings_lines[index].find("source ") != -1:
                        script_settings_lines.pop(index)
                        script_settings_lines.insert(index, "source ../../config_syn.tcl\n")
                        break
            with open(self.path_synthesis + "/synth.tcl", mode = "w") as synthesize_settings:
                synthesize_settings.writelines(script_settings_lines)
        except:
            pass

    def synthesize_genus(self):
        """Starts the simulation of the configured system and checks the outputs.

        This includes the cbadc Python high level simulation.
        A self programmed simulation, which was used to check the SystemVerilog implementation.
        The RTL simulation.
        The checks include checking the output logs of the RTL simulation,
        comparing the high level simulation against the RTL simulation
        and the achieved performance of the system.
        """
        self.copy_design_files_for_synthesis()
        self.write_synthesis_scripts_genus()
        synth_genus = subprocess.Popen(["./synth"], cwd = self.path_synthesis, text = True, shell = True)
        synth_genus.wait()

    def write_synthesis_scripts_synopsys(self):
        try:
            directory: str = self.path_synthesis
            Path(directory).mkdir(parents = True, exist_ok = True)
            assert Path.exists(Path(directory))

            shutil.copyfile(self.scripts_base_folder + "/common_setup_template.tcl", self.path_synthesis + "/common_setup.tcl")
            shutil.copyfile(self.scripts_base_folder + "/dc_setup_template.tcl", self.path_synthesis + "/dc_setup.tcl")
            #shutil.copyfile(self.scripts_base_folder + "/dc_template.tcl", self.path_synthesis + "/dc.tcl")
            script_settings_lines: list[str] = list[str]()
            with open(self.scripts_base_folder + "/dc_template.tcl", mode = "r") as synthesize_settings_old:
                script_settings_lines = synthesize_settings_old.readlines()
                for index in range(len(script_settings_lines)):
                    if script_settings_lines[index].find("create_clock -name \"clk\"") != -1:
                        script_settings_lines.pop(index)
                        script_settings_lines.insert(index, f"create_clock -name \"clk\" -period {str(self.calculate_clock_periode_ns())} -waveform{{0.0 {str(self.calculate_clock_periode_ns() / 2.0)}}} [get_ports clk]")
                        break
            with open(self.path_synthesis + "/dc.tcl", mode = "w") as synthesize_settings:
                synthesize_settings.writelines(script_settings_lines)

            script_lines: list[str] = list[str]()
            with open(self.scripts_base_folder + "/runSynopsysDesignCompilerSynthesis_Template.sh", mode = "r") as synthesize_script_old:
                script_lines = synthesize_script_old.readlines()
                for index in range(len(script_lines)):
                    if script_lines[index].find("RTL_SOURCE_FILES=") != -1:
                        script_lines.pop(index)
                        design_files: str = "\""
                        for design_file_name in self.module_list:
                            design_files += design_file_name.name + ".sv "
                        design_files = design_files.removesuffix(" ") + "\""
                        script_lines.insert(index, "RTL_SOURCE_FILES=" + design_files + "\n")
                        #continue
                    if script_lines[index].find("DESIGN_NAME=") != -1:
                        script_lines.pop(index)
                        script_lines.insert(index, "DESIGN_NAME=\"" + self.top_module_name + "\"\n")
                        #continue
                    if script_lines[index].find("DESIGN_LABEL=") != -1:
                        script_lines.pop(index)
                        script_lines.insert(index, "DESIGN_LABEL=\"" + self.top_module_name + "\"\n")
                        #continue
            with open(directory + "/runSynopsysDesignCompilerSynthesis.sh", mode = "w") as synthesize_script:
                synthesize_script.writelines(script_lines)
            Path(directory + "/runSynopsysDesignCompilerSynthesis.sh").chmod(S_IRWXU)
        except:
            pass

    def synthesize_synopsys(self):
        """Starts the simulation of the configured system and checks the outputs.

        This includes the cbadc Python high level simulation.
        A self programmed simulation, which was used to check the SystemVerilog implementation.
        The RTL simulation.
        The checks include checking the output logs of the RTL simulation,
        comparing the high level simulation against the RTL simulation
        and the achieved performance of the system.
        """
        self.copy_design_files_for_synthesis()
        self.write_synthesis_scripts_synopsys()
        synth_synopsys = subprocess.Popen(["./runSynopsysDesignCompilerSynthesis.sh"], cwd = self.path_synthesis, text = True, shell = True)
        synth_synopsys.wait()

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
            
    def generate_vcs_simulation_file(self, name: str = "sim_vcs.sh"):
        """Generates the script for the RTL simulation with Synopsys VCS.
        """
        simulation_file: FileGenerator.FileGenerator = FileGenerator.FileGenerator()
        simulation_file.set_path(self.path)
        simulation_file.set_name(name)
        simulation_file.open_output_file()

        settings: list[str] = list[str]()
        settings.append("vcs -f vcs_options")
        settings.append("./simv")
        settings.append("rm simv")
        settings.append("rm -R ./csrc")
        settings.append("rm -R simv.daidir")
        #settings.append("rm -R waves.shm")
        for line in settings:
            simulation_file.write_line_linebreak(line)
        simulation_file.close_output_file()
        Path(self.path + "/" + name).chmod(S_IRWXU)

    def generate_vcs_options_file(self, name: str = "vcs_options"):
        """Generates the options file used for the RTL simulation with Synopsys VCS.
        """
        options_file: FileGenerator.FileGenerator = FileGenerator.FileGenerator()
        options_file.set_path(self.path)
        options_file.set_name(name)
        options_file.open_output_file()

        options: list[str] = list[str]()
        options.append("-sverilog")
        options.append("-timescale=1ns/1ps")
        for module in self.module_list:
            options.append(module.name + ".sv")
        for simulation_module in self.simulation_module_list:
            options.append(simulation_module.name + ".sv")
        for line in options:
            options_file.write_line_linebreak(line)
        options_file.close_output_file()

    def simulate_vcs(self):
        self.generate_vcs_simulation_file()
        self.generate_vcs_options_file()
        sim_vcs = subprocess.Popen(["./sim_vcs.sh"], cwd = self.path, text = True, shell = True)
        sim_vcs.wait()
        os.rename(self.path + "/digital_estimation.csv", self.path + "/digital_estimation.vcs.csv")
            
    def generate_xrun_mapped_simulation_script(self, name: str = "sim_mapped_genus.sh", synthesis_program: str = "genus"):
        mapped_simulation_script: FileGenerator.FileGenerator = FileGenerator.FileGenerator()
        mapped_simulation_script.set_path(self.path)
        mapped_simulation_script.set_name(name)
        mapped_simulation_script.open_output_file()
        mapped_settings: list[str] = list[str]()
        mapped_settings.append("source ~/pro/acmos2/virtuoso/setup_user")
        mapped_settings.append("source ~/pro/fall2022/bash/setup_user")
        mapped_settings.append("")
        mapped_settings.append("xrun -f xrun_options_mapped_" + synthesis_program)
        mapped_settings.append("rm -R xcelium.d/")
        for line in mapped_settings:
            mapped_simulation_script.write_line_linebreak(line)
        mapped_simulation_script.close_output_file()
        Path(self.path + "/" + name).chmod(S_IRWXU)

    def generate_xrun_mapped_simulation_options_file(self, name: str = "xrun_options_mapped_genus", synthesis_program: str = "genus"):
        mapped_options_file: FileGenerator.FileGenerator = FileGenerator.FileGenerator()
        mapped_options_file.set_path(self.path)
        mapped_options_file.set_name(name)
        mapped_options_file.open_output_file()
        mapped_options: list[str] = list[str]()
        mapped_options.append("-64bit")
        mapped_options.append("-access +rwc")
        mapped_options.append("-v /eda/kits/stm/28nm_fdsoi_v1.3a/C28SOI_SC_12_CORE_LR/5.1-05.81/behaviour/verilog/C28SOI_SC_12_CORE_LR.v")
        mapped_options.append("-v /eda/kits/stm/28nm_fdsoi_v1.3a/C28SOI_SC_12_CLK_LR/5.1-06.81/behaviour/verilog/C28SOI_SC_12_CLK_LR.v")
        mapped_options.append("-v /eda/kits/stm/28nm_fdsoi_v1.3a/C28SOI_SC_12_PR_LR/5.3.a-00.80/behaviour/verilog/C28SOI_SC_12_PR_LR.v")
        mapped_options.append("-timescale 1ns/1ps")
        mapped_options.append("-top DigitalEstimatorTestbench")
        mapped_options.append("-input xrun_mapped_" + synthesis_program + ".tcl")
        mapped_options.append("DigitalEstimatorTestbench_mapped_" + synthesis_program + "_xrun.sv")
        mapped_options.append(self.top_module_name + ".mapped." + synthesis_program + ".v")
        for line in mapped_options:
            mapped_options_file.write_line_linebreak(line)
        mapped_options_file.close_output_file()

    def generate_xrun_mapped_simulation_tcl_command_file(self, name: str = "xrun_mapped_genus.tcl", synthesis_program: str = "genus"):
        mapped_tcl_command_file: FileGenerator.FileGenerator = FileGenerator.FileGenerator()
        mapped_tcl_command_file.set_path(self.path)
        mapped_tcl_command_file.set_name(name)
        mapped_tcl_command_file.open_output_file()
        mapped_commands: list[str] = list[str]()
        mapped_commands.append("database -open mapped_signal_activity_" + synthesis_program + "_vcd -vcd -into mapped_signal_activity_" + synthesis_program + ".vcd")
        lookback_lookup_table_entries_size: int = self.configuration_m_number_of_digital_states * self.configuration_lookback_length * self.configuration_fir_data_width * 2**self.configuration_fir_lut_input_width / self.configuration_fir_lut_input_width
        lookahead_lookup_table_entries_size: int = self.configuration_m_number_of_digital_states * self.configuration_lookahead_length * self.configuration_fir_data_width * 2**self.configuration_fir_lut_input_width / self.configuration_fir_lut_input_width
        #mapped_commands.append("probe -create -vcd -database mapped_signal_activity_" + synthesis_program + "_vcd -packed " + str(lookback_lookup_table_entries_size).removesuffix(".0") + " DigitalEstimatorTestbench.dut_digital_estimator.lookback_lookup_table_entries")
        #mapped_commands.append("probe -create -vcd -database mapped_signal_activity_" + synthesis_program + "_vcd -packed " + str(lookahead_lookup_table_entries_size).removesuffix(".0") + " DigitalEstimatorTestbench.dut_digital_estimator.lookahead_lookup_table_entries")
        #mapped_commands.append("probe -create -vcd -database mapped_signal_activity_" + synthesis_program + "_vcd -packed " + str(lookback_lookup_table_entries_size).removesuffix(".0") + " DigitalEstimatorTestbench.dut_digital_estimator.lookback_lookup_table_block.lookup_table_entries")
        #mapped_commands.append("probe -create -vcd -database mapped_signal_activity_" + synthesis_program + "_vcd -packed " + str(lookahead_lookup_table_entries_size).removesuffix(".0") + " DigitalEstimatorTestbench.dut_digital_estimator.lookahead_lookup_table_block.lookup_table_entries")
        if synthesis_program == "genus":
            if self.configuration_coefficients_variable_fixed == "variable":
                mapped_commands.append("probe -create -vcd -database mapped_signal_activity_" + synthesis_program + "_vcd -packed " + str(lookback_lookup_table_entries_size).removesuffix(".0") + " DigitalEstimatorTestbench.lookback_lookup_table_entries")
                mapped_commands.append("probe -create -vcd -database mapped_signal_activity_" + synthesis_program + "_vcd -packed " + str(lookback_lookup_table_entries_size).removesuffix(".0") + " DigitalEstimatorTestbench.lookahead_lookup_table_entries")
        if synthesis_program == "synopsys":
            mapped_commands.append("probe -create -vcd -database mapped_signal_activity_" + synthesis_program + "_vcd -packed " + str(lookback_lookup_table_entries_size).removesuffix(".0") + " DigitalEstimatorTestbench.dut_digital_estimator.lookback_lookup_table_block.lookup_table_entries")
            mapped_commands.append("probe -create -vcd -database mapped_signal_activity_" + synthesis_program + "_vcd -packed " + str(lookahead_lookup_table_entries_size).removesuffix(".0") + " DigitalEstimatorTestbench.dut_digital_estimator.lookahead_lookup_table_block.lookup_table_entries")
            if self.configuration_coefficients_variable_fixed == "variable":
                mapped_commands.append("probe -create -vcd -database mapped_signal_activity_" + synthesis_program + "_vcd -packed " + str(lookahead_lookup_table_entries_size).removesuffix(".0") + " DigitalEstimatorTestbench.lookback_lookup_table_entries")
                mapped_commands.append("probe -create -vcd -database mapped_signal_activity_" + synthesis_program + "_vcd -packed " + str(lookahead_lookup_table_entries_size).removesuffix(".0") + " DigitalEstimatorTestbench.lookahead_lookup_table_entries")
                mapped_commands.append("probe -create -vcd -database mapped_signal_activity_" + synthesis_program + "_vcd -packed " + str(lookahead_lookup_table_entries_size).removesuffix(".0") + " DigitalEstimatorTestbench.dut_digital_estimator.lookback_lookup_table_entries")
                mapped_commands.append("probe -create -vcd -database mapped_signal_activity_" + synthesis_program + "_vcd -packed " + str(lookahead_lookup_table_entries_size).removesuffix(".0") + " DigitalEstimatorTestbench.dut_digital_estimator.lookahead_lookup_table_entries")
                mapped_commands.append("probe -create -vcd -database mapped_signal_activity_" + synthesis_program + "_vcd -packed " + str(lookahead_lookup_table_entries_size).removesuffix(".0") + " DigitalEstimatorTestbench.dut_digital_estimator.lookup_table_coefficient_register.lookback_coefficients")
                mapped_commands.append("probe -create -vcd -database mapped_signal_activity_" + synthesis_program + "_vcd -packed " + str(lookahead_lookup_table_entries_size).removesuffix(".0") + " DigitalEstimatorTestbench.dut_digital_estimator.lookup_table_coefficient_register.lookahead_coefficients")
                mapped_commands.append("probe -create -vcd -database mapped_signal_activity_" + synthesis_program + "_vcd -packed " + str(lookahead_lookup_table_entries_size).removesuffix(".0") + " DigitalEstimatorTestbench.dut_digital_estimator.lookback_lookup_table_block.lookup_table_entries")
                mapped_commands.append("probe -create -vcd -database mapped_signal_activity_" + synthesis_program + "_vcd -packed " + str(lookahead_lookup_table_entries_size).removesuffix(".0") + " DigitalEstimatorTestbench.dut_digital_estimator.lookahead_lookup_table_block.lookup_table_entries")
        mapped_commands.append("probe -create -vcd -database mapped_signal_activity_" + synthesis_program + "_vcd -depth all -all")
        mapped_commands.append("run")
        mapped_commands.append("exit")
        for line in mapped_commands:
            mapped_tcl_command_file.write_line_linebreak(line)
        mapped_tcl_command_file.close_output_file()
        
    def generate_vcs_mapped_simulation_file(self, name: str = "sim_vcs_mapped_genus.sh", synthesis_program: str = "genus"):
        """Generates the script for the RTL simulation with Synopsys VCS.
        """
        simulation_file: FileGenerator.FileGenerator = FileGenerator.FileGenerator()
        simulation_file.set_path(self.path)
        simulation_file.set_name(name)
        simulation_file.open_output_file()

        settings: list[str] = list[str]()
        settings.append("vcs -f vcs_mapped_options_" + synthesis_program)
        settings.append("./simv")
        settings.append("rm simv")
        settings.append("rm -R ./csrc")
        settings.append("rm -R simv.daidir")
        #settings.append("rm -R waves.shm")
        for line in settings:
            simulation_file.write_line_linebreak(line)
        simulation_file.close_output_file()
        Path(self.path + "/" + name).chmod(S_IRWXU)

    def generate_vcs_mapped_options_file(self, name: str = "vcs_mapped_options_genus", synthesis_program: str = "genus"):
        """Generates the options file used for the RTL simulation with Synopsys VCS.
        """
        options_file: FileGenerator.FileGenerator = FileGenerator.FileGenerator()
        options_file.set_path(self.path)
        options_file.set_name(name)
        options_file.open_output_file()

        options: list[str] = list[str]()
        options.append("-sverilog")
        options.append("-timescale=1ns/1ps")
        options.append("/eda/kits/stm/28nm_fdsoi_v1.3a/C28SOI_SC_12_CORE_LR/5.1-05.81/behaviour/verilog/C28SOI_SC_12_CORE_LR.v")
        options.append("/eda/kits/stm/28nm_fdsoi_v1.3a/C28SOI_SC_12_CLK_LR/5.1-06.81/behaviour/verilog/C28SOI_SC_12_CLK_LR.v")
        options.append("/eda/kits/stm/28nm_fdsoi_v1.3a/C28SOI_SC_12_PR_LR/5.3.a-00.80/behaviour/verilog/C28SOI_SC_12_PR_LR.v")
        options.append("DigitalEstimatorTestbench_mapped_" + synthesis_program + "_vcs.sv")
        options.append(self.top_module_name + ".mapped." + synthesis_program + ".v") 
        for line in options:
            options_file.write_line_linebreak(line)
        options_file.close_output_file()

    def simulate_vcs_mapped(self, synthesis_program: str = "genus"):
        if synthesis_program == "genus":
            self.copy_mapped_design_genus()
        elif synthesis_program == "synopsys":
            self.copy_mapped_design_synopsys()
        self.generate_testbench_mapped_design(simulation_program = "vcs", synthesis_program = synthesis_program)
        self.generate_vcs_mapped_simulation_file(name = "sim_vcs_mapped_" + synthesis_program + ".sh", synthesis_program = synthesis_program)
        self.generate_vcs_mapped_options_file(name = "vcs_mapped_options_" + synthesis_program, synthesis_program = synthesis_program)
        sim_vcs_mapped = subprocess.Popen(["./sim_vcs_mapped_" + synthesis_program + ".sh"], cwd = self.path, text = True, shell = True)
        sim_vcs_mapped.wait()
        #os.rename(self.path + "/digital_estimation_mapped_" + synthesis_program + ".csv", self.path + "/digital_estimation_mapped_" + synthesis_program + ".vcs.csv")

    def copy_mapped_design_genus(self):
        #shutil.copyfile("../df/out/" + self.top_module_name + "/syn/" + self.top_module_name + ".v", self.path + "/" + self.top_module_name + ".mapped.genus.v")
        shutil.copyfile(self.path_synthesis + "/synthesis_output_genus/" + self.top_module_name + "/" + self.top_module_name + ".v", self.path + "/" + self.top_module_name + ".mapped.genus.v")

    def copy_mapped_design_synopsys(self):
        shutil.copyfile(self.path_synthesis + "/out/" + self.top_module_name + "/results/" + self.top_module_name + ".mapped.v", self.path + "/" + self.top_module_name + ".mapped.synopsys.v")

    def generate_testbench_mapped_design(self, simulation_program: str = "xrun", synthesis_program: str = "genus"):
        digital_estimator_testbench: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorVerificationModules.DigitalEstimatorTestbench.DigitalEstimatorTestbench(self.path, name = "DigitalEstimatorTestbench_mapped_" + synthesis_program + "_" + simulation_program)
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
        digital_estimator_testbench.configuration_coefficients_variable_fixed = self.configuration_coefficients_variable_fixed
        digital_estimator_testbench.configuration_mapped_simulation = True
        digital_estimator_testbench.configuration_placedandrouted_simulation = False
        digital_estimator_testbench.configuration_simulation_program = simulation_program
        digital_estimator_testbench.configuration_synthesis_program = synthesis_program
        digital_estimator_testbench.top_module_name = self.top_module_name
        digital_estimator_testbench.configuration_reduce_size_coefficients = self.configuration_reduce_size_coefficients
        digital_estimator_testbench.configuration_reduce_size_luts = self.configuration_reduce_size_luts
        digital_estimator_testbench.configuration_reduce_size_adders = self.configuration_reduce_size_adders
        digital_estimator_testbench.generate()

    def simulate_mapped_design(self, synthesis_program: str = "genus"):
        self.generate_xrun_mapped_simulation_script(name = "sim_mapped_" + synthesis_program + ".sh", synthesis_program = synthesis_program)
        self.generate_xrun_mapped_simulation_options_file(name = "xrun_options_mapped_" + synthesis_program, synthesis_program = synthesis_program)
        self.generate_xrun_mapped_simulation_tcl_command_file(name = "xrun_mapped_" + synthesis_program + ".tcl", synthesis_program = synthesis_program)
        if synthesis_program == "genus":
            self.copy_mapped_design_genus()
        elif synthesis_program == "synopsys":
            self.copy_mapped_design_synopsys()
        self.generate_testbench_mapped_design(synthesis_program = synthesis_program)
        simulation_mapped = subprocess.Popen(["./sim_mapped_" + synthesis_program + ".sh"], cwd = self.path, text = True, shell = True)
        simulation_mapped.wait()

    def copy_annotation_files_genus(self):
        #shutil.copyfile("../df/out/" + self.top_module_name + "/syn/" + self.top_module_name + ".sdc.gz", self.path + "/" + self.top_module_name + ".genus.sdc.gz")
        shutil.copyfile(self.path_synthesis + "/synthesis_output_genus/" + self.top_module_name + "/" + self.top_module_name + ".sdc.gz", self.path + "/" + self.top_module_name + ".genus.sdc.gz")
        with gzip.open(self.path + "/" + self.top_module_name + ".genus.sdc.gz", "rb") as compressed_genus_sdc_file:
            with open(self.path + "/" + self.top_module_name + ".genus.sdc", "wb") as genus_sdc_file:
                shutil.copyfileobj(compressed_genus_sdc_file, genus_sdc_file)
                os.remove(self.path + "/" + self.top_module_name + ".genus.sdc.gz")
        #shutil.copyfile("../df/out/" + self.top_module_name + "/syn/" + self.top_module_name + ".sdf.gz", self.path + "/" + self.top_module_name + ".genus.sdf.gz")
        shutil.copyfile(self.path_synthesis + "/synthesis_output_genus/" + self.top_module_name + "/" + self.top_module_name + ".sdf.gz", self.path + "/" + self.top_module_name + ".genus.sdf.gz")
        with gzip.open(self.path + "/" + self.top_module_name + ".genus.sdf.gz", "rb") as compressed_genus_sdf_file:
            with open(self.path + "/" + self.top_module_name + ".genus.sdf", "wb") as genus_sdf_file:
                shutil.copyfileobj(compressed_genus_sdf_file, genus_sdf_file)
                os.remove(self.path + "/" + self.top_module_name + ".genus.sdf.gz")
        #shutil.copyfile("../df/out/" + self.top_module_name + "/syn/" + self.top_module_name + ".spef", self.path + "/" + self.top_module_name + ".genus.spef")
        shutil.copyfile(self.path_synthesis + "/synthesis_output_genus/" + self.top_module_name + "/" + self.top_module_name + ".spef", self.path + "/" + self.top_module_name + ".genus.spef")

    def copy_annotation_files_synopsys(self):
        shutil.copyfile(self.path_synthesis + "/out/" + self.top_module_name + "/results/" + self.top_module_name + ".mapped.sdc", self.path + "/" + self.top_module_name + ".synopsys.sdc")
        shutil.copyfile(self.path_synthesis + "/out/" + self.top_module_name + "/results/" + self.top_module_name + ".mapped.sdf", self.path + "/" + self.top_module_name + ".synopsys.sdf")
        shutil.copyfile(self.path_synthesis + "/out/" + self.top_module_name + "/results/" + self.top_module_name + ".mapped.spef", self.path + "/" + self.top_module_name + ".synopsys.spef")

    def copy_primetime_power_tcl_script(self):
        shutil.copyfile(self.scripts_base_folder + "/primetime_power_template.tcl", self.path + "/primetime_power_estimation.tcl")

    def generate_primetime_power_script(self, synthesis_program: str = "genus"):
        commands: list[str] = list[str]()
        commands.append("SYNTHESIS_PROGRAM=" + synthesis_program + "\n")
        commands.append("MAPPED_DESIGN_FILE=./" + self.top_module_name + ".mapped." + synthesis_program + ".v\n")
        commands.append("MAPPED_DESIGN_NAME=" + self.top_module_name + "\n")
        commands.append("SDF_FILE=./" + self.top_module_name + "." + synthesis_program + ".sdf\n")
        commands.append("SDC_FILE=./" + self.top_module_name + "." + synthesis_program + ".sdc\n")
        commands.append("SPEF_FILE=./" + self.top_module_name + "." + synthesis_program + ".spef\n")
        commands.append("VCD_FILE=./mapped_signal_activity_" + synthesis_program + ".vcd\n")
        commands.append("export SYNTHESIS_PROGRAM\n")
        commands.append("export MAPPED_DESIGN_FILE\n")
        commands.append("export MAPPED_DESIGN_NAME\n")
        commands.append("export SDF_FILE\n")
        commands.append("export SDC_FILE\n")
        commands.append("export SPEF_FILE\n")
        commands.append("export VCD_FILE\n")
        commands.append("primetime -file primetime_power_estimation.tcl\n")
        commands.append("SYNTHESIS_PROGRAM=\n")
        commands.append("MAPPED_DESIGN_FILE=\n")
        commands.append("MAPPED_DESIGN_NAME=\n")
        commands.append("SDF_FILE=\n")
        commands.append("SDC_FILE=\n")
        commands.append("SPEF_FILE=\n")
        commands.append("VCD_FILE=\n")
        with open(self.path + "/run_primetime_power_estimation_" + synthesis_program + ".sh", "w") as primetime_power_script:
            primetime_power_script.writelines(commands)
            Path(self.path + "/run_primetime_power_estimation_" + synthesis_program + ".sh").chmod(S_IRWXU)

    def estimate_power_primetime(self, synthesis_program: str = "genus"):
        if synthesis_program == "genus":
            self.copy_annotation_files_genus()
        elif synthesis_program == "synopsys":
            self.copy_annotation_files_synopsys()
        self.copy_primetime_power_tcl_script()
        self.generate_primetime_power_script(synthesis_program = synthesis_program)
        power_estimation_primetime = subprocess.Popen(["./run_primetime_power_estimation_" + synthesis_program + ".sh"], cwd = self.path, text = True, shell = True)
        power_estimation_primetime.wait()

    def generate_placeandroute_scripts_innovus(self, synthesis_program: str = "genus"):
        script_lines: list[str] = list[str]()
        if synthesis_program == "genus":
            with open(self.scripts_base_folder + "/pnr_innovus_synthesis_from_genus_template", mode = "r") as placeandroute_script_old:
                script_lines = placeandroute_script_old.readlines()
                for index in range(len(script_lines)):
                    if script_lines[index].find("DESIGN_TOP_MODULE=") != -1:
                        script_lines.pop(index)
                        script_lines.insert(index, "DESIGN_TOP_MODULE=\"" + self.top_module_name + "\"\n")
                    elif script_lines[index].find("DESIGN_TOP_MODULE=") != -1:
                        pass
            with open(self.path_synthesis + "/pnr_innovus_synthesis_from_genus", mode = "w") as placeandroute_script:
                placeandroute_script.writelines(script_lines)
            Path(self.path_synthesis + "/pnr_innovus_synthesis_from_genus").chmod(S_IRWXU)
                    
            shutil.copyfile(self.scripts_base_folder + "/pnr_innovus_synthesis_from_genus_template.tcl", self.path_synthesis + "/pnr_innovus_synthesis_from_genus.tcl")
            shutil.copyfile(self.scripts_base_folder + "/init_innovus_synthesis_from_genus_template.tcl", self.path_synthesis + "/init_innovus_synthesis_from_genus.tcl")
        elif synthesis_program == "synopsys":
            with open(self.scripts_base_folder + "/pnr_innovus_synthesis_from_synopsys_template", mode = "r") as placeandroute_script_old:
                script_lines = placeandroute_script_old.readlines()
                for index in range(len(script_lines)):
                    if script_lines[index].find("DESIGN_TOP_MODULE=") != -1:
                        script_lines.pop(index)
                        script_lines.insert(index, "DESIGN_TOP_MODULE=\"" + self.top_module_name + "\"\n")
                    elif script_lines[index].find("DESIGN_TOP_MODULE=") != -1:
                        pass
            with open(self.path_synthesis + "/pnr_innovus_synthesis_from_synopsys", mode = "w") as placeandroute_script:
                placeandroute_script.writelines(script_lines)
            Path(self.path_synthesis + "/pnr_innovus_synthesis_from_synopsys").chmod(S_IRWXU)
                    
            shutil.copyfile(self.scripts_base_folder + "/pnr_innovus_synthesis_from_synopsys_template.tcl", self.path_synthesis + "/pnr_innovus_synthesis_from_synopsys.tcl")
            shutil.copyfile(self.scripts_base_folder + "/init_innovus_synthesis_from_synopsys_template.tcl", self.path_synthesis + "/init_innovus_synthesis_from_synopsys.tcl")
        
        shutil.copyfile(self.scripts_base_folder + "/config_pnr_innovus_template.tcl", self.path_synthesis + "/config_pnr_innovus.tcl")
        shutil.copyfile(self.scripts_base_folder + "/checks_innovus_template.tcl", self.path_synthesis + "/checks_innovus.tcl")
        shutil.copyfile(self.scripts_base_folder + "/clock_innovus_template.tcl", self.path_synthesis + "/clock_innovus.tcl")
        shutil.copyfile(self.scripts_base_folder + "/export_innovus_template.tcl", self.path_synthesis + "/export_innovus.tcl")
        shutil.copyfile(self.scripts_base_folder + "/fillers_innovus_template.tcl", self.path_synthesis + "/fillers_innovus.tcl")
        shutil.copyfile(self.scripts_base_folder + "/floorplan_innovus_template.tcl", self.path_synthesis + "/floorplan_innovus.tcl")
        shutil.copyfile(self.scripts_base_folder + "/mmmc_innovus_template.tcl", self.path_synthesis + "/mmmc_innovus.tcl")
        shutil.copyfile(self.scripts_base_folder + "/place_innovus_template.tcl", self.path_synthesis + "/place_innovus.tcl")
        shutil.copyfile(self.scripts_base_folder + "/power_innovus_template.tcl", self.path_synthesis + "/power_innovus.tcl")
        shutil.copyfile(self.scripts_base_folder + "/route_innovus_template.tcl", self.path_synthesis + "/route_innovus.tcl")
        
    def placeandroute_innovus(self, synthesis_program: str = "genus"):
        self.generate_placeandroute_scripts_innovus(synthesis_program = synthesis_program)
        placeandroute_innovus = subprocess.Popen(["./pnr_innovus_synthesis_from_" + synthesis_program], cwd = self.path_synthesis, text = True, shell = True)
        placeandroute_innovus.wait()
            
    def generate_testbench_placedandrouted_design(self, synthesis_program: str = "genus"):
        digital_estimator_testbench: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorVerificationModules.DigitalEstimatorTestbench.DigitalEstimatorTestbench(self.path, name = "DigitalEstimatorTestbench_placedandrouted_innovus_" + synthesis_program)
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
        digital_estimator_testbench.configuration_coefficients_variable_fixed = self.configuration_coefficients_variable_fixed
        digital_estimator_testbench.configuration_mapped_simulation = False
        digital_estimator_testbench.configuration_placedandrouted_simulation = True
        digital_estimator_testbench.configuration_simulation_program = "xrun"
        digital_estimator_testbench.configuration_synthesis_program = synthesis_program
        digital_estimator_testbench.top_module_name = self.top_module_name
        digital_estimator_testbench.configuration_reduce_size_coefficients = self.configuration_reduce_size_coefficients
        digital_estimator_testbench.configuration_reduce_size_luts = self.configuration_reduce_size_luts
        digital_estimator_testbench.configuration_reduce_size_adders = self.configuration_reduce_size_adders
        digital_estimator_testbench.generate()
        
    def copy_placedandrouted_design_files_innovus(self, synthesis_program: str = "genus"):
        shutil.copyfile(self.path_synthesis + "/placeandroute_output_innovus/synthesis_from_" + synthesis_program + "/" + self.top_module_name + "/" + self.top_module_name + "_elc.v", self.path + "/" + self.top_module_name + "_elc_" + synthesis_program + ".v")
        shutil.copyfile(self.path_synthesis + "/placeandroute_output_innovus/synthesis_from_" + synthesis_program + "/" + self.top_module_name + "/" + self.top_module_name + "_physical.v", self.path + "/" + self.top_module_name + "_physical_" + synthesis_program + ".v")
        shutil.copyfile(self.path_synthesis + "/placeandroute_output_innovus/synthesis_from_" + synthesis_program + "/" + self.top_module_name + "/" + self.top_module_name + "_final.sdf", self.path + "/" + self.top_module_name + "_final_" + synthesis_program + ".sdf")
        
    def generate_xrun_placeandroute_simulation_script(self, name: str = "sim_placeandroute_innovus_genus.sh", synthesis_program: str = "genus"):
        placeandroute_simulation_script: FileGenerator.FileGenerator = FileGenerator.FileGenerator()
        placeandroute_simulation_script.set_path(self.path)
        placeandroute_simulation_script.set_name(name)
        placeandroute_simulation_script.open_output_file()
        placeandroute_settings: list[str] = list[str]()
        placeandroute_settings.append("source ~/pro/acmos2/virtuoso/setup_user")
        placeandroute_settings.append("source ~/pro/fall2022/bash/setup_user")
        placeandroute_settings.append("")
        placeandroute_settings.append("xrun -f xrun_options_placeandroute_innovus_" + synthesis_program)
        placeandroute_settings.append("rm -R xcelium.d/")
        for line in placeandroute_settings:
            placeandroute_simulation_script.write_line_linebreak(line)
        placeandroute_simulation_script.close_output_file()
        Path(self.path + "/" + name).chmod(S_IRWXU)

    def generate_xrun_placeandroute_simulation_options_file(self, name: str = "xrun_options_placeandroute_innovus_genus", synthesis_program: str = "genus"):
        placeandroute_options_file: FileGenerator.FileGenerator = FileGenerator.FileGenerator()
        placeandroute_options_file.set_path(self.path)
        placeandroute_options_file.set_name(name)
        placeandroute_options_file.open_output_file()
        placeandroute_options: list[str] = list[str]()
        placeandroute_options.append("-64bit")
        placeandroute_options.append("-access +rwc")
        placeandroute_options.append("-v /eda/kits/stm/28nm_fdsoi_v1.3a/C28SOI_SC_12_CORE_LR/5.1-05.81/behaviour/verilog/C28SOI_SC_12_CORE_LR.v")
        placeandroute_options.append("-v /eda/kits/stm/28nm_fdsoi_v1.3a/C28SOI_SC_12_CLK_LR/5.1-06.81/behaviour/verilog/C28SOI_SC_12_CLK_LR.v")
        placeandroute_options.append("-v /eda/kits/stm/28nm_fdsoi_v1.3a/C28SOI_SC_12_PR_LR/5.3.a-00.80/behaviour/verilog/C28SOI_SC_12_PR_LR.v")
        placeandroute_options.append("-timescale 1ns/1ps")
        placeandroute_options.append("-top DigitalEstimatorTestbench")
        placeandroute_options.append("-input xrun_placeandroute_innovus_" + synthesis_program + ".tcl")
        placeandroute_options.append("DigitalEstimatorTestbench_placedandrouted_innovus_" + synthesis_program + ".sv")
        placeandroute_options.append(self.top_module_name + "_elc_" + synthesis_program + ".v")
        for line in placeandroute_options:
            placeandroute_options_file.write_line_linebreak(line)
        placeandroute_options_file.close_output_file()

    def generate_xrun_placeandroute_simulation_tcl_command_file(self, name: str = "xrun_placeandroute_innovus_genus.tcl", synthesis_program: str = "genus"):
        placeandroute_tcl_command_file: FileGenerator.FileGenerator = FileGenerator.FileGenerator()
        placeandroute_tcl_command_file.set_path(self.path)
        placeandroute_tcl_command_file.set_name(name)
        placeandroute_tcl_command_file.open_output_file()
        placeandroute_commands: list[str] = list[str]()
        placeandroute_commands.append("database -open placeandroute_signal_activity_innovus_" + synthesis_program + "_vcd -vcd -into placeandroute_signal_activity_innovus_" + synthesis_program + ".vcd")
        lookback_lookup_table_entries_size: int = self.configuration_m_number_of_digital_states * self.configuration_lookback_length * self.configuration_fir_data_width * 2**self.configuration_fir_lut_input_width / self.configuration_fir_lut_input_width
        lookahead_lookup_table_entries_size: int = self.configuration_m_number_of_digital_states * self.configuration_lookahead_length * self.configuration_fir_data_width * 2**self.configuration_fir_lut_input_width / self.configuration_fir_lut_input_width
        #placeandroute_commands.append("probe -create -vcd -database placeandroute_signal_activity_innovus_" + synthesis_program + "_vcd -packed " + str(lookback_lookup_table_entries_size).removesuffix(".0") + " DigitalEstimatorTestbench.dut_digital_estimator.lookback_lookup_table_entries")
        #placeandroute_commands.append("probe -create -vcd -database placeandroute_signal_activity_innovus_" + synthesis_program + "_vcd -packed " + str(lookahead_lookup_table_entries_size).removesuffix(".0") + " DigitalEstimatorTestbench.dut_digital_estimator.lookahead_lookup_table_entries")
        #placeandroute_commands.append("probe -create -vcd -database placeandroute_signal_activity_innovus_" + synthesis_program + "_vcd -packed " + str(lookback_lookup_table_entries_size).removesuffix(".0") + " DigitalEstimatorTestbench.dut_digital_estimator.lookback_lookup_table_block.lookup_table_entries")
        #placeandroute_commands.append("probe -create -vcd -database placeandroute_signal_activity_innovus_" + synthesis_program + "_vcd -packed " + str(lookahead_lookup_table_entries_size).removesuffix(".0") + " DigitalEstimatorTestbench.dut_digital_estimator.lookahead_lookup_table_block.lookup_table_entries")
        if synthesis_program == "genus":
            if self.configuration_coefficients_variable_fixed == "variable":
                placeandroute_commands.append("probe -create -vcd -database placeandroute_signal_activity_innovus_" + synthesis_program + "_vcd -packed " + str(lookback_lookup_table_entries_size).removesuffix(".0") + " DigitalEstimatorTestbench.lookback_lookup_table_entries")
                placeandroute_commands.append("probe -create -vcd -database placeandroute_signal_activity_innovus_" + synthesis_program + "_vcd -packed " + str(lookback_lookup_table_entries_size).removesuffix(".0") + " DigitalEstimatorTestbench.lookahead_lookup_table_entries")
        if synthesis_program == "synopsys":
            placeandroute_commands.append("probe -create -vcd -database placeandroute_signal_activity_innovus_" + synthesis_program + "_vcd -packed " + str(lookback_lookup_table_entries_size).removesuffix(".0") + " DigitalEstimatorTestbench.dut_digital_estimator.lookback_lookup_table_block.lookup_table_entries")
            placeandroute_commands.append("probe -create -vcd -database placeandroute_signal_activity_innovus_" + synthesis_program + "_vcd -packed " + str(lookahead_lookup_table_entries_size).removesuffix(".0") + " DigitalEstimatorTestbench.dut_digital_estimator.lookahead_lookup_table_block.lookup_table_entries")
            if self.configuration_coefficients_variable_fixed == "variable":
                placeandroute_commands.append("probe -create -vcd -database placeandroute_signal_activity_innovus_" + synthesis_program + "_vcd -packed " + str(lookahead_lookup_table_entries_size).removesuffix(".0") + " DigitalEstimatorTestbench.lookback_lookup_table_entries")
                placeandroute_commands.append("probe -create -vcd -database placeandroute_signal_activity_innovus_" + synthesis_program + "_vcd -packed " + str(lookahead_lookup_table_entries_size).removesuffix(".0") + " DigitalEstimatorTestbench.lookahead_lookup_table_entries")
                placeandroute_commands.append("probe -create -vcd -database placeandroute_signal_activity_innovus_" + synthesis_program + "_vcd -packed " + str(lookahead_lookup_table_entries_size).removesuffix(".0") + " DigitalEstimatorTestbench.dut_digital_estimator.lookback_lookup_table_entries")
                placeandroute_commands.append("probe -create -vcd -database placeandroute_signal_activity_innovus_" + synthesis_program + "_vcd -packed " + str(lookahead_lookup_table_entries_size).removesuffix(".0") + " DigitalEstimatorTestbench.dut_digital_estimator.lookahead_lookup_table_entries")
                placeandroute_commands.append("probe -create -vcd -database placeandroute_signal_activity_innovus_" + synthesis_program + "_vcd -packed " + str(lookahead_lookup_table_entries_size).removesuffix(".0") + " DigitalEstimatorTestbench.dut_digital_estimator.lookup_table_coefficient_register.lookback_coefficients")
                placeandroute_commands.append("probe -create -vcd -database placeandroute_signal_activity_innovus_" + synthesis_program + "_vcd -packed " + str(lookahead_lookup_table_entries_size).removesuffix(".0") + " DigitalEstimatorTestbench.dut_digital_estimator.lookup_table_coefficient_register.lookahead_coefficients")
                placeandroute_commands.append("probe -create -vcd -database placeandroute_signal_activity_innovus_" + synthesis_program + "_vcd -packed " + str(lookahead_lookup_table_entries_size).removesuffix(".0") + " DigitalEstimatorTestbench.dut_digital_estimator.lookback_lookup_table_block.lookup_table_entries")
                placeandroute_commands.append("probe -create -vcd -database placeandroute_signal_activity_innovus_" + synthesis_program + "_vcd -packed " + str(lookahead_lookup_table_entries_size).removesuffix(".0") + " DigitalEstimatorTestbench.dut_digital_estimator.lookahead_lookup_table_block.lookup_table_entries")
        placeandroute_commands.append("probe -create -vcd -database placeandroute_signal_activity_innovus_" + synthesis_program + "_vcd -depth all -all")
        placeandroute_commands.append("run")
        placeandroute_commands.append("exit")
        for line in placeandroute_commands:
            placeandroute_tcl_command_file.write_line_linebreak(line)
        placeandroute_tcl_command_file.close_output_file()
        
    def simulate_placedandrouted_design(self, synthesis_program: str = "genus"):
        self.generate_testbench_placedandrouted_design(synthesis_program = synthesis_program)
        self.copy_placedandrouted_design_files_innovus(synthesis_program = synthesis_program)
        self.generate_xrun_placeandroute_simulation_script(name = "sim_placeandroute_innovus_" + synthesis_program + ".sh", synthesis_program = synthesis_program)
        self.generate_xrun_placeandroute_simulation_options_file(name = "xrun_options_placeandroute_innovus_" + synthesis_program, synthesis_program = synthesis_program)
        self.generate_xrun_placeandroute_simulation_tcl_command_file(name = "xrun_placeandroute_innovus_" + synthesis_program + ".tcl", synthesis_program = synthesis_program)
        simulation_placedandrouted = subprocess.Popen(["./sim_placeandroute_innovus_" + synthesis_program + ".sh"], cwd = self.path, text = True, shell = True)
        simulation_placedandrouted.wait()
        
    def copy_primetime_power_placeandroute_tcl_script(self):
        shutil.copyfile(self.scripts_base_folder + "/primetime_power_template_placeandroute.tcl", self.path + "/primetime_power_estimation_placeandroute.tcl")

    def generate_primetime_power_placeandroute_script(self, synthesis_program: str = "genus"):
        commands: list[str] = list[str]()
        commands.append("SYNTHESIS_PROGRAM=" + synthesis_program + "\n")
        commands.append("PLACEDANDROUTED_DESIGN_FILE=./" + self.top_module_name + "_physical_" + synthesis_program + ".v\n")
        commands.append("PLACEDANDROUTED_DESIGN_NAME=" + self.top_module_name + "\n")
        commands.append("SDF_FILE=./" + self.top_module_name + "_final_" + synthesis_program + ".sdf\n")
        commands.append("VCD_FILE=./placeandroute_signal_activity_innovus_" + synthesis_program + ".vcd\n")
        commands.append("export SYNTHESIS_PROGRAM\n")
        commands.append("export PLACEDANDROUTED_DESIGN_FILE\n")
        commands.append("export PLACEDANDROUTED_DESIGN_NAME\n")
        commands.append("export SDF_FILE\n")
        commands.append("export VCD_FILE\n")
        commands.append("primetime -file primetime_power_estimation_placeandroute.tcl\n")
        commands.append("SYNTHESIS_PROGRAM=\n")
        commands.append("PLACEDANDROUTED_DESIGN_FILE=\n")
        commands.append("PLACEDANDROUTED_DESIGN_NAME=\n")
        commands.append("SDF_FILE=\n")
        commands.append("VCD_FILE=\n")
        with open(self.path + "/run_primetime_power_estimation_placeandroute_" + synthesis_program + ".sh", "w") as primetime_power_script:
            primetime_power_script.writelines(commands)
            Path(self.path + "/run_primetime_power_estimation_placeandroute_" + synthesis_program + ".sh").chmod(S_IRWXU)

    def estimate_power_primetime_placeandroute(self, synthesis_program: str = "genus"):
        self.copy_primetime_power_placeandroute_tcl_script()
        self.generate_primetime_power_placeandroute_script(synthesis_program = synthesis_program)
        power_estimation_primetime_placeandroute = subprocess.Popen(["./run_primetime_power_estimation_placeandroute_" + synthesis_program + ".sh"], cwd = self.path, text = True, shell = True)
        power_estimation_primetime_placeandroute.wait()
        

if __name__ == '__main__':
    """Main function for testing the implementation.

    This is mainly used for debugging purposes.
    Generating digital estimation filters should normally be done with the pytest framework.
    """
    digital_estimator_generator: DigitalEstimatorGenerator = DigitalEstimatorGenerator()
    digital_estimator_generator.generate()
    #simulation_result: tuple[int, str] = (0, "Skip simulation.")
    simulation_result: tuple[int, str] = digital_estimator_generator.simulate()
    simulation_result: tuple[int, str] = (0, "Ignore fails in simulation.")
    #digital_estimator_generator.simulate_vcs()
    if simulation_result[0] == 0:
        pass
        #digital_estimator_generator.synthesize_genus()
        #digital_estimator_generator.synthesize_synopsys()
        #digital_estimator_generator.simulate_mapped_design(synthesis_program = "genus")
        #digital_estimator_generator.simulate_mapped_design(synthesis_program = "synopsys")
        #digital_estimator_generator.simulate_vcs_mapped(synthesis_program = "genus")
        #digital_estimator_generator.simulate_vcs_mapped(synthesis_program = "synopsys")
        #digital_estimator_generator.estimate_power_primetime(synthesis_program = "genus")
        #digital_estimator_generator.estimate_power_primetime(synthesis_program = "synopsys")
        #digital_estimator_generator.high_level_simulation.plot_results_mapped(file_name = "digital_estimation_mapped_genus.csv", synthesis_program = "genus")
        digital_estimator_generator.high_level_simulation.plot_results_mapped(file_name = "digital_estimation_mapped_synopsys.csv", synthesis_program = "synopsys")
        #digital_estimator_generator.placeandroute_innovus(synthesis_program = "genus")
        #digital_estimator_generator.placeandroute_innovus(synthesis_program = "synopsys")
        
        #digital_estimator_generator.simulate_placedandrouted_design("genus")
        #digital_estimator_generator.estimate_power_primetime_placeandroute("genus")
        #digital_estimator_generator.simulate_placedandrouted_design("synopsys")
        #digital_estimator_generator.estimate_power_primetime_placeandroute("synopsys")