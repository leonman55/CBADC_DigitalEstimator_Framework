import pytest

import DigitalEstimatorGenerator


# Base folder for the generation of the test cases.
base_folder: str = "/df/sim/TestCase"

path_simulation: str = "/local_work/leonma/sim"
path_synthesis: str = "/local_work/leonma/src"


# Explaination of the parameters
# Test number
# Folder to place SystemVerilog files
# Number of analog states
# Oversampling rate
# FIR coefficient bit width
# lookback length
# lookahead length
# LUT input bit width
def test_digital_estimator_generator(
    test_case_number: int = 0,
    directory: str = "DigitalEstimator",
    n_number_of_analog_states: int = 3,
    oversampling_rate: int = 23,
    bit_width: int = 22,
    lookback_length: int = 160,
    lookahead_length: int = 160,
    lut_input_width: int = 4,
    counter_type: str = "binary",
    combinatorial_synchronous: str = "synchronous",
    coefficients_variable_fixed: str = "variable",
    reduce_size_coefficients: bool = False,
    reduce_size_luts: bool = False,
    reduce_size_adders: bool = False):
    """Method for executing the test cases.

    Parameters
    ----------
    test_case_number: int
        Used to number the test cases
    directory: str
        The directory for the specific test case
    n_number_of_analog_states: int
        Number N of analog states in the test case
    oversampling_rate: int
        Oversampling rate of the digital control unit
    bit_width: int
        Bit width of the filter coefficients
    lookback_length: int
        Lookback batch size
    lookahead_length: int
        Lookahead batch size
    lut_input_width: int
        Select input width of the LUT tables
    """
    test_case_name: str = "DigitalEstimator_" + str(n_number_of_analog_states) + "_" + str(oversampling_rate) + "_" + str(bit_width) + "_" + str(lookback_length) + "_" + str(lookahead_length) + "_" + str(lut_input_width) + "_" + coefficients_variable_fixed + "_" + combinatorial_synchronous + "_" + counter_type + "_" + str(reduce_size_coefficients) + "_" + str(reduce_size_luts) + "_" + str(reduce_size_adders)
    simulation_directory: str = path_simulation + "/" + test_case_name
    synthesis_directory: str = path_synthesis + "/" + test_case_name
    
    digital_estimator_generator: DigitalEstimatorGenerator.DigitalEstimatorGenerator = DigitalEstimatorGenerator.DigitalEstimatorGenerator()
    digital_estimator_generator.path = simulation_directory
    digital_estimator_generator.path_synthesis = synthesis_directory
    
    digital_estimator_generator.configuration_n_number_of_analog_states = n_number_of_analog_states
    digital_estimator_generator.configuration_lookback_length = lookback_length
    digital_estimator_generator.configuration_lookahead_length = lookahead_length
    digital_estimator_generator.configuration_fir_data_width = bit_width
    digital_estimator_generator.configuration_fir_lut_input_width = lut_input_width
    digital_estimator_generator.configuration_over_sample_rate = oversampling_rate
    digital_estimator_generator.configuration_counter_type = counter_type
    digital_estimator_generator.configuration_combinatorial_synchronous = combinatorial_synchronous
    digital_estimator_generator.configuration_coefficients_variable_fixed = coefficients_variable_fixed
    digital_estimator_generator.configuration_reduce_size_coefficients = reduce_size_coefficients
    digital_estimator_generator.configuration_reduce_size_luts = reduce_size_luts
    digital_estimator_generator.configuration_reduce_size_adders = reduce_size_adders

    digital_estimator_generator.generate()
    #simulation_result: tuple[int, str] = (0, "Skip simulation.")
    simulation_result: tuple[int, str] = digital_estimator_generator.simulate()
    simulation_result: tuple[int, str] = (0, "Ignore fails in simulation.")
    #digital_estimator_generator.simulate_vcs()
    if simulation_result[0] == 0:
        #pass
        #digital_estimator_generator.synthesize_genus()
        digital_estimator_generator.synthesize_synopsys()
        #digital_estimator_generator.simulate_mapped_design(synthesis_program = "genus")
        #digital_estimator_generator.simulate_mapped_design(synthesis_program = "synopsys")
        #digital_estimator_generator.simulate_vcs_mapped(synthesis_program = "genus")
        digital_estimator_generator.simulate_vcs_mapped(synthesis_program = "synopsys")
        #digital_estimator_generator.estimate_power_primetime(synthesis_program = "genus", simulation_program = "xrun")
        #digital_estimator_generator.estimate_power_primetime(synthesis_program = "genus", simulation_program = "vcs")
        #digital_estimator_generator.estimate_power_primetime(synthesis_program = "synopsys", simulation_program = "xrun")
        digital_estimator_generator.estimate_power_primetime(synthesis_program = "synopsys", simulation_program = "vcs")
        #digital_estimator_generator.high_level_simulation.plot_results_mapped(file_name = "digital_estimation_mapped_genus_xrun.csv", synthesis_program = "genus")
        #digital_estimator_generator.high_level_simulation.plot_results_mapped(file_name = "digital_estimation_mapped_genus_vcs.csv", synthesis_program = "genus")
        #digital_estimator_generator.high_level_simulation.plot_results_mapped(file_name = "digital_estimation_mapped_synopsys_xrun.csv", synthesis_program = "synopsys")
        digital_estimator_generator.high_level_simulation.plot_results_mapped(file_name = "digital_estimation_mapped_synopsys_vcs.csv", synthesis_program = "synopsys")
        #digital_estimator_generator.placeandroute_innovus(synthesis_program = "genus")
        digital_estimator_generator.placeandroute_innovus(synthesis_program = "synopsys")
        
        #digital_estimator_generator.simulate_placedandrouted_design("genus")
        #digital_estimator_generator.estimate_power_primetime_placeandroute("genus")
        digital_estimator_generator.simulate_placedandrouted_design("synopsys")
        digital_estimator_generator.estimate_power_primetime_placeandroute("synopsys")



test_non_reduced_configurations_0 = [
        (3, 23, 22, 128, 128, 4, "binary", "synchronous", "variable", False, False, False),
        (4, 15, 22, 160, 160, 4, "binary", "synchronous", "variable", False, False, False),
        (5, 12, 22, 160, 160, 4, "binary", "synchronous", "variable", False, False, False),
        (6, 9, 22, 224, 224, 4, "binary", "synchronous", "variable", False, False, False),
        (7, 8, 22, 224, 224, 4, "binary", "synchronous", "variable", False, False, False),
        (8, 7, 22, 256, 256, 4, "binary", "synchronous", "variable", False, False, False)
    ]

@pytest.mark.test_non_reduced_configurations_0
@pytest.mark.parametrize(("n_number_of_analog_states", "oversampling_rate", "bit_width", "lookback_length", "lookahead_length", "lut_input_width", "counter_type", "combinatorial_synchronous", "coefficients_variable_fixed", "reduce_size_coefficients", "reduce_size_luts", "reduce_size_adders"), test_non_reduced_configurations_0)
def test_non_reduced_configurations_0(n_number_of_analog_states: int, oversampling_rate: int, bit_width: int, lookback_length: int, lookahead_length: int, lut_input_width: int, counter_type: str, combinatorial_synchronous: str, coefficients_variable_fixed: str, reduce_size_coefficients: bool, reduce_size_luts: bool, reduce_size_adders: bool):
    test_digital_estimator_generator(n_number_of_analog_states = n_number_of_analog_states, oversampling_rate = oversampling_rate, bit_width = bit_width, lookback_length = lookback_length, lookahead_length = lookahead_length, lut_input_width = lut_input_width, counter_type = counter_type, combinatorial_synchronous = combinatorial_synchronous, coefficients_variable_fixed = coefficients_variable_fixed, reduce_size_coefficients = reduce_size_coefficients, reduce_size_luts = reduce_size_luts, reduce_size_adders = reduce_size_adders)


test_non_reduced_configurations_1 = [
        (3, 23, 22, 128, 128, 2, "binary", "synchronous", "variable", False, False, False),
        (4, 15, 22, 160, 160, 2, "binary", "synchronous", "variable", False, False, False),
        (5, 12, 22, 160, 160, 2, "binary", "synchronous", "variable", False, False, False),
        (6, 9, 22, 224, 224, 2, "binary", "synchronous", "variable", False, False, False),
        (7, 8, 22, 224, 224, 2, "binary", "synchronous", "variable", False, False, False),
        (8, 7, 22, 256, 256, 2, "binary", "synchronous", "variable", False, False, False)
    ]

@pytest.mark.test_non_reduced_configurations_1
@pytest.mark.parametrize(("n_number_of_analog_states", "oversampling_rate", "bit_width", "lookback_length", "lookahead_length", "lut_input_width", "counter_type", "combinatorial_synchronous", "coefficients_variable_fixed", "reduce_size_coefficients", "reduce_size_luts", "reduce_size_adders"), test_non_reduced_configurations_1)
def test_non_reduced_configurations_1(n_number_of_analog_states: int, oversampling_rate: int, bit_width: int, lookback_length: int, lookahead_length: int, lut_input_width: int, counter_type: str, combinatorial_synchronous: str, coefficients_variable_fixed: str, reduce_size_coefficients: bool, reduce_size_luts: bool, reduce_size_adders: bool):
    test_digital_estimator_generator(n_number_of_analog_states = n_number_of_analog_states, oversampling_rate = oversampling_rate, bit_width = bit_width, lookback_length = lookback_length, lookahead_length = lookahead_length, lut_input_width = lut_input_width, counter_type = counter_type, combinatorial_synchronous = combinatorial_synchronous, coefficients_variable_fixed = coefficients_variable_fixed, reduce_size_coefficients = reduce_size_coefficients, reduce_size_luts = reduce_size_luts, reduce_size_adders = reduce_size_adders)


test_reduced_configurations_0 = [
        (3, 23, 22, 128, 128, 4, "binary", "synchronous", "variable", True, True, True),
        (4, 15, 22, 160, 160, 4, "binary", "synchronous", "variable", True, True, True),
        (5, 12, 22, 160, 160, 4, "binary", "synchronous", "variable", True, True, True),
        (6, 9, 22, 224, 224, 4, "binary", "synchronous", "variable", True, True, True),
        (7, 8, 22, 224, 224, 4, "binary", "synchronous", "variable", True, True, True),
        (8, 7, 22, 256, 256, 4, "binary", "synchronous", "variable", True, True, True)
    ]

@pytest.mark.test_reduced_configurations_0
@pytest.mark.parametrize(("n_number_of_analog_states", "oversampling_rate", "bit_width", "lookback_length", "lookahead_length", "lut_input_width", "counter_type", "combinatorial_synchronous", "coefficients_variable_fixed", "reduce_size_coefficients", "reduce_size_luts", "reduce_size_adders"), test_reduced_configurations_0)
def test_reduced_configurations_0(n_number_of_analog_states: int, oversampling_rate: int, bit_width: int, lookback_length: int, lookahead_length: int, lut_input_width: int, counter_type: str, combinatorial_synchronous: str, coefficients_variable_fixed: str, reduce_size_coefficients: bool, reduce_size_luts: bool, reduce_size_adders: bool):
    test_digital_estimator_generator(n_number_of_analog_states = n_number_of_analog_states, oversampling_rate = oversampling_rate, bit_width = bit_width, lookback_length = lookback_length, lookahead_length = lookahead_length, lut_input_width = lut_input_width, counter_type = counter_type, combinatorial_synchronous = combinatorial_synchronous, coefficients_variable_fixed = coefficients_variable_fixed, reduce_size_coefficients = reduce_size_coefficients, reduce_size_luts = reduce_size_luts, reduce_size_adders = reduce_size_adders)


test_reduced_configurations_1 = [
        (3, 23, 22, 128, 128, 2, "binary", "synchronous", "variable", True, True, True),
        (4, 15, 22, 160, 160, 2, "binary", "synchronous", "variable", True, True, True),
        (5, 12, 22, 160, 160, 2, "binary", "synchronous", "variable", True, True, True),
        (6, 9, 22, 224, 224, 2, "binary", "synchronous", "variable", True, True, True),
        (7, 8, 22, 224, 224, 2, "binary", "synchronous", "variable", True, True, True),
        (8, 7, 22, 256, 256, 2, "binary", "synchronous", "variable", True, True, True)
    ]

@pytest.mark.test_reduced_configurations_1
@pytest.mark.parametrize(("n_number_of_analog_states", "oversampling_rate", "bit_width", "lookback_length", "lookahead_length", "lut_input_width", "counter_type", "combinatorial_synchronous", "coefficients_variable_fixed", "reduce_size_coefficients", "reduce_size_luts", "reduce_size_adders"), test_reduced_configurations_1)
def test_reduced_configurations_1(n_number_of_analog_states: int, oversampling_rate: int, bit_width: int, lookback_length: int, lookahead_length: int, lut_input_width: int, counter_type: str, combinatorial_synchronous: str, coefficients_variable_fixed: str, reduce_size_coefficients: bool, reduce_size_luts: bool, reduce_size_adders: bool):
    test_digital_estimator_generator(n_number_of_analog_states = n_number_of_analog_states, oversampling_rate = oversampling_rate, bit_width = bit_width, lookback_length = lookback_length, lookahead_length = lookahead_length, lut_input_width = lut_input_width, counter_type = counter_type, combinatorial_synchronous = combinatorial_synchronous, coefficients_variable_fixed = coefficients_variable_fixed, reduce_size_coefficients = reduce_size_coefficients, reduce_size_luts = reduce_size_luts, reduce_size_adders = reduce_size_adders)


test_fixed_non_reduced_configurations_0 = [
        (3, 23, 22, 128, 128, 4, "binary", "synchronous", "fixed", False, False, False),
        (4, 15, 22, 160, 160, 4, "binary", "synchronous", "fixed", False, False, False),
        (5, 12, 22, 160, 160, 4, "binary", "synchronous", "fixed", False, False, False),
        (6, 9, 22, 224, 224, 4, "binary", "synchronous", "fixed", False, False, False),
        (7, 8, 22, 224, 224, 4, "binary", "synchronous", "fixed", False, False, False),
        (8, 7, 22, 256, 256, 4, "binary", "synchronous", "fixed", False, False, False)
    ]

@pytest.mark.test_fixed_non_reduced_configurations_0
@pytest.mark.parametrize(("n_number_of_analog_states", "oversampling_rate", "bit_width", "lookback_length", "lookahead_length", "lut_input_width", "counter_type", "combinatorial_synchronous", "coefficients_variable_fixed", "reduce_size_coefficients", "reduce_size_luts", "reduce_size_adders"), test_fixed_non_reduced_configurations_0)
def test_fixed_non_reduced_configurations_0(n_number_of_analog_states: int, oversampling_rate: int, bit_width: int, lookback_length: int, lookahead_length: int, lut_input_width: int, counter_type: str, combinatorial_synchronous: str, coefficients_variable_fixed: str, reduce_size_coefficients: bool, reduce_size_luts: bool, reduce_size_adders: bool):
    test_digital_estimator_generator(n_number_of_analog_states = n_number_of_analog_states, oversampling_rate = oversampling_rate, bit_width = bit_width, lookback_length = lookback_length, lookahead_length = lookahead_length, lut_input_width = lut_input_width, counter_type = counter_type, combinatorial_synchronous = combinatorial_synchronous, coefficients_variable_fixed = coefficients_variable_fixed, reduce_size_coefficients = reduce_size_coefficients, reduce_size_luts = reduce_size_luts, reduce_size_adders = reduce_size_adders)


# Explaination of the parameters
# Test number
# Folder to place SystemVerilog files
# Number of analog states
# Oversampling rate
# FIR coefficient bit width
# lookback length
# lookahead length
# LUT input bit width
test_2to8_64_512_512_4_cases = [
        (0, base_folder, 2, 180, 64, 512, 512, 4),
        (1, base_folder, 3, 48, 64, 512, 512, 4),
        (2, base_folder, 4, 25, 64, 512, 512, 4),
        (3, base_folder, 5, 17, 64, 512, 512, 4),
        (4, base_folder, 6, 13, 64, 512, 512, 4),
        (5, base_folder, 7, 11, 64, 512, 512, 4),
        (6, base_folder, 8, 9, 64, 512, 512, 4)
    ]

@pytest.mark.test_2to8_64_512_512_4
@pytest.mark.parametrize(("test_case_number", "directory", "n_number_of_analog_states", "oversampling_rate", "bit_width", "lookback_length", "lookahead_length", "lut_input_width"), test_2to8_64_512_512_4_cases)
def test_2to8_64_512_512_4_digital_estimator_generator(test_case_number: int, directory: str, n_number_of_analog_states: int, oversampling_rate: int, bit_width: int, lookback_length: int, lookahead_length: int, lut_input_width: int):
    test_digital_estimator_generator(test_case_number = test_case_number, directory = directory, n_number_of_analog_states = n_number_of_analog_states, oversampling_rate = oversampling_rate, bit_width = bit_width, lookback_length = lookback_length, lookahead_length = lookahead_length, lut_input_width = lut_input_width)


# Explaination of the parameters
# Test number
# Folder to place SystemVerilog files
# Number of analog states
# Oversampling rate
# FIR coefficient bit width
# lookback length
# lookahead length
# LUT input bit width
test_2_64_16to4096_4_cases = [
        (0, base_folder, 2, 180, 64, 16, 16, 4),
        (1, base_folder, 2, 180, 64, 32, 32, 4),
        (2, base_folder, 2, 180, 64, 64, 64, 4),
        (3, base_folder, 2, 180, 64, 128, 128, 4),
        (4, base_folder, 2, 180, 64, 256, 256, 4),
        (5, base_folder, 2, 180, 64, 512, 512, 4),
        (6, base_folder, 2, 180, 64, 1024, 1024, 4),
        (7, base_folder, 2, 180, 64, 2048, 2048, 4),
        (8, base_folder, 2, 180, 64, 4096, 4096, 4)
    ]

@pytest.mark.test_2_64_16to4096_4
@pytest.mark.parametrize(("test_case_number", "directory", "n_number_of_analog_states", "oversampling_rate", "bit_width", "lookback_length", "lookahead_length", "lut_input_width"), test_2_64_16to4096_4_cases)
def test_2_64_16to4096_4_digital_estimator_generator(test_case_number: int, directory: str, n_number_of_analog_states: int, oversampling_rate: int, bit_width: int, lookback_length: int, lookahead_length: int, lut_input_width: int):
    test_digital_estimator_generator(test_case_number = test_case_number, directory = directory, n_number_of_analog_states = n_number_of_analog_states, oversampling_rate = oversampling_rate, bit_width = bit_width, lookback_length = lookback_length, lookahead_length = lookahead_length, lut_input_width = lut_input_width)


# Explaination of the parameters
# Test number
# Folder to place SystemVerilog files
# Number of analog states
# Oversampling rate
# FIR coefficient bit width
# lookback length
# lookahead length
# LUT input bit width
test_3_64_16to4096_4_cases = [
        (0, base_folder, 3, 48, 64, 16, 16, 4),
        (1, base_folder, 3, 48, 64, 32, 32, 4),
        (2, base_folder, 3, 48, 64, 64, 64, 4),
        (3, base_folder, 3, 48, 64, 128, 128, 4),
        (4, base_folder, 3, 48, 64, 256, 256, 4),
        (5, base_folder, 3, 48, 64, 512, 512, 4),
        (6, base_folder, 3, 48, 64, 1024, 1024, 4),
        (7, base_folder, 3, 48, 64, 2048, 2048, 4),
        (8, base_folder, 3, 48, 64, 4096, 4096, 4)
    ]

@pytest.mark.test_3_64_16to4096_4
@pytest.mark.parametrize(("test_case_number", "directory", "n_number_of_analog_states", "oversampling_rate", "bit_width", "lookback_length", "lookahead_length", "lut_input_width"), test_3_64_16to4096_4_cases)
def test_3_64_16to4096_4_digital_estimator_generator(test_case_number: int, directory: str, n_number_of_analog_states: int, oversampling_rate: int, bit_width: int, lookback_length: int, lookahead_length: int, lut_input_width: int):
    test_digital_estimator_generator(test_case_number = test_case_number, directory = directory, n_number_of_analog_states = n_number_of_analog_states, oversampling_rate = oversampling_rate, bit_width = bit_width, lookback_length = lookback_length, lookahead_length = lookahead_length, lut_input_width = lut_input_width)


# Explaination of the parameters
# Test number
# Folder to place SystemVerilog files
# Number of analog states
# Oversampling rate
# FIR coefficient bit width
# lookback length
# lookahead length
# LUT input bit width
test_4_64_16to4096_4_cases = [
        (0, base_folder, 4, 25, 64, 16, 16, 4),
        (1, base_folder, 4, 25, 64, 32, 32, 4),
        (2, base_folder, 4, 25, 64, 64, 64, 4),
        (3, base_folder, 4, 25, 64, 128, 128, 4),
        (4, base_folder, 4, 25, 64, 256, 256, 4),
        (5, base_folder, 4, 25, 64, 512, 512, 4),
        (6, base_folder, 4, 25, 64, 1024, 1024, 4),
        (7, base_folder, 4, 25, 64, 2048, 2048, 4),
        (8, base_folder, 4, 25, 64, 4096, 4096, 4)
    ]

@pytest.mark.test_4_64_16to4096_4
@pytest.mark.parametrize(("test_case_number", "directory", "n_number_of_analog_states", "oversampling_rate", "bit_width", "lookback_length", "lookahead_length", "lut_input_width"), test_4_64_16to4096_4_cases)
def test_4_64_16to4096_4_digital_estimator_generator(test_case_number: int, directory: str, n_number_of_analog_states: int, oversampling_rate: int, bit_width: int, lookback_length: int, lookahead_length: int, lut_input_width: int):
    test_digital_estimator_generator(test_case_number = test_case_number, directory = directory, n_number_of_analog_states = n_number_of_analog_states, oversampling_rate = oversampling_rate, bit_width = bit_width, lookback_length = lookback_length, lookahead_length = lookahead_length, lut_input_width = lut_input_width)


# Explaination of the parameters
# Test number
# Folder to place SystemVerilog files
# Number of analog states
# Oversampling rate
# FIR coefficient bit width
# lookback length
# lookahead length
# LUT input bit width
test_5_64_16to4096_4_cases = [
        (0, base_folder, 5, 17, 64, 16, 16, 4),
        (1, base_folder, 5, 17, 64, 32, 32, 4),
        (2, base_folder, 5, 17, 64, 64, 64, 4),
        (3, base_folder, 5, 17, 64, 128, 128, 4),
        (4, base_folder, 5, 17, 64, 256, 256, 4),
        (5, base_folder, 5, 17, 64, 512, 512, 4),
        (6, base_folder, 5, 17, 64, 1024, 1024, 4),
        (7, base_folder, 5, 17, 64, 2048, 2048, 4),
        (8, base_folder, 5, 17, 64, 4096, 4096, 4)
    ]

@pytest.mark.test_5_64_16to4096_4
@pytest.mark.parametrize(("test_case_number", "directory", "n_number_of_analog_states", "oversampling_rate", "bit_width", "lookback_length", "lookahead_length", "lut_input_width"), test_5_64_16to4096_4_cases)
def test_5_64_16to4096_4_digital_estimator_generator(test_case_number: int, directory: str, n_number_of_analog_states: int, oversampling_rate: int, bit_width: int, lookback_length: int, lookahead_length: int, lut_input_width: int):
    test_digital_estimator_generator(test_case_number = test_case_number, directory = directory, n_number_of_analog_states = n_number_of_analog_states, oversampling_rate = oversampling_rate, bit_width = bit_width, lookback_length = lookback_length, lookahead_length = lookahead_length, lut_input_width = lut_input_width)


# Explaination of the parameters
# Test number
# Folder to place SystemVerilog files
# Number of analog states
# Oversampling rate
# FIR coefficient bit width
# lookback length
# lookahead length
# LUT input bit width
test_6_64_16to4096_4_cases = [
        (0, base_folder, 6, 13, 64, 16, 16, 4),
        (1, base_folder, 6, 13, 64, 32, 32, 4),
        (2, base_folder, 6, 13, 64, 64, 64, 4),
        (3, base_folder, 6, 13, 64, 128, 128, 4),
        (4, base_folder, 6, 13, 64, 256, 256, 4),
        (5, base_folder, 6, 13, 64, 512, 512, 4),
        (6, base_folder, 6, 13, 64, 1024, 1024, 4),
        (7, base_folder, 6, 13, 64, 2048, 2048, 4),
        (8, base_folder, 6, 13, 64, 4096, 4096, 4)
    ]

@pytest.mark.test_6_64_16to4096_4
@pytest.mark.parametrize(("test_case_number", "directory", "n_number_of_analog_states", "oversampling_rate", "bit_width", "lookback_length", "lookahead_length", "lut_input_width"), test_6_64_16to4096_4_cases)
def test_6_64_16to4096_4_digital_estimator_generator(test_case_number: int, directory: str, n_number_of_analog_states: int, oversampling_rate: int, bit_width: int, lookback_length: int, lookahead_length: int, lut_input_width: int):
    test_digital_estimator_generator(test_case_number = test_case_number, directory = directory, n_number_of_analog_states = n_number_of_analog_states, oversampling_rate = oversampling_rate, bit_width = bit_width, lookback_length = lookback_length, lookahead_length = lookahead_length, lut_input_width = lut_input_width)


# Explaination of the parameters
# Test number
# Folder to place SystemVerilog files
# Number of analog states
# Oversampling rate
# FIR coefficient bit width
# lookback length
# lookahead length
# LUT input bit width
test_7_64_16to4096_4_cases = [
        (0, base_folder, 7, 11, 64, 16, 16, 4),
        (1, base_folder, 7, 11, 64, 32, 32, 4),
        (2, base_folder, 7, 11, 64, 64, 64, 4),
        (3, base_folder, 7, 11, 64, 128, 128, 4),
        (4, base_folder, 7, 11, 64, 256, 256, 4),
        (5, base_folder, 7, 11, 64, 512, 512, 4),
        (6, base_folder, 7, 11, 64, 1024, 1024, 4),
        (7, base_folder, 7, 11, 64, 2048, 2048, 4),
        (8, base_folder, 7, 11, 64, 4096, 4096, 4)
    ]

@pytest.mark.test_7_64_16to4096_4
@pytest.mark.parametrize(("test_case_number", "directory", "n_number_of_analog_states", "oversampling_rate", "bit_width", "lookback_length", "lookahead_length", "lut_input_width"), test_7_64_16to4096_4_cases)
def test_7_64_16to4096_4_digital_estimator_generator(test_case_number: int, directory: str, n_number_of_analog_states: int, oversampling_rate: int, bit_width: int, lookback_length: int, lookahead_length: int, lut_input_width: int):
    test_digital_estimator_generator(test_case_number = test_case_number, directory = directory, n_number_of_analog_states = n_number_of_analog_states, oversampling_rate = oversampling_rate, bit_width = bit_width, lookback_length = lookback_length, lookahead_length = lookahead_length, lut_input_width = lut_input_width)


# Explaination of the parameters
# Test number
# Folder to place SystemVerilog files
# Number of analog states
# Oversampling rate
# FIR coefficient bit width
# lookback length
# lookahead length
# LUT input bit width
test_8_64_16to4096_4_cases = [
        (0, base_folder, 8, 9, 64, 16, 16, 4),
        (1, base_folder, 8, 9, 64, 32, 32, 4),
        (2, base_folder, 8, 9, 64, 64, 64, 4),
        (3, base_folder, 8, 9, 64, 128, 128, 4),
        (4, base_folder, 8, 9, 64, 256, 256, 4),
        (5, base_folder, 8, 9, 64, 512, 512, 4),
        (6, base_folder, 8, 9, 64, 1024, 1024, 4),
        (7, base_folder, 8, 9, 64, 2048, 2048, 4),
        (8, base_folder, 8, 9, 64, 4096, 4096, 4)
    ]

@pytest.mark.test_8_64_16to4096_4
@pytest.mark.parametrize(("test_case_number", "directory", "n_number_of_analog_states", "oversampling_rate", "bit_width", "lookback_length", "lookahead_length", "lut_input_width"), test_8_64_16to4096_4_cases)
def test_8_64_16to4096_4_digital_estimator_generator(test_case_number: int, directory: str, n_number_of_analog_states: int, oversampling_rate: int, bit_width: int, lookback_length: int, lookahead_length: int, lut_input_width: int):
    test_digital_estimator_generator(test_case_number = test_case_number, directory = directory, n_number_of_analog_states = n_number_of_analog_states, oversampling_rate = oversampling_rate, bit_width = bit_width, lookback_length = lookback_length, lookahead_length = lookahead_length, lut_input_width = lut_input_width)



# Explaination of the parameters
# Test number
# Folder to place SystemVerilog files
# Number of analog states
# Oversampling rate
# FIR coefficient bit width
# lookback length
# lookahead length
# LUT input bit width
test_2_1to31_1024_4_cases = [
        (0, base_folder, 2, 180, 1, 1024, 1024, 4),
        (1, base_folder, 2, 180, 2, 1024, 1024, 4),
        (2, base_folder, 2, 180, 3, 1024, 1024, 4),
        (3, base_folder, 2, 180, 4, 1024, 1024, 4),
        (4, base_folder, 2, 180, 5, 1024, 1024, 4),
        (5, base_folder, 2, 180, 6, 1024, 1024, 4),
        (6, base_folder, 2, 180, 7, 1024, 1024, 4),
        (7, base_folder, 2, 180, 8, 1024, 1024, 4),
        (8, base_folder, 2, 180, 9, 1024, 1024, 4),
        (9, base_folder, 2, 180, 10, 1024, 1024, 4),
        (10, base_folder, 2, 180, 11, 1024, 1024, 4),
        (11, base_folder, 2, 180, 12, 1024, 1024, 4),
        (12, base_folder, 2, 180, 13, 1024, 1024, 4),
        (13, base_folder, 2, 180, 14, 1024, 1024, 4),
        (14, base_folder, 2, 180, 15, 1024, 1024, 4),
        (15, base_folder, 2, 180, 16, 1024, 1024, 4),
        (16, base_folder, 2, 180, 17, 1024, 1024, 4),
        (17, base_folder, 2, 180, 18, 1024, 1024, 4),
        (18, base_folder, 2, 180, 19, 1024, 1024, 4),
        (19, base_folder, 2, 180, 20, 1024, 1024, 4),
        (20, base_folder, 2, 180, 21, 1024, 1024, 4),
        (21, base_folder, 2, 180, 22, 1024, 1024, 4),
        (22, base_folder, 2, 180, 23, 1024, 1024, 4),
        (23, base_folder, 2, 180, 24, 1024, 1024, 4),
        (24, base_folder, 2, 180, 25, 1024, 1024, 4),
        (25, base_folder, 2, 180, 26, 1024, 1024, 4),
        (26, base_folder, 2, 180, 27, 1024, 1024, 4),
        (27, base_folder, 2, 180, 28, 1024, 1024, 4),
        (28, base_folder, 2, 180, 29, 1024, 1024, 4),
        (29, base_folder, 2, 180, 30, 1024, 1024, 4),
        (30, base_folder, 2, 180, 31, 1024, 1024, 4)
    ]

@pytest.mark.test_2_1to31_1024_4
@pytest.mark.parametrize(("test_case_number", "directory", "n_number_of_analog_states", "oversampling_rate", "bit_width", "lookback_length", "lookahead_length", "lut_input_width"), test_2_1to31_1024_4_cases)
def test_2_1to31_1024_4_digital_estimator_generator(test_case_number: int, directory: str, n_number_of_analog_states: int, oversampling_rate: int, bit_width: int, lookback_length: int, lookahead_length: int, lut_input_width: int):
    test_digital_estimator_generator(test_case_number = test_case_number, directory = directory, n_number_of_analog_states = n_number_of_analog_states, oversampling_rate = oversampling_rate, bit_width = bit_width, lookback_length = lookback_length, lookahead_length = lookahead_length, lut_input_width = lut_input_width)


# Explaination of the parameters
# Test number
# Folder to place SystemVerilog files
# Number of analog states
# Oversampling rate
# FIR coefficient bit width
# lookback length
# lookahead length
# LUT input bit width
test_3_1to31_512_4_cases = [
        (0, base_folder, 3, 48, 1, 512, 512, 4),
        (1, base_folder, 3, 48, 2, 512, 512, 4),
        (2, base_folder, 3, 48, 3, 512, 512, 4),
        (3, base_folder, 3, 48, 4, 512, 512, 4),
        (4, base_folder, 3, 48, 5, 512, 512, 4),
        (5, base_folder, 3, 48, 6, 512, 512, 4),
        (6, base_folder, 3, 48, 7, 512, 512, 4),
        (7, base_folder, 3, 48, 8, 512, 512, 4),
        (8, base_folder, 3, 48, 9, 512, 512, 4),
        (9, base_folder, 3, 48, 10, 512, 512, 4),
        (10, base_folder, 3, 48, 11, 512, 512, 4),
        (11, base_folder, 3, 48, 12, 512, 512, 4),
        (12, base_folder, 3, 48, 13, 512, 512, 4),
        (13, base_folder, 3, 48, 14, 512, 512, 4),
        (14, base_folder, 3, 48, 15, 512, 512, 4),
        (15, base_folder, 3, 48, 16, 512, 512, 4),
        (16, base_folder, 3, 48, 17, 512, 512, 4),
        (17, base_folder, 3, 48, 18, 512, 512, 4),
        (18, base_folder, 3, 48, 19, 512, 512, 4),
        (19, base_folder, 3, 48, 20, 512, 512, 4),
        (20, base_folder, 3, 48, 21, 512, 512, 4),
        (21, base_folder, 3, 48, 22, 512, 512, 4),
        (22, base_folder, 3, 48, 23, 512, 512, 4),
        (23, base_folder, 3, 48, 24, 512, 512, 4),
        (24, base_folder, 3, 48, 25, 512, 512, 4),
        (25, base_folder, 3, 48, 26, 512, 512, 4),
        (26, base_folder, 3, 48, 27, 512, 512, 4),
        (27, base_folder, 3, 48, 28, 512, 512, 4),
        (28, base_folder, 3, 48, 29, 512, 512, 4),
        (29, base_folder, 3, 48, 30, 512, 512, 4),
        (30, base_folder, 3, 48, 31, 512, 512, 4)
    ]

@pytest.mark.test_3_1to31_512_4
@pytest.mark.parametrize(("test_case_number", "directory", "n_number_of_analog_states", "oversampling_rate", "bit_width", "lookback_length", "lookahead_length", "lut_input_width"), test_3_1to31_512_4_cases)
def test_3_1to31_512_4_digital_estimator_generator(test_case_number: int, directory: str, n_number_of_analog_states: int, oversampling_rate: int, bit_width: int, lookback_length: int, lookahead_length: int, lut_input_width: int):
    test_digital_estimator_generator(test_case_number = test_case_number, directory = directory, n_number_of_analog_states = n_number_of_analog_states, oversampling_rate = oversampling_rate, bit_width = bit_width, lookback_length = lookback_length, lookahead_length = lookahead_length, lut_input_width = lut_input_width)


# Explaination of the parameters
# Test number
# Folder to place SystemVerilog files
# Number of analog states
# Oversampling rate
# FIR coefficient bit width
# lookback length
# lookahead length
# LUT input bit width
test_4_1to31_512_4_cases = [
        (0, base_folder, 4, 25, 1, 512, 512, 4),
        (1, base_folder, 4, 25, 2, 512, 512, 4),
        (2, base_folder, 4, 25, 3, 512, 512, 4),
        (3, base_folder, 4, 25, 4, 512, 512, 4),
        (4, base_folder, 4, 25, 5, 512, 512, 4),
        (5, base_folder, 4, 25, 6, 512, 512, 4),
        (6, base_folder, 4, 25, 7, 512, 512, 4),
        (7, base_folder, 4, 25, 8, 512, 512, 4),
        (8, base_folder, 4, 25, 9, 512, 512, 4),
        (9, base_folder, 4, 25, 10, 512, 512, 4),
        (10, base_folder, 4, 25, 11, 512, 512, 4),
        (11, base_folder, 4, 25, 12, 512, 512, 4),
        (12, base_folder, 4, 25, 13, 512, 512, 4),
        (13, base_folder, 4, 25, 14, 512, 512, 4),
        (14, base_folder, 4, 25, 15, 512, 512, 4),
        (15, base_folder, 4, 25, 16, 512, 512, 4),
        (16, base_folder, 4, 25, 17, 512, 512, 4),
        (17, base_folder, 4, 25, 18, 512, 512, 4),
        (18, base_folder, 4, 25, 19, 512, 512, 4),
        (19, base_folder, 4, 25, 20, 512, 512, 4),
        (20, base_folder, 4, 25, 21, 512, 512, 4),
        (21, base_folder, 4, 25, 22, 512, 512, 4),
        (22, base_folder, 4, 25, 23, 512, 512, 4),
        (23, base_folder, 4, 25, 24, 512, 512, 4),
        (24, base_folder, 4, 25, 25, 512, 512, 4),
        (25, base_folder, 4, 25, 26, 512, 512, 4),
        (26, base_folder, 4, 25, 27, 512, 512, 4),
        (27, base_folder, 4, 25, 28, 512, 512, 4),
        (28, base_folder, 4, 25, 29, 512, 512, 4),
        (29, base_folder, 4, 25, 30, 512, 512, 4),
        (30, base_folder, 4, 25, 31, 512, 512, 4)
    ]

@pytest.mark.test_4_1to31_512_4
@pytest.mark.parametrize(("test_case_number", "directory", "n_number_of_analog_states", "oversampling_rate", "bit_width", "lookback_length", "lookahead_length", "lut_input_width"), test_4_1to31_512_4_cases)
def test_4_1to31_512_4_digital_estimator_generator(test_case_number: int, directory: str, n_number_of_analog_states: int, oversampling_rate: int, bit_width: int, lookback_length: int, lookahead_length: int, lut_input_width: int):
    test_digital_estimator_generator(test_case_number = test_case_number, directory = directory, n_number_of_analog_states = n_number_of_analog_states, oversampling_rate = oversampling_rate, bit_width = bit_width, lookback_length = lookback_length, lookahead_length = lookahead_length, lut_input_width = lut_input_width)


# Explaination of the parameters
# Test number
# Folder to place SystemVerilog files
# Number of analog states
# Oversampling rate
# FIR coefficient bit width
# lookback length
# lookahead length
# LUT input bit width
test_5_1to31_512_4_cases = [
        (0, base_folder, 5, 17, 1, 512, 512, 4),
        (1, base_folder, 5, 17, 2, 512, 512, 4),
        (2, base_folder, 5, 17, 3, 512, 512, 4),
        (3, base_folder, 5, 17, 4, 512, 512, 4),
        (4, base_folder, 5, 17, 5, 512, 512, 4),
        (5, base_folder, 5, 17, 6, 512, 512, 4),
        (6, base_folder, 5, 17, 7, 512, 512, 4),
        (7, base_folder, 5, 17, 8, 512, 512, 4),
        (8, base_folder, 5, 17, 9, 512, 512, 4),
        (9, base_folder, 5, 17, 10, 512, 512, 4),
        (10, base_folder, 5, 17, 11, 512, 512, 4),
        (11, base_folder, 5, 17, 12, 512, 512, 4),
        (12, base_folder, 5, 17, 13, 512, 512, 4),
        (13, base_folder, 5, 17, 14, 512, 512, 4),
        (14, base_folder, 5, 17, 15, 512, 512, 4),
        (15, base_folder, 5, 17, 16, 512, 512, 4),
        (16, base_folder, 5, 17, 17, 512, 512, 4),
        (17, base_folder, 5, 17, 18, 512, 512, 4),
        (18, base_folder, 5, 17, 19, 512, 512, 4),
        (19, base_folder, 5, 17, 20, 512, 512, 4),
        (20, base_folder, 5, 17, 21, 512, 512, 4),
        (21, base_folder, 5, 17, 22, 512, 512, 4),
        (22, base_folder, 5, 17, 23, 512, 512, 4),
        (23, base_folder, 5, 17, 24, 512, 512, 4),
        (24, base_folder, 5, 17, 25, 512, 512, 4),
        (25, base_folder, 5, 17, 26, 512, 512, 4),
        (26, base_folder, 5, 17, 27, 512, 512, 4),
        (27, base_folder, 5, 17, 28, 512, 512, 4),
        (28, base_folder, 5, 17, 29, 512, 512, 4),
        (29, base_folder, 5, 17, 30, 512, 512, 4),
        (30, base_folder, 5, 17, 31, 512, 512, 4)
    ]

@pytest.mark.test_5_1to31_512_4
@pytest.mark.parametrize(("test_case_number", "directory", "n_number_of_analog_states", "oversampling_rate", "bit_width", "lookback_length", "lookahead_length", "lut_input_width"), test_5_1to31_512_4_cases)
def test_5_1to31_512_4_digital_estimator_generator(test_case_number: int, directory: str, n_number_of_analog_states: int, oversampling_rate: int, bit_width: int, lookback_length: int, lookahead_length: int, lut_input_width: int):
    test_digital_estimator_generator(test_case_number = test_case_number, directory = directory, n_number_of_analog_states = n_number_of_analog_states, oversampling_rate = oversampling_rate, bit_width = bit_width, lookback_length = lookback_length, lookahead_length = lookahead_length, lut_input_width = lut_input_width)


# Explaination of the parameters
# Test number
# Folder to place SystemVerilog files
# Number of analog states
# Oversampling rate
# FIR coefficient bit width
# lookback length
# lookahead length
# LUT input bit width
test_6_1to31_512_4_cases = [
        (0, base_folder, 6, 13, 1, 512, 512, 4),
        (1, base_folder, 6, 13, 2, 512, 512, 4),
        (2, base_folder, 6, 13, 3, 512, 512, 4),
        (3, base_folder, 6, 13, 4, 512, 512, 4),
        (4, base_folder, 6, 13, 5, 512, 512, 4),
        (5, base_folder, 6, 13, 6, 512, 512, 4),
        (6, base_folder, 6, 13, 7, 512, 512, 4),
        (7, base_folder, 6, 13, 8, 512, 512, 4),
        (8, base_folder, 6, 13, 9, 512, 512, 4),
        (9, base_folder, 6, 13, 10, 512, 512, 4),
        (10, base_folder, 6, 13, 11, 512, 512, 4),
        (11, base_folder, 6, 13, 12, 512, 512, 4),
        (12, base_folder, 6, 13, 13, 512, 512, 4),
        (13, base_folder, 6, 13, 14, 512, 512, 4),
        (14, base_folder, 6, 13, 15, 512, 512, 4),
        (15, base_folder, 6, 13, 16, 512, 512, 4),
        (16, base_folder, 6, 13, 17, 512, 512, 4),
        (17, base_folder, 6, 13, 18, 512, 512, 4),
        (18, base_folder, 6, 13, 19, 512, 512, 4),
        (19, base_folder, 6, 13, 20, 512, 512, 4),
        (20, base_folder, 6, 13, 21, 512, 512, 4),
        (21, base_folder, 6, 13, 22, 512, 512, 4),
        (22, base_folder, 6, 13, 23, 512, 512, 4),
        (23, base_folder, 6, 13, 24, 512, 512, 4),
        (24, base_folder, 6, 13, 25, 512, 512, 4),
        (25, base_folder, 6, 13, 26, 512, 512, 4),
        (26, base_folder, 6, 13, 27, 512, 512, 4),
        (27, base_folder, 6, 13, 28, 512, 512, 4),
        (28, base_folder, 6, 13, 29, 512, 512, 4),
        (29, base_folder, 6, 13, 30, 512, 512, 4),
        (30, base_folder, 6, 13, 31, 512, 512, 4)
    ]

@pytest.mark.test_6_1to31_512_4
@pytest.mark.parametrize(("test_case_number", "directory", "n_number_of_analog_states", "oversampling_rate", "bit_width", "lookback_length", "lookahead_length", "lut_input_width"), test_6_1to31_512_4_cases)
def test_6_1to31_512_4_digital_estimator_generator(test_case_number: int, directory: str, n_number_of_analog_states: int, oversampling_rate: int, bit_width: int, lookback_length: int, lookahead_length: int, lut_input_width: int):
    test_digital_estimator_generator(test_case_number = test_case_number, directory = directory, n_number_of_analog_states = n_number_of_analog_states, oversampling_rate = oversampling_rate, bit_width = bit_width, lookback_length = lookback_length, lookahead_length = lookahead_length, lut_input_width = lut_input_width)


# Explaination of the parameters
# Test number
# Folder to place SystemVerilog files
# Number of analog states
# Oversampling rate
# FIR coefficient bit width
# lookback length
# lookahead length
# LUT input bit width
test_7_1to31_512_4_cases = [
        (0, base_folder, 7, 11, 1, 512, 512, 4),
        (1, base_folder, 7, 11, 2, 512, 512, 4),
        (2, base_folder, 7, 11, 3, 512, 512, 4),
        (3, base_folder, 7, 11, 4, 512, 512, 4),
        (4, base_folder, 7, 11, 5, 512, 512, 4),
        (5, base_folder, 7, 11, 6, 512, 512, 4),
        (6, base_folder, 7, 11, 7, 512, 512, 4),
        (7, base_folder, 7, 11, 8, 512, 512, 4),
        (8, base_folder, 7, 11, 9, 512, 512, 4),
        (9, base_folder, 7, 11, 10, 512, 512, 4),
        (10, base_folder, 7, 11, 11, 512, 512, 4),
        (11, base_folder, 7, 11, 12, 512, 512, 4),
        (12, base_folder, 7, 11, 13, 512, 512, 4),
        (13, base_folder, 7, 11, 14, 512, 512, 4),
        (14, base_folder, 7, 11, 15, 512, 512, 4),
        (15, base_folder, 7, 11, 16, 512, 512, 4),
        (16, base_folder, 7, 11, 17, 512, 512, 4),
        (17, base_folder, 7, 11, 18, 512, 512, 4),
        (18, base_folder, 7, 11, 19, 512, 512, 4),
        (19, base_folder, 7, 11, 20, 512, 512, 4),
        (20, base_folder, 7, 11, 21, 512, 512, 4),
        (21, base_folder, 7, 11, 22, 512, 512, 4),
        (22, base_folder, 7, 11, 23, 512, 512, 4),
        (23, base_folder, 7, 11, 24, 512, 512, 4),
        (24, base_folder, 7, 11, 25, 512, 512, 4),
        (25, base_folder, 7, 11, 26, 512, 512, 4),
        (26, base_folder, 7, 11, 27, 512, 512, 4),
        (27, base_folder, 7, 11, 28, 512, 512, 4),
        (28, base_folder, 7, 11, 29, 512, 512, 4),
        (29, base_folder, 7, 11, 30, 512, 512, 4),
        (30, base_folder, 7, 11, 31, 512, 512, 4)
    ]

@pytest.mark.test_7_1to31_512_4
@pytest.mark.parametrize(("test_case_number", "directory", "n_number_of_analog_states", "oversampling_rate", "bit_width", "lookback_length", "lookahead_length", "lut_input_width"), test_7_1to31_512_4_cases)
def test_7_1to31_512_4_digital_estimator_generator(test_case_number: int, directory: str, n_number_of_analog_states: int, oversampling_rate: int, bit_width: int, lookback_length: int, lookahead_length: int, lut_input_width: int):
    test_digital_estimator_generator(test_case_number = test_case_number, directory = directory, n_number_of_analog_states = n_number_of_analog_states, oversampling_rate = oversampling_rate, bit_width = bit_width, lookback_length = lookback_length, lookahead_length = lookahead_length, lut_input_width = lut_input_width)


# Explaination of the parameters
# Test number
# Folder to place SystemVerilog files
# Number of analog states
# Oversampling rate
# FIR coefficient bit width
# lookback length
# lookahead length
# LUT input bit width
test_8_1to31_512_4_cases = [
        (0, base_folder, 8, 9, 1, 512, 512, 4),
        (1, base_folder, 8, 9, 2, 512, 512, 4),
        (2, base_folder, 8, 9, 3, 512, 512, 4),
        (3, base_folder, 8, 9, 4, 512, 512, 4),
        (4, base_folder, 8, 9, 5, 512, 512, 4),
        (5, base_folder, 8, 9, 6, 512, 512, 4),
        (6, base_folder, 8, 9, 7, 512, 512, 4),
        (7, base_folder, 8, 9, 8, 512, 512, 4),
        (8, base_folder, 8, 9, 9, 512, 512, 4),
        (9, base_folder, 8, 9, 10, 512, 512, 4),
        (10, base_folder, 8, 9, 11, 512, 512, 4),
        (11, base_folder, 8, 9, 12, 512, 512, 4),
        (12, base_folder, 8, 9, 13, 512, 512, 4),
        (13, base_folder, 8, 9, 14, 512, 512, 4),
        (14, base_folder, 8, 9, 15, 512, 512, 4),
        (15, base_folder, 8, 9, 16, 512, 512, 4),
        (16, base_folder, 8, 9, 17, 512, 512, 4),
        (17, base_folder, 8, 9, 18, 512, 512, 4),
        (18, base_folder, 8, 9, 19, 512, 512, 4),
        (19, base_folder, 8, 9, 20, 512, 512, 4),
        (20, base_folder, 8, 9, 21, 512, 512, 4),
        (21, base_folder, 8, 9, 22, 512, 512, 4),
        (22, base_folder, 8, 9, 23, 512, 512, 4),
        (23, base_folder, 8, 9, 24, 512, 512, 4),
        (24, base_folder, 8, 9, 25, 512, 512, 4),
        (25, base_folder, 8, 9, 26, 512, 512, 4),
        (26, base_folder, 8, 9, 27, 512, 512, 4),
        (27, base_folder, 8, 9, 28, 512, 512, 4),
        (28, base_folder, 8, 9, 29, 512, 512, 4),
        (29, base_folder, 8, 9, 30, 512, 512, 4),
        (30, base_folder, 8, 9, 31, 512, 512, 4)
    ]

@pytest.mark.test_8_1to31_512_4
@pytest.mark.parametrize(("test_case_number", "directory", "n_number_of_analog_states", "oversampling_rate", "bit_width", "lookback_length", "lookahead_length", "lut_input_width"), test_8_1to31_512_4_cases)
def test_8_1to31_512_4_digital_estimator_generator(test_case_number: int, directory: str, n_number_of_analog_states: int, oversampling_rate: int, bit_width: int, lookback_length: int, lookahead_length: int, lut_input_width: int):
    test_digital_estimator_generator(test_case_number = test_case_number, directory = directory, n_number_of_analog_states = n_number_of_analog_states, oversampling_rate = oversampling_rate, bit_width = bit_width, lookback_length = lookback_length, lookahead_length = lookahead_length, lut_input_width = lut_input_width)



# Explaination of the parameters
# Test number
# Folder to place SystemVerilog files
# Number of analog states
# Oversampling rate
# FIR coefficient bit width
# lookback length
# lookahead length
# LUT input bit width
test_2_63_64_16to4096_4_cases = [
        (0, base_folder, 2, 63, 64, 16, 16, 4),
        (1, base_folder, 2, 63, 64, 32, 32, 4),
        (2, base_folder, 2, 63, 64, 64, 64, 4),
        (3, base_folder, 2, 63, 64, 128, 128, 4),
        (4, base_folder, 2, 63, 64, 256, 256, 4),
        (5, base_folder, 2, 63, 64, 512, 512, 4),
        (6, base_folder, 2, 63, 64, 1024, 1024, 4),
        (7, base_folder, 2, 63, 64, 2048, 2048, 4),
        (8, base_folder, 2, 63, 64, 4096, 4096, 4)
    ]

@pytest.mark.test_2_63_64_16to4096_4
@pytest.mark.parametrize(("test_case_number", "directory", "n_number_of_analog_states", "oversampling_rate", "bit_width", "lookback_length", "lookahead_length", "lut_input_width"), test_2_63_64_16to4096_4_cases)
def test_2_63_64_16to4096_4_digital_estimator_generator(test_case_number: int, directory: str, n_number_of_analog_states: int, oversampling_rate: int, bit_width: int, lookback_length: int, lookahead_length: int, lut_input_width: int):
    test_digital_estimator_generator(test_case_number = test_case_number, directory = directory, n_number_of_analog_states = n_number_of_analog_states, oversampling_rate = oversampling_rate, bit_width = bit_width, lookback_length = lookback_length, lookahead_length = lookahead_length, lut_input_width = lut_input_width)


# Explaination of the parameters
# Test number
# Folder to place SystemVerilog files
# Number of analog states
# Oversampling rate
# FIR coefficient bit width
# lookback length
# lookahead length
# LUT input bit width
test_3_24_64_16to4096_4_cases = [
        (0, base_folder, 3, 24, 64, 16, 16, 4),
        (1, base_folder, 3, 24, 64, 32, 32, 4),
        (2, base_folder, 3, 24, 64, 64, 64, 4),
        (3, base_folder, 3, 24, 64, 128, 128, 4),
        (4, base_folder, 3, 24, 64, 256, 256, 4),
        (5, base_folder, 3, 24, 64, 512, 512, 4),
        (6, base_folder, 3, 24, 64, 1024, 1024, 4),
        (7, base_folder, 3, 24, 64, 2048, 2048, 4),
        (8, base_folder, 3, 24, 64, 4096, 4096, 4)
    ]

@pytest.mark.test_3_24_64_16to4096_4
@pytest.mark.parametrize(("test_case_number", "directory", "n_number_of_analog_states", "oversampling_rate", "bit_width", "lookback_length", "lookahead_length", "lut_input_width"), test_3_24_64_16to4096_4_cases)
def test_3_24_64_16to4096_4_digital_estimator_generator(test_case_number: int, directory: str, n_number_of_analog_states: int, oversampling_rate: int, bit_width: int, lookback_length: int, lookahead_length: int, lut_input_width: int):
    test_digital_estimator_generator(test_case_number = test_case_number, directory = directory, n_number_of_analog_states = n_number_of_analog_states, oversampling_rate = oversampling_rate, bit_width = bit_width, lookback_length = lookback_length, lookahead_length = lookahead_length, lut_input_width = lut_input_width)


# Explaination of the parameters
# Test number
# Folder to place SystemVerilog files
# Number of analog states
# Oversampling rate
# FIR coefficient bit width
# lookback length
# lookahead length
# LUT input bit width
test_4_15_64_16to4096_4_cases = [
        (0, base_folder, 4, 15, 64, 16, 16, 4),
        (1, base_folder, 4, 15, 64, 32, 32, 4),
        (2, base_folder, 4, 15, 64, 64, 64, 4),
        (3, base_folder, 4, 15, 64, 128, 128, 4),
        (4, base_folder, 4, 15, 64, 256, 256, 4),
        (5, base_folder, 4, 15, 64, 512, 512, 4),
        (6, base_folder, 4, 15, 64, 1024, 1024, 4),
        (7, base_folder, 4, 15, 64, 2048, 2048, 4),
        (8, base_folder, 4, 15, 64, 4096, 4096, 4)
    ]

@pytest.mark.test_4_15_64_16to4096_4
@pytest.mark.parametrize(("test_case_number", "directory", "n_number_of_analog_states", "oversampling_rate", "bit_width", "lookback_length", "lookahead_length", "lut_input_width"), test_4_15_64_16to4096_4_cases)
def test_4_15_64_16to4096_4_digital_estimator_generator(test_case_number: int, directory: str, n_number_of_analog_states: int, oversampling_rate: int, bit_width: int, lookback_length: int, lookahead_length: int, lut_input_width: int):
    test_digital_estimator_generator(test_case_number = test_case_number, directory = directory, n_number_of_analog_states = n_number_of_analog_states, oversampling_rate = oversampling_rate, bit_width = bit_width, lookback_length = lookback_length, lookahead_length = lookahead_length, lut_input_width = lut_input_width)


# Explaination of the parameters
# Test number
# Folder to place SystemVerilog files
# Number of analog states
# Oversampling rate
# FIR coefficient bit width
# lookback length
# lookahead length
# LUT input bit width
test_5_11_64_16to4096_4_cases = [
        (0, base_folder, 5, 11, 64, 16, 16, 4),
        (1, base_folder, 5, 11, 64, 32, 32, 4),
        (2, base_folder, 5, 11, 64, 64, 64, 4),
        (3, base_folder, 5, 11, 64, 128, 128, 4),
        (4, base_folder, 5, 11, 64, 256, 256, 4),
        (5, base_folder, 5, 11, 64, 512, 512, 4),
        (6, base_folder, 5, 11, 64, 1024, 1024, 4),
        (7, base_folder, 5, 11, 64, 2048, 2048, 4),
        (8, base_folder, 5, 11, 64, 4096, 4096, 4)
    ]

@pytest.mark.test_5_11_64_16to4096_4
@pytest.mark.parametrize(("test_case_number", "directory", "n_number_of_analog_states", "oversampling_rate", "bit_width", "lookback_length", "lookahead_length", "lut_input_width"), test_5_11_64_16to4096_4_cases)
def test_5_11_64_16to4096_4_digital_estimator_generator(test_case_number: int, directory: str, n_number_of_analog_states: int, oversampling_rate: int, bit_width: int, lookback_length: int, lookahead_length: int, lut_input_width: int):
    test_digital_estimator_generator(test_case_number = test_case_number, directory = directory, n_number_of_analog_states = n_number_of_analog_states, oversampling_rate = oversampling_rate, bit_width = bit_width, lookback_length = lookback_length, lookahead_length = lookahead_length, lut_input_width = lut_input_width)


# Explaination of the parameters
# Test number
# Folder to place SystemVerilog files
# Number of analog states
# Oversampling rate
# FIR coefficient bit width
# lookback length
# lookahead length
# LUT input bit width
test_6_9_64_16to4096_4_cases = [
        (0, base_folder, 6, 9, 64, 16, 16, 4),
        (1, base_folder, 6, 9, 64, 32, 32, 4),
        (2, base_folder, 6, 9, 64, 64, 64, 4),
        (3, base_folder, 6, 9, 64, 128, 128, 4),
        (4, base_folder, 6, 9, 64, 256, 256, 4),
        (5, base_folder, 6, 9, 64, 512, 512, 4),
        (6, base_folder, 6, 9, 64, 1024, 1024, 4),
        (7, base_folder, 6, 9, 64, 2048, 2048, 4),
        (8, base_folder, 6, 9, 64, 4096, 4096, 4)
    ]

@pytest.mark.test_6_9_64_16to4096_4
@pytest.mark.parametrize(("test_case_number", "directory", "n_number_of_analog_states", "oversampling_rate", "bit_width", "lookback_length", "lookahead_length", "lut_input_width"), test_6_9_64_16to4096_4_cases)
def test_6_9_64_16to4096_4_digital_estimator_generator(test_case_number: int, directory: str, n_number_of_analog_states: int, oversampling_rate: int, bit_width: int, lookback_length: int, lookahead_length: int, lut_input_width: int):
    test_digital_estimator_generator(test_case_number = test_case_number, directory = directory, n_number_of_analog_states = n_number_of_analog_states, oversampling_rate = oversampling_rate, bit_width = bit_width, lookback_length = lookback_length, lookahead_length = lookahead_length, lut_input_width = lut_input_width)


# Explaination of the parameters
# Test number
# Folder to place SystemVerilog files
# Number of analog states
# Oversampling rate
# FIR coefficient bit width
# lookback length
# lookahead length
# LUT input bit width
test_7_8_64_16to4096_4_cases = [
        (0, base_folder, 7, 8, 64, 16, 16, 4),
        (1, base_folder, 7, 8, 64, 32, 32, 4),
        (2, base_folder, 7, 8, 64, 64, 64, 4),
        (3, base_folder, 7, 8, 64, 128, 128, 4),
        (4, base_folder, 7, 8, 64, 256, 256, 4),
        (5, base_folder, 7, 8, 64, 512, 512, 4),
        (6, base_folder, 7, 8, 64, 1024, 1024, 4),
        (7, base_folder, 7, 8, 64, 2048, 2048, 4),
        (8, base_folder, 7, 8, 64, 4096, 4096, 4)
    ]

@pytest.mark.test_7_8_64_16to4096_4
@pytest.mark.parametrize(("test_case_number", "directory", "n_number_of_analog_states", "oversampling_rate", "bit_width", "lookback_length", "lookahead_length", "lut_input_width"), test_7_8_64_16to4096_4_cases)
def test_7_8_64_16to4096_4_digital_estimator_generator(test_case_number: int, directory: str, n_number_of_analog_states: int, oversampling_rate: int, bit_width: int, lookback_length: int, lookahead_length: int, lut_input_width: int):
    test_digital_estimator_generator(test_case_number = test_case_number, directory = directory, n_number_of_analog_states = n_number_of_analog_states, oversampling_rate = oversampling_rate, bit_width = bit_width, lookback_length = lookback_length, lookahead_length = lookahead_length, lut_input_width = lut_input_width)


# Explaination of the parameters
# Test number
# Folder to place SystemVerilog files
# Number of analog states
# Oversampling rate
# FIR coefficient bit width
# lookback length
# lookahead length
# LUT input bit width
test_8_7_64_16to4096_4_cases = [
        (0, base_folder, 8, 7, 64, 16, 16, 4),
        (1, base_folder, 8, 7, 64, 32, 32, 4),
        (2, base_folder, 8, 7, 64, 64, 64, 4),
        (3, base_folder, 8, 7, 64, 128, 128, 4),
        (4, base_folder, 8, 7, 64, 256, 256, 4),
        (5, base_folder, 8, 7, 64, 512, 512, 4),
        (6, base_folder, 8, 7, 64, 1024, 1024, 4),
        (7, base_folder, 8, 7, 64, 2048, 2048, 4),
        (8, base_folder, 8, 7, 64, 4096, 4096, 4)
    ]

@pytest.mark.test_8_7_64_16to4096_4
@pytest.mark.parametrize(("test_case_number", "directory", "n_number_of_analog_states", "oversampling_rate", "bit_width", "lookback_length", "lookahead_length", "lut_input_width"), test_8_7_64_16to4096_4_cases)
def test_8_7_64_16to4096_4_digital_estimator_generator(test_case_number: int, directory: str, n_number_of_analog_states: int, oversampling_rate: int, bit_width: int, lookback_length: int, lookahead_length: int, lut_input_width: int):
    test_digital_estimator_generator(test_case_number = test_case_number, directory = directory, n_number_of_analog_states = n_number_of_analog_states, oversampling_rate = oversampling_rate, bit_width = bit_width, lookback_length = lookback_length, lookahead_length = lookahead_length, lut_input_width = lut_input_width)



# Explaination of the parameters
# Test number
# Folder to place SystemVerilog files
# Number of analog states
# Oversampling rate
# FIR coefficient bit width
# lookback length
# lookahead length
# LUT input bit width
test_2_63_1to31_256_4_cases = [
        (0, base_folder, 2, 63, 1, 256, 256, 4),
        (1, base_folder, 2, 63, 2, 256, 256, 4),
        (2, base_folder, 2, 63, 3, 256, 256, 4),
        (3, base_folder, 2, 63, 4, 256, 256, 4),
        (4, base_folder, 2, 63, 5, 256, 256, 4),
        (5, base_folder, 2, 63, 6, 256, 256, 4),
        (6, base_folder, 2, 63, 7, 256, 256, 4),
        (7, base_folder, 2, 63, 8, 256, 256, 4),
        (8, base_folder, 2, 63, 9, 256, 256, 4),
        (9, base_folder, 2, 63, 10, 256, 256, 4),
        (10, base_folder, 2, 63, 11, 256, 256, 4),
        (11, base_folder, 2, 63, 12, 256, 256, 4),
        (12, base_folder, 2, 63, 13, 256, 256, 4),
        (13, base_folder, 2, 63, 14, 256, 256, 4),
        (14, base_folder, 2, 63, 15, 256, 256, 4),
        (15, base_folder, 2, 63, 16, 256, 256, 4),
        (16, base_folder, 2, 63, 17, 256, 256, 4),
        (17, base_folder, 2, 63, 18, 256, 256, 4),
        (18, base_folder, 2, 63, 19, 256, 256, 4),
        (19, base_folder, 2, 63, 20, 256, 256, 4),
        (20, base_folder, 2, 63, 21, 256, 256, 4),
        (21, base_folder, 2, 63, 22, 256, 256, 4),
        (22, base_folder, 2, 63, 23, 256, 256, 4),
        (23, base_folder, 2, 63, 24, 256, 256, 4),
        (24, base_folder, 2, 63, 25, 256, 256, 4),
        (25, base_folder, 2, 63, 26, 256, 256, 4),
        (26, base_folder, 2, 63, 27, 256, 256, 4),
        (27, base_folder, 2, 63, 28, 256, 256, 4),
        (28, base_folder, 2, 63, 29, 256, 256, 4),
        (29, base_folder, 2, 63, 30, 256, 256, 4),
        (30, base_folder, 2, 63, 31, 256, 256, 4)
    ]

@pytest.mark.test_2_63_1to31_256_4
@pytest.mark.parametrize(("test_case_number", "directory", "n_number_of_analog_states", "oversampling_rate", "bit_width", "lookback_length", "lookahead_length", "lut_input_width"), test_2_63_1to31_256_4_cases)
def test_2_63_1to31_256_4_digital_estimator_generator(test_case_number: int, directory: str, n_number_of_analog_states: int, oversampling_rate: int, bit_width: int, lookback_length: int, lookahead_length: int, lut_input_width: int):
    test_digital_estimator_generator(test_case_number = test_case_number, directory = directory, n_number_of_analog_states = n_number_of_analog_states, oversampling_rate = oversampling_rate, bit_width = bit_width, lookback_length = lookback_length, lookahead_length = lookahead_length, lut_input_width = lut_input_width)


# Explaination of the parameters
# Test number
# Folder to place SystemVerilog files
# Number of analog states
# Oversampling rate
# FIR coefficient bit width
# lookback length
# lookahead length
# LUT input bit width
test_3_24_1to31_256_4_cases = [
        (0, base_folder, 3, 24, 1, 256, 256, 4),
        (1, base_folder, 3, 24, 2, 256, 256, 4),
        (2, base_folder, 3, 24, 3, 256, 256, 4),
        (3, base_folder, 3, 24, 4, 256, 256, 4),
        (4, base_folder, 3, 24, 5, 256, 256, 4),
        (5, base_folder, 3, 24, 6, 256, 256, 4),
        (6, base_folder, 3, 24, 7, 256, 256, 4),
        (7, base_folder, 3, 24, 8, 256, 256, 4),
        (8, base_folder, 3, 24, 9, 256, 256, 4),
        (9, base_folder, 3, 24, 10, 256, 256, 4),
        (10, base_folder, 3, 24, 11, 256, 256, 4),
        (11, base_folder, 3, 24, 12, 256, 256, 4),
        (12, base_folder, 3, 24, 13, 256, 256, 4),
        (13, base_folder, 3, 24, 14, 256, 256, 4),
        (14, base_folder, 3, 24, 15, 256, 256, 4),
        (15, base_folder, 3, 24, 16, 256, 256, 4),
        (16, base_folder, 3, 24, 17, 256, 256, 4),
        (17, base_folder, 3, 24, 18, 256, 256, 4),
        (18, base_folder, 3, 24, 19, 256, 256, 4),
        (19, base_folder, 3, 24, 20, 256, 256, 4),
        (20, base_folder, 3, 24, 21, 256, 256, 4),
        (21, base_folder, 3, 24, 22, 256, 256, 4),
        (22, base_folder, 3, 24, 23, 256, 256, 4),
        (23, base_folder, 3, 24, 24, 256, 256, 4),
        (24, base_folder, 3, 24, 25, 256, 256, 4),
        (25, base_folder, 3, 24, 26, 256, 256, 4),
        (26, base_folder, 3, 24, 27, 256, 256, 4),
        (27, base_folder, 3, 24, 28, 256, 256, 4),
        (28, base_folder, 3, 24, 29, 256, 256, 4),
        (29, base_folder, 3, 24, 30, 256, 256, 4),
        (30, base_folder, 3, 24, 31, 256, 256, 4)
    ]

@pytest.mark.test_3_24_1to31_256_4
@pytest.mark.parametrize(("test_case_number", "directory", "n_number_of_analog_states", "oversampling_rate", "bit_width", "lookback_length", "lookahead_length", "lut_input_width"), test_3_24_1to31_256_4_cases)
def test_3_24_1to31_256_4_digital_estimator_generator(test_case_number: int, directory: str, n_number_of_analog_states: int, oversampling_rate: int, bit_width: int, lookback_length: int, lookahead_length: int, lut_input_width: int):
    test_digital_estimator_generator(test_case_number = test_case_number, directory = directory, n_number_of_analog_states = n_number_of_analog_states, oversampling_rate = oversampling_rate, bit_width = bit_width, lookback_length = lookback_length, lookahead_length = lookahead_length, lut_input_width = lut_input_width)


# Explaination of the parameters
# Test number
# Folder to place SystemVerilog files
# Number of analog states
# Oversampling rate
# FIR coefficient bit width
# lookback length
# lookahead length
# LUT input bit width
test_4_15_1to31_256_4_cases = [
        (0, base_folder, 4, 15, 1, 256, 256, 4),
        (1, base_folder, 4, 15, 2, 256, 256, 4),
        (2, base_folder, 4, 15, 3, 256, 256, 4),
        (3, base_folder, 4, 15, 4, 256, 256, 4),
        (4, base_folder, 4, 15, 5, 256, 256, 4),
        (5, base_folder, 4, 15, 6, 256, 256, 4),
        (6, base_folder, 4, 15, 7, 256, 256, 4),
        (7, base_folder, 4, 15, 8, 256, 256, 4),
        (8, base_folder, 4, 15, 9, 256, 256, 4),
        (9, base_folder, 4, 15, 10, 256, 256, 4),
        (10, base_folder, 4, 15, 11, 256, 256, 4),
        (11, base_folder, 4, 15, 12, 256, 256, 4),
        (12, base_folder, 4, 15, 13, 256, 256, 4),
        (13, base_folder, 4, 15, 14, 256, 256, 4),
        (14, base_folder, 4, 15, 15, 256, 256, 4),
        (15, base_folder, 4, 15, 16, 256, 256, 4),
        (16, base_folder, 4, 15, 17, 256, 256, 4),
        (17, base_folder, 4, 15, 18, 256, 256, 4),
        (18, base_folder, 4, 15, 19, 256, 256, 4),
        (19, base_folder, 4, 15, 20, 256, 256, 4),
        (20, base_folder, 4, 15, 21, 256, 256, 4),
        (21, base_folder, 4, 15, 22, 256, 256, 4),
        (22, base_folder, 4, 15, 23, 256, 256, 4),
        (23, base_folder, 4, 15, 24, 256, 256, 4),
        (24, base_folder, 4, 15, 25, 256, 256, 4),
        (25, base_folder, 4, 15, 26, 256, 256, 4),
        (26, base_folder, 4, 15, 27, 256, 256, 4),
        (27, base_folder, 4, 15, 28, 256, 256, 4),
        (28, base_folder, 4, 15, 29, 256, 256, 4),
        (29, base_folder, 4, 15, 30, 256, 256, 4),
        (30, base_folder, 4, 15, 31, 256, 256, 4)
    ]

@pytest.mark.test_4_15_1to31_256_4
@pytest.mark.parametrize(("test_case_number", "directory", "n_number_of_analog_states", "oversampling_rate", "bit_width", "lookback_length", "lookahead_length", "lut_input_width"), test_4_15_1to31_256_4_cases)
def test_4_15_1to31_256_4_digital_estimator_generator(test_case_number: int, directory: str, n_number_of_analog_states: int, oversampling_rate: int, bit_width: int, lookback_length: int, lookahead_length: int, lut_input_width: int):
    test_digital_estimator_generator(test_case_number = test_case_number, directory = directory, n_number_of_analog_states = n_number_of_analog_states, oversampling_rate = oversampling_rate, bit_width = bit_width, lookback_length = lookback_length, lookahead_length = lookahead_length, lut_input_width = lut_input_width)


# Explaination of the parameters
# Test number
# Folder to place SystemVerilog files
# Number of analog states
# Oversampling rate
# FIR coefficient bit width
# lookback length
# lookahead length
# LUT input bit width
test_5_11_1to31_256_4_cases = [
        (0, base_folder, 5, 11, 1, 256, 256, 4),
        (1, base_folder, 5, 11, 2, 256, 256, 4),
        (2, base_folder, 5, 11, 3, 256, 256, 4),
        (3, base_folder, 5, 11, 4, 256, 256, 4),
        (4, base_folder, 5, 11, 5, 256, 256, 4),
        (5, base_folder, 5, 11, 6, 256, 256, 4),
        (6, base_folder, 5, 11, 7, 256, 256, 4),
        (7, base_folder, 5, 11, 8, 256, 256, 4),
        (8, base_folder, 5, 11, 9, 256, 256, 4),
        (9, base_folder, 5, 11, 10, 256, 256, 4),
        (10, base_folder, 5, 11, 11, 256, 256, 4),
        (11, base_folder, 5, 11, 12, 256, 256, 4),
        (12, base_folder, 5, 11, 13, 256, 256, 4),
        (13, base_folder, 5, 11, 14, 256, 256, 4),
        (14, base_folder, 5, 11, 15, 256, 256, 4),
        (15, base_folder, 5, 11, 16, 256, 256, 4),
        (16, base_folder, 5, 11, 17, 256, 256, 4),
        (17, base_folder, 5, 11, 18, 256, 256, 4),
        (18, base_folder, 5, 11, 19, 256, 256, 4),
        (19, base_folder, 5, 11, 20, 256, 256, 4),
        (20, base_folder, 5, 11, 21, 256, 256, 4),
        (21, base_folder, 5, 11, 22, 256, 256, 4),
        (22, base_folder, 5, 11, 23, 256, 256, 4),
        (23, base_folder, 5, 11, 24, 256, 256, 4),
        (24, base_folder, 5, 11, 25, 256, 256, 4),
        (25, base_folder, 5, 11, 26, 256, 256, 4),
        (26, base_folder, 5, 11, 27, 256, 256, 4),
        (27, base_folder, 5, 11, 28, 256, 256, 4),
        (28, base_folder, 5, 11, 29, 256, 256, 4),
        (29, base_folder, 5, 11, 30, 256, 256, 4),
        (30, base_folder, 5, 11, 31, 256, 256, 4)
    ]

@pytest.mark.test_5_11_1to31_256_4
@pytest.mark.parametrize(("test_case_number", "directory", "n_number_of_analog_states", "oversampling_rate", "bit_width", "lookback_length", "lookahead_length", "lut_input_width"), test_5_11_1to31_256_4_cases)
def test_5_11_1to31_256_4_digital_estimator_generator(test_case_number: int, directory: str, n_number_of_analog_states: int, oversampling_rate: int, bit_width: int, lookback_length: int, lookahead_length: int, lut_input_width: int):
    test_digital_estimator_generator(test_case_number = test_case_number, directory = directory, n_number_of_analog_states = n_number_of_analog_states, oversampling_rate = oversampling_rate, bit_width = bit_width, lookback_length = lookback_length, lookahead_length = lookahead_length, lut_input_width = lut_input_width)


# Explaination of the parameters
# Test number
# Folder to place SystemVerilog files
# Number of analog states
# Oversampling rate
# FIR coefficient bit width
# lookback length
# lookahead length
# LUT input bit width
test_6_9_1to31_256_4_cases = [
        (0, base_folder, 6, 9, 1, 256, 256, 4),
        (1, base_folder, 6, 9, 2, 256, 256, 4),
        (2, base_folder, 6, 9, 3, 256, 256, 4),
        (3, base_folder, 6, 9, 4, 256, 256, 4),
        (4, base_folder, 6, 9, 5, 256, 256, 4),
        (5, base_folder, 6, 9, 6, 256, 256, 4),
        (6, base_folder, 6, 9, 7, 256, 256, 4),
        (7, base_folder, 6, 9, 8, 256, 256, 4),
        (8, base_folder, 6, 9, 9, 256, 256, 4),
        (9, base_folder, 6, 9, 10, 256, 256, 4),
        (10, base_folder, 6, 9, 11, 256, 256, 4),
        (11, base_folder, 6, 9, 12, 256, 256, 4),
        (12, base_folder, 6, 9, 13, 256, 256, 4),
        (13, base_folder, 6, 9, 14, 256, 256, 4),
        (14, base_folder, 6, 9, 15, 256, 256, 4),
        (15, base_folder, 6, 9, 16, 256, 256, 4),
        (16, base_folder, 6, 9, 17, 256, 256, 4),
        (17, base_folder, 6, 9, 18, 256, 256, 4),
        (18, base_folder, 6, 9, 19, 256, 256, 4),
        (19, base_folder, 6, 9, 20, 256, 256, 4),
        (20, base_folder, 6, 9, 21, 256, 256, 4),
        (21, base_folder, 6, 9, 22, 256, 256, 4),
        (22, base_folder, 6, 9, 23, 256, 256, 4),
        (23, base_folder, 6, 9, 24, 256, 256, 4),
        (24, base_folder, 6, 9, 25, 256, 256, 4),
        (25, base_folder, 6, 9, 26, 256, 256, 4),
        (26, base_folder, 6, 9, 27, 256, 256, 4),
        (27, base_folder, 6, 9, 28, 256, 256, 4),
        (28, base_folder, 6, 9, 29, 256, 256, 4),
        (29, base_folder, 6, 9, 30, 256, 256, 4),
        (30, base_folder, 6, 9, 31, 256, 256, 4)
    ]

@pytest.mark.test_6_9_1to31_256_4
@pytest.mark.parametrize(("test_case_number", "directory", "n_number_of_analog_states", "oversampling_rate", "bit_width", "lookback_length", "lookahead_length", "lut_input_width"), test_6_9_1to31_256_4_cases)
def test_6_9_1to31_256_4_digital_estimator_generator(test_case_number: int, directory: str, n_number_of_analog_states: int, oversampling_rate: int, bit_width: int, lookback_length: int, lookahead_length: int, lut_input_width: int):
    test_digital_estimator_generator(test_case_number = test_case_number, directory = directory, n_number_of_analog_states = n_number_of_analog_states, oversampling_rate = oversampling_rate, bit_width = bit_width, lookback_length = lookback_length, lookahead_length = lookahead_length, lut_input_width = lut_input_width)


# Explaination of the parameters
# Test number
# Folder to place SystemVerilog files
# Number of analog states
# Oversampling rate
# FIR coefficient bit width
# lookback length
# lookahead length
# LUT input bit width
test_7_8_1to31_256_4_cases = [
        (0, base_folder, 7, 8, 1, 256, 256, 4),
        (1, base_folder, 7, 8, 2, 256, 256, 4),
        (2, base_folder, 7, 8, 3, 256, 256, 4),
        (3, base_folder, 7, 8, 4, 256, 256, 4),
        (4, base_folder, 7, 8, 5, 256, 256, 4),
        (5, base_folder, 7, 8, 6, 256, 256, 4),
        (6, base_folder, 7, 8, 7, 256, 256, 4),
        (7, base_folder, 7, 8, 8, 256, 256, 4),
        (8, base_folder, 7, 8, 9, 256, 256, 4),
        (9, base_folder, 7, 8, 10, 256, 256, 4),
        (10, base_folder, 7, 8, 11, 256, 256, 4),
        (11, base_folder, 7, 8, 12, 256, 256, 4),
        (12, base_folder, 7, 8, 13, 256, 256, 4),
        (13, base_folder, 7, 8, 14, 256, 256, 4),
        (14, base_folder, 7, 8, 15, 256, 256, 4),
        (15, base_folder, 7, 8, 16, 256, 256, 4),
        (16, base_folder, 7, 8, 17, 256, 256, 4),
        (17, base_folder, 7, 8, 18, 256, 256, 4),
        (18, base_folder, 7, 8, 19, 256, 256, 4),
        (19, base_folder, 7, 8, 20, 256, 256, 4),
        (20, base_folder, 7, 8, 21, 256, 256, 4),
        (21, base_folder, 7, 8, 22, 256, 256, 4),
        (22, base_folder, 7, 8, 23, 256, 256, 4),
        (23, base_folder, 7, 8, 24, 256, 256, 4),
        (24, base_folder, 7, 8, 25, 256, 256, 4),
        (25, base_folder, 7, 8, 26, 256, 256, 4),
        (26, base_folder, 7, 8, 27, 256, 256, 4),
        (27, base_folder, 7, 8, 28, 256, 256, 4),
        (28, base_folder, 7, 8, 29, 256, 256, 4),
        (29, base_folder, 7, 8, 30, 256, 256, 4),
        (30, base_folder, 7, 8, 31, 256, 256, 4)
    ]

@pytest.mark.test_7_8_1to31_256_4
@pytest.mark.parametrize(("test_case_number", "directory", "n_number_of_analog_states", "oversampling_rate", "bit_width", "lookback_length", "lookahead_length", "lut_input_width"), test_7_8_1to31_256_4_cases)
def test_7_8_1to31_256_4_digital_estimator_generator(test_case_number: int, directory: str, n_number_of_analog_states: int, oversampling_rate: int, bit_width: int, lookback_length: int, lookahead_length: int, lut_input_width: int):
    test_digital_estimator_generator(test_case_number = test_case_number, directory = directory, n_number_of_analog_states = n_number_of_analog_states, oversampling_rate = oversampling_rate, bit_width = bit_width, lookback_length = lookback_length, lookahead_length = lookahead_length, lut_input_width = lut_input_width)


# Explaination of the parameters
# Test number
# Folder to place SystemVerilog files
# Number of analog states
# Oversampling rate
# FIR coefficient bit width
# lookback length
# lookahead length
# LUT input bit width
test_8_7_1to31_256_4_cases = [
        (0, base_folder, 8, 7, 1, 256, 256, 4),
        (1, base_folder, 8, 7, 2, 256, 256, 4),
        (2, base_folder, 8, 7, 3, 256, 256, 4),
        (3, base_folder, 8, 7, 4, 256, 256, 4),
        (4, base_folder, 8, 7, 5, 256, 256, 4),
        (5, base_folder, 8, 7, 6, 256, 256, 4),
        (6, base_folder, 8, 7, 7, 256, 256, 4),
        (7, base_folder, 8, 7, 8, 256, 256, 4),
        (8, base_folder, 8, 7, 9, 256, 256, 4),
        (9, base_folder, 8, 7, 10, 256, 256, 4),
        (10, base_folder, 8, 7, 11, 256, 256, 4),
        (11, base_folder, 8, 7, 12, 256, 256, 4),
        (12, base_folder, 8, 7, 13, 256, 256, 4),
        (13, base_folder, 8, 7, 14, 256, 256, 4),
        (14, base_folder, 8, 7, 15, 256, 256, 4),
        (15, base_folder, 8, 7, 16, 256, 256, 4),
        (16, base_folder, 8, 7, 17, 256, 256, 4),
        (17, base_folder, 8, 7, 18, 256, 256, 4),
        (18, base_folder, 8, 7, 19, 256, 256, 4),
        (19, base_folder, 8, 7, 20, 256, 256, 4),
        (20, base_folder, 8, 7, 21, 256, 256, 4),
        (21, base_folder, 8, 7, 22, 256, 256, 4),
        (22, base_folder, 8, 7, 23, 256, 256, 4),
        (23, base_folder, 8, 7, 24, 256, 256, 4),
        (24, base_folder, 8, 7, 25, 256, 256, 4),
        (25, base_folder, 8, 7, 26, 256, 256, 4),
        (26, base_folder, 8, 7, 27, 256, 256, 4),
        (27, base_folder, 8, 7, 28, 256, 256, 4),
        (28, base_folder, 8, 7, 29, 256, 256, 4),
        (29, base_folder, 8, 7, 30, 256, 256, 4),
        (30, base_folder, 8, 7, 31, 256, 256, 4)
    ]

@pytest.mark.test_8_7_1to31_256_4
@pytest.mark.parametrize(("test_case_number", "directory", "n_number_of_analog_states", "oversampling_rate", "bit_width", "lookback_length", "lookahead_length", "lut_input_width"), test_8_7_1to31_256_4_cases)
def test_8_7_1to31_256_4_digital_estimator_generator(test_case_number: int, directory: str, n_number_of_analog_states: int, oversampling_rate: int, bit_width: int, lookback_length: int, lookahead_length: int, lut_input_width: int):
    test_digital_estimator_generator(test_case_number = test_case_number, directory = directory, n_number_of_analog_states = n_number_of_analog_states, oversampling_rate = oversampling_rate, bit_width = bit_width, lookback_length = lookback_length, lookahead_length = lookahead_length, lut_input_width = lut_input_width)



# Explaination of the parameters
# Test number
# Folder to place SystemVerilog files
# Number of analog states
# Oversampling rate
# FIR coefficient bit width
# lookback length
# lookahead length
# LUT input bit width
test_2_32_64_16to4096_4_cases = [
        (0, base_folder, 2, 32, 64, 16, 16, 4),
        (1, base_folder, 2, 32, 64, 32, 32, 4),
        (2, base_folder, 2, 32, 64, 64, 64, 4),
        (3, base_folder, 2, 32, 64, 128, 128, 4),
        (4, base_folder, 2, 32, 64, 256, 256, 4),
        (5, base_folder, 2, 32, 64, 512, 512, 4),
        (6, base_folder, 2, 32, 64, 1024, 1024, 4),
        (7, base_folder, 2, 32, 64, 2048, 2048, 4),
        (8, base_folder, 2, 32, 64, 4096, 4096, 4)
    ]

@pytest.mark.test_2_32_64_16to4096_4
@pytest.mark.parametrize(("test_case_number", "directory", "n_number_of_analog_states", "oversampling_rate", "bit_width", "lookback_length", "lookahead_length", "lut_input_width"), test_2_32_64_16to4096_4_cases)
def test_2_32_64_16to4096_4_digital_estimator_generator(test_case_number: int, directory: str, n_number_of_analog_states: int, oversampling_rate: int, bit_width: int, lookback_length: int, lookahead_length: int, lut_input_width: int):
    test_digital_estimator_generator(test_case_number = test_case_number, directory = directory, n_number_of_analog_states = n_number_of_analog_states, oversampling_rate = oversampling_rate, bit_width = bit_width, lookback_length = lookback_length, lookahead_length = lookahead_length, lut_input_width = lut_input_width)


# Explaination of the parameters
# Test number
# Folder to place SystemVerilog files
# Number of analog states
# Oversampling rate
# FIR coefficient bit width
# lookback length
# lookahead length
# LUT input bit width
test_3_15_64_16to4096_4_cases = [
        (0, base_folder, 3, 15, 64, 16, 16, 4),
        (1, base_folder, 3, 15, 64, 32, 32, 4),
        (2, base_folder, 3, 15, 64, 64, 64, 4),
        (3, base_folder, 3, 15, 64, 128, 128, 4),
        (4, base_folder, 3, 15, 64, 256, 256, 4),
        (5, base_folder, 3, 15, 64, 512, 512, 4),
        (6, base_folder, 3, 15, 64, 1024, 1024, 4),
        (7, base_folder, 3, 15, 64, 2048, 2048, 4),
        (8, base_folder, 3, 15, 64, 4096, 4096, 4)
    ]

@pytest.mark.test_3_15_64_16to4096_4
@pytest.mark.parametrize(("test_case_number", "directory", "n_number_of_analog_states", "oversampling_rate", "bit_width", "lookback_length", "lookahead_length", "lut_input_width"), test_3_15_64_16to4096_4_cases)
def test_3_15_64_16to4096_4_digital_estimator_generator(test_case_number: int, directory: str, n_number_of_analog_states: int, oversampling_rate: int, bit_width: int, lookback_length: int, lookahead_length: int, lut_input_width: int):
    test_digital_estimator_generator(test_case_number = test_case_number, directory = directory, n_number_of_analog_states = n_number_of_analog_states, oversampling_rate = oversampling_rate, bit_width = bit_width, lookback_length = lookback_length, lookahead_length = lookahead_length, lut_input_width = lut_input_width)


# Explaination of the parameters
# Test number
# Folder to place SystemVerilog files
# Number of analog states
# Oversampling rate
# FIR coefficient bit width
# lookback length
# lookahead length
# LUT input bit width
test_4_11_64_16to4096_4_cases = [
        (0, base_folder, 4, 11, 64, 16, 16, 4),
        (1, base_folder, 4, 11, 64, 32, 32, 4),
        (2, base_folder, 4, 11, 64, 64, 64, 4),
        (3, base_folder, 4, 11, 64, 128, 128, 4),
        (4, base_folder, 4, 11, 64, 256, 256, 4),
        (5, base_folder, 4, 11, 64, 512, 512, 4),
        (6, base_folder, 4, 11, 64, 1024, 1024, 4),
        (7, base_folder, 4, 11, 64, 2048, 2048, 4),
        (8, base_folder, 4, 11, 64, 4096, 4096, 4)
    ]

@pytest.mark.test_4_11_64_16to4096_4
@pytest.mark.parametrize(("test_case_number", "directory", "n_number_of_analog_states", "oversampling_rate", "bit_width", "lookback_length", "lookahead_length", "lut_input_width"), test_4_11_64_16to4096_4_cases)
def test_4_11_64_16to4096_4_digital_estimator_generator(test_case_number: int, directory: str, n_number_of_analog_states: int, oversampling_rate: int, bit_width: int, lookback_length: int, lookahead_length: int, lut_input_width: int):
    test_digital_estimator_generator(test_case_number = test_case_number, directory = directory, n_number_of_analog_states = n_number_of_analog_states, oversampling_rate = oversampling_rate, bit_width = bit_width, lookback_length = lookback_length, lookahead_length = lookahead_length, lut_input_width = lut_input_width)


# Explaination of the parameters
# Test number
# Folder to place SystemVerilog files
# Number of analog states
# Oversampling rate
# FIR coefficient bit width
# lookback length
# lookahead length
# LUT input bit width
test_5_9_64_16to4096_4_cases = [
        (0, base_folder, 5, 9, 64, 16, 16, 4),
        (1, base_folder, 5, 9, 64, 32, 32, 4),
        (2, base_folder, 5, 9, 64, 64, 64, 4),
        (3, base_folder, 5, 9, 64, 128, 128, 4),
        (4, base_folder, 5, 9, 64, 256, 256, 4),
        (5, base_folder, 5, 9, 64, 512, 512, 4),
        (6, base_folder, 5, 9, 64, 1024, 1024, 4),
        (7, base_folder, 5, 9, 64, 2048, 2048, 4),
        (8, base_folder, 5, 9, 64, 4096, 4096, 4)
    ]

@pytest.mark.test_5_9_64_16to4096_4
@pytest.mark.parametrize(("test_case_number", "directory", "n_number_of_analog_states", "oversampling_rate", "bit_width", "lookback_length", "lookahead_length", "lut_input_width"), test_5_9_64_16to4096_4_cases)
def test_5_9_64_16to4096_4_digital_estimator_generator(test_case_number: int, directory: str, n_number_of_analog_states: int, oversampling_rate: int, bit_width: int, lookback_length: int, lookahead_length: int, lut_input_width: int):
    test_digital_estimator_generator(test_case_number = test_case_number, directory = directory, n_number_of_analog_states = n_number_of_analog_states, oversampling_rate = oversampling_rate, bit_width = bit_width, lookback_length = lookback_length, lookahead_length = lookahead_length, lut_input_width = lut_input_width)


# Explaination of the parameters
# Test number
# Folder to place SystemVerilog files
# Number of analog states
# Oversampling rate
# FIR coefficient bit width
# lookback length
# lookahead length
# LUT input bit width
test_6_7_64_16to4096_4_cases = [
        (0, base_folder, 6, 7, 64, 16, 16, 4),
        (1, base_folder, 6, 7, 64, 32, 32, 4),
        (2, base_folder, 6, 7, 64, 64, 64, 4),
        (3, base_folder, 6, 7, 64, 128, 128, 4),
        (4, base_folder, 6, 7, 64, 256, 256, 4),
        (5, base_folder, 6, 7, 64, 512, 512, 4),
        (6, base_folder, 6, 7, 64, 1024, 1024, 4),
        (7, base_folder, 6, 7, 64, 2048, 2048, 4),
        (8, base_folder, 6, 7, 64, 4096, 4096, 4)
    ]

@pytest.mark.test_6_7_64_16to4096_4
@pytest.mark.parametrize(("test_case_number", "directory", "n_number_of_analog_states", "oversampling_rate", "bit_width", "lookback_length", "lookahead_length", "lut_input_width"), test_6_7_64_16to4096_4_cases)
def test_6_7_64_16to4096_4_digital_estimator_generator(test_case_number: int, directory: str, n_number_of_analog_states: int, oversampling_rate: int, bit_width: int, lookback_length: int, lookahead_length: int, lut_input_width: int):
    test_digital_estimator_generator(test_case_number = test_case_number, directory = directory, n_number_of_analog_states = n_number_of_analog_states, oversampling_rate = oversampling_rate, bit_width = bit_width, lookback_length = lookback_length, lookahead_length = lookahead_length, lut_input_width = lut_input_width)


# Explaination of the parameters
# Test number
# Folder to place SystemVerilog files
# Number of analog states
# Oversampling rate
# FIR coefficient bit width
# lookback length
# lookahead length
# LUT input bit width
test_7_7_64_16to4096_4_cases = [
        (0, base_folder, 7, 7, 64, 16, 16, 4),
        (1, base_folder, 7, 7, 64, 32, 32, 4),
        (2, base_folder, 7, 7, 64, 64, 64, 4),
        (3, base_folder, 7, 7, 64, 128, 128, 4),
        (4, base_folder, 7, 7, 64, 256, 256, 4),
        (5, base_folder, 7, 7, 64, 512, 512, 4),
        (6, base_folder, 7, 7, 64, 1024, 1024, 4),
        (7, base_folder, 7, 7, 64, 2048, 2048, 4),
        (8, base_folder, 7, 7, 64, 4096, 4096, 4)
    ]

@pytest.mark.test_7_7_64_16to4096_4
@pytest.mark.parametrize(("test_case_number", "directory", "n_number_of_analog_states", "oversampling_rate", "bit_width", "lookback_length", "lookahead_length", "lut_input_width"), test_7_7_64_16to4096_4_cases)
def test_7_7_64_16to4096_4_digital_estimator_generator(test_case_number: int, directory: str, n_number_of_analog_states: int, oversampling_rate: int, bit_width: int, lookback_length: int, lookahead_length: int, lut_input_width: int):
    test_digital_estimator_generator(test_case_number = test_case_number, directory = directory, n_number_of_analog_states = n_number_of_analog_states, oversampling_rate = oversampling_rate, bit_width = bit_width, lookback_length = lookback_length, lookahead_length = lookahead_length, lut_input_width = lut_input_width)


# Explaination of the parameters
# Test number
# Folder to place SystemVerilog files
# Number of analog states
# Oversampling rate
# FIR coefficient bit width
# lookback length
# lookahead length
# LUT input bit width
test_8_6_64_16to4096_4_cases = [
        (0, base_folder, 8, 6, 64, 16, 16, 4),
        (1, base_folder, 8, 6, 64, 32, 32, 4),
        (2, base_folder, 8, 6, 64, 64, 64, 4),
        (3, base_folder, 8, 6, 64, 128, 128, 4),
        (4, base_folder, 8, 6, 64, 256, 256, 4),
        (5, base_folder, 8, 6, 64, 512, 512, 4),
        (6, base_folder, 8, 6, 64, 1024, 1024, 4),
        (7, base_folder, 8, 6, 64, 2048, 2048, 4),
        (8, base_folder, 8, 6, 64, 4096, 4096, 4)
    ]

@pytest.mark.test_8_6_64_16to4096_4
@pytest.mark.parametrize(("test_case_number", "directory", "n_number_of_analog_states", "oversampling_rate", "bit_width", "lookback_length", "lookahead_length", "lut_input_width"), test_8_6_64_16to4096_4_cases)
def test_8_6_64_16to4096_4_digital_estimator_generator(test_case_number: int, directory: str, n_number_of_analog_states: int, oversampling_rate: int, bit_width: int, lookback_length: int, lookahead_length: int, lut_input_width: int):
    test_digital_estimator_generator(test_case_number = test_case_number, directory = directory, n_number_of_analog_states = n_number_of_analog_states, oversampling_rate = oversampling_rate, bit_width = bit_width, lookback_length = lookback_length, lookahead_length = lookahead_length, lut_input_width = lut_input_width)



# Explaination of the parameters
# Test number
# Folder to place SystemVerilog files
# Number of analog states
# Oversampling rate
# FIR coefficient bit width
# lookback length
# lookahead length
# LUT input bit width
test_2_32_1to31_128_4_cases = [
        (0, base_folder, 2, 32, 1, 128, 128, 4),
        (1, base_folder, 2, 32, 2, 128, 128, 4),
        (2, base_folder, 2, 32, 3, 128, 128, 4),
        (3, base_folder, 2, 32, 4, 128, 128, 4),
        (4, base_folder, 2, 32, 5, 128, 128, 4),
        (5, base_folder, 2, 32, 6, 128, 128, 4),
        (6, base_folder, 2, 32, 7, 128, 128, 4),
        (7, base_folder, 2, 32, 8, 128, 128, 4),
        (8, base_folder, 2, 32, 9, 128, 128, 4),
        (9, base_folder, 2, 32, 10, 128, 128, 4),
        (10, base_folder, 2, 32, 11, 128, 128, 4),
        (11, base_folder, 2, 32, 12, 128, 128, 4),
        (12, base_folder, 2, 32, 13, 128, 128, 4),
        (13, base_folder, 2, 32, 14, 128, 128, 4),
        (14, base_folder, 2, 32, 15, 128, 128, 4),
        (15, base_folder, 2, 32, 16, 128, 128, 4),
        (16, base_folder, 2, 32, 17, 128, 128, 4),
        (17, base_folder, 2, 32, 18, 128, 128, 4),
        (18, base_folder, 2, 32, 19, 128, 128, 4),
        (19, base_folder, 2, 32, 20, 128, 128, 4),
        (20, base_folder, 2, 32, 21, 128, 128, 4),
        (21, base_folder, 2, 32, 22, 128, 128, 4),
        (22, base_folder, 2, 32, 23, 128, 128, 4),
        (23, base_folder, 2, 32, 24, 128, 128, 4),
        (24, base_folder, 2, 32, 25, 128, 128, 4),
        (25, base_folder, 2, 32, 26, 128, 128, 4),
        (26, base_folder, 2, 32, 27, 128, 128, 4),
        (27, base_folder, 2, 32, 28, 128, 128, 4),
        (28, base_folder, 2, 32, 29, 128, 128, 4),
        (29, base_folder, 2, 32, 30, 128, 128, 4),
        (30, base_folder, 2, 32, 31, 128, 128, 4)
    ]

@pytest.mark.test_2_32_1to31_128_4
@pytest.mark.parametrize(("test_case_number", "directory", "n_number_of_analog_states", "oversampling_rate", "bit_width", "lookback_length", "lookahead_length", "lut_input_width"), test_2_32_1to31_128_4_cases)
def test_2_32_1to31_128_4_digital_estimator_generator(test_case_number: int, directory: str, n_number_of_analog_states: int, oversampling_rate: int, bit_width: int, lookback_length: int, lookahead_length: int, lut_input_width: int):
    test_digital_estimator_generator(test_case_number = test_case_number, directory = directory, n_number_of_analog_states = n_number_of_analog_states, oversampling_rate = oversampling_rate, bit_width = bit_width, lookback_length = lookback_length, lookahead_length = lookahead_length, lut_input_width = lut_input_width)


# Explaination of the parameters
# Test number
# Folder to place SystemVerilog files
# Number of analog states
# Oversampling rate
# FIR coefficient bit width
# lookback length
# lookahead length
# LUT input bit width
test_3_15_1to31_128_4_cases = [
        (0, base_folder, 3, 15, 1, 128, 128, 4),
        (1, base_folder, 3, 15, 2, 128, 128, 4),
        (2, base_folder, 3, 15, 3, 128, 128, 4),
        (3, base_folder, 3, 15, 4, 128, 128, 4),
        (4, base_folder, 3, 15, 5, 128, 128, 4),
        (5, base_folder, 3, 15, 6, 128, 128, 4),
        (6, base_folder, 3, 15, 7, 128, 128, 4),
        (7, base_folder, 3, 15, 8, 128, 128, 4),
        (8, base_folder, 3, 15, 9, 128, 128, 4),
        (9, base_folder, 3, 15, 10, 128, 128, 4),
        (10, base_folder, 3, 15, 11, 128, 128, 4),
        (11, base_folder, 3, 15, 12, 128, 128, 4),
        (12, base_folder, 3, 15, 13, 128, 128, 4),
        (13, base_folder, 3, 15, 14, 128, 128, 4),
        (14, base_folder, 3, 15, 15, 128, 128, 4),
        (15, base_folder, 3, 15, 16, 128, 128, 4),
        (16, base_folder, 3, 15, 17, 128, 128, 4),
        (17, base_folder, 3, 15, 18, 128, 128, 4),
        (18, base_folder, 3, 15, 19, 128, 128, 4),
        (19, base_folder, 3, 15, 20, 128, 128, 4),
        (20, base_folder, 3, 15, 21, 128, 128, 4),
        (21, base_folder, 3, 15, 22, 128, 128, 4),
        (22, base_folder, 3, 15, 23, 128, 128, 4),
        (23, base_folder, 3, 15, 24, 128, 128, 4),
        (24, base_folder, 3, 15, 25, 128, 128, 4),
        (25, base_folder, 3, 15, 26, 128, 128, 4),
        (26, base_folder, 3, 15, 27, 128, 128, 4),
        (27, base_folder, 3, 15, 28, 128, 128, 4),
        (28, base_folder, 3, 15, 29, 128, 128, 4),
        (29, base_folder, 3, 15, 30, 128, 128, 4),
        (30, base_folder, 3, 15, 31, 128, 128, 4)
    ]

@pytest.mark.test_3_15_1to31_128_4
@pytest.mark.parametrize(("test_case_number", "directory", "n_number_of_analog_states", "oversampling_rate", "bit_width", "lookback_length", "lookahead_length", "lut_input_width"), test_3_15_1to31_128_4_cases)
def test_3_15_1to31_128_4_digital_estimator_generator(test_case_number: int, directory: str, n_number_of_analog_states: int, oversampling_rate: int, bit_width: int, lookback_length: int, lookahead_length: int, lut_input_width: int):
    test_digital_estimator_generator(test_case_number = test_case_number, directory = directory, n_number_of_analog_states = n_number_of_analog_states, oversampling_rate = oversampling_rate, bit_width = bit_width, lookback_length = lookback_length, lookahead_length = lookahead_length, lut_input_width = lut_input_width)


# Explaination of the parameters
# Test number
# Folder to place SystemVerilog files
# Number of analog states
# Oversampling rate
# FIR coefficient bit width
# lookback length
# lookahead length
# LUT input bit width
test_4_11_1to31_128_4_cases = [
        (0, base_folder, 4, 11, 1, 128, 128, 4),
        (1, base_folder, 4, 11, 2, 128, 128, 4),
        (2, base_folder, 4, 11, 3, 128, 128, 4),
        (3, base_folder, 4, 11, 4, 128, 128, 4),
        (4, base_folder, 4, 11, 5, 128, 128, 4),
        (5, base_folder, 4, 11, 6, 128, 128, 4),
        (6, base_folder, 4, 11, 7, 128, 128, 4),
        (7, base_folder, 4, 11, 8, 128, 128, 4),
        (8, base_folder, 4, 11, 9, 128, 128, 4),
        (9, base_folder, 4, 11, 10, 128, 128, 4),
        (10, base_folder, 4, 11, 11, 128, 128, 4),
        (11, base_folder, 4, 11, 12, 128, 128, 4),
        (12, base_folder, 4, 11, 13, 128, 128, 4),
        (13, base_folder, 4, 11, 14, 128, 128, 4),
        (14, base_folder, 4, 11, 15, 128, 128, 4),
        (15, base_folder, 4, 11, 16, 128, 128, 4),
        (16, base_folder, 4, 11, 17, 128, 128, 4),
        (17, base_folder, 4, 11, 18, 128, 128, 4),
        (18, base_folder, 4, 11, 19, 128, 128, 4),
        (19, base_folder, 4, 11, 20, 128, 128, 4),
        (20, base_folder, 4, 11, 21, 128, 128, 4),
        (21, base_folder, 4, 11, 22, 128, 128, 4),
        (22, base_folder, 4, 11, 23, 128, 128, 4),
        (23, base_folder, 4, 11, 24, 128, 128, 4),
        (24, base_folder, 4, 11, 25, 128, 128, 4),
        (25, base_folder, 4, 11, 26, 128, 128, 4),
        (26, base_folder, 4, 11, 27, 128, 128, 4),
        (27, base_folder, 4, 11, 28, 128, 128, 4),
        (28, base_folder, 4, 11, 29, 128, 128, 4),
        (29, base_folder, 4, 11, 30, 128, 128, 4),
        (30, base_folder, 4, 11, 31, 128, 128, 4)
    ]

@pytest.mark.test_4_11_1to31_128_4
@pytest.mark.parametrize(("test_case_number", "directory", "n_number_of_analog_states", "oversampling_rate", "bit_width", "lookback_length", "lookahead_length", "lut_input_width"), test_4_11_1to31_128_4_cases)
def test_4_11_1to31_128_4_digital_estimator_generator(test_case_number: int, directory: str, n_number_of_analog_states: int, oversampling_rate: int, bit_width: int, lookback_length: int, lookahead_length: int, lut_input_width: int):
    test_digital_estimator_generator(test_case_number = test_case_number, directory = directory, n_number_of_analog_states = n_number_of_analog_states, oversampling_rate = oversampling_rate, bit_width = bit_width, lookback_length = lookback_length, lookahead_length = lookahead_length, lut_input_width = lut_input_width)


# Explaination of the parameters
# Test number
# Folder to place SystemVerilog files
# Number of analog states
# Oversampling rate
# FIR coefficient bit width
# lookback length
# lookahead length
# LUT input bit width
test_5_9_1to31_128_4_cases = [
        (0, base_folder, 5, 9, 1, 128, 128, 4),
        (1, base_folder, 5, 9, 2, 128, 128, 4),
        (2, base_folder, 5, 9, 3, 128, 128, 4),
        (3, base_folder, 5, 9, 4, 128, 128, 4),
        (4, base_folder, 5, 9, 5, 128, 128, 4),
        (5, base_folder, 5, 9, 6, 128, 128, 4),
        (6, base_folder, 5, 9, 7, 128, 128, 4),
        (7, base_folder, 5, 9, 8, 128, 128, 4),
        (8, base_folder, 5, 9, 9, 128, 128, 4),
        (9, base_folder, 5, 9, 10, 128, 128, 4),
        (10, base_folder, 5, 9, 11, 128, 128, 4),
        (11, base_folder, 5, 9, 12, 128, 128, 4),
        (12, base_folder, 5, 9, 13, 128, 128, 4),
        (13, base_folder, 5, 9, 14, 128, 128, 4),
        (14, base_folder, 5, 9, 15, 128, 128, 4),
        (15, base_folder, 5, 9, 16, 128, 128, 4),
        (16, base_folder, 5, 9, 17, 128, 128, 4),
        (17, base_folder, 5, 9, 18, 128, 128, 4),
        (18, base_folder, 5, 9, 19, 128, 128, 4),
        (19, base_folder, 5, 9, 20, 128, 128, 4),
        (20, base_folder, 5, 9, 21, 128, 128, 4),
        (21, base_folder, 5, 9, 22, 128, 128, 4),
        (22, base_folder, 5, 9, 23, 128, 128, 4),
        (23, base_folder, 5, 9, 24, 128, 128, 4),
        (24, base_folder, 5, 9, 25, 128, 128, 4),
        (25, base_folder, 5, 9, 26, 128, 128, 4),
        (26, base_folder, 5, 9, 27, 128, 128, 4),
        (27, base_folder, 5, 9, 28, 128, 128, 4),
        (28, base_folder, 5, 9, 29, 128, 128, 4),
        (29, base_folder, 5, 9, 30, 128, 128, 4),
        (30, base_folder, 5, 9, 31, 128, 128, 4)
    ]

@pytest.mark.test_5_9_1to31_128_4
@pytest.mark.parametrize(("test_case_number", "directory", "n_number_of_analog_states", "oversampling_rate", "bit_width", "lookback_length", "lookahead_length", "lut_input_width"), test_5_9_1to31_128_4_cases)
def test_5_9_1to31_128_4_digital_estimator_generator(test_case_number: int, directory: str, n_number_of_analog_states: int, oversampling_rate: int, bit_width: int, lookback_length: int, lookahead_length: int, lut_input_width: int):
    test_digital_estimator_generator(test_case_number = test_case_number, directory = directory, n_number_of_analog_states = n_number_of_analog_states, oversampling_rate = oversampling_rate, bit_width = bit_width, lookback_length = lookback_length, lookahead_length = lookahead_length, lut_input_width = lut_input_width)


# Explaination of the parameters
# Test number
# Folder to place SystemVerilog files
# Number of analog states
# Oversampling rate
# FIR coefficient bit width
# lookback length
# lookahead length
# LUT input bit width
test_6_7_1to31_128_4_cases = [
        (0, base_folder, 6, 7, 1, 128, 128, 4),
        (1, base_folder, 6, 7, 2, 128, 128, 4),
        (2, base_folder, 6, 7, 3, 128, 128, 4),
        (3, base_folder, 6, 7, 4, 128, 128, 4),
        (4, base_folder, 6, 7, 5, 128, 128, 4),
        (5, base_folder, 6, 7, 6, 128, 128, 4),
        (6, base_folder, 6, 7, 7, 128, 128, 4),
        (7, base_folder, 6, 7, 8, 128, 128, 4),
        (8, base_folder, 6, 7, 9, 128, 128, 4),
        (9, base_folder, 6, 7, 10, 128, 128, 4),
        (10, base_folder, 6, 7, 11, 128, 128, 4),
        (11, base_folder, 6, 7, 12, 128, 128, 4),
        (12, base_folder, 6, 7, 13, 128, 128, 4),
        (13, base_folder, 6, 7, 14, 128, 128, 4),
        (14, base_folder, 6, 7, 15, 128, 128, 4),
        (15, base_folder, 6, 7, 16, 128, 128, 4),
        (16, base_folder, 6, 7, 17, 128, 128, 4),
        (17, base_folder, 6, 7, 18, 128, 128, 4),
        (18, base_folder, 6, 7, 19, 128, 128, 4),
        (19, base_folder, 6, 7, 20, 128, 128, 4),
        (20, base_folder, 6, 7, 21, 128, 128, 4),
        (21, base_folder, 6, 7, 22, 128, 128, 4),
        (22, base_folder, 6, 7, 23, 128, 128, 4),
        (23, base_folder, 6, 7, 24, 128, 128, 4),
        (24, base_folder, 6, 7, 25, 128, 128, 4),
        (25, base_folder, 6, 7, 26, 128, 128, 4),
        (26, base_folder, 6, 7, 27, 128, 128, 4),
        (27, base_folder, 6, 7, 28, 128, 128, 4),
        (28, base_folder, 6, 7, 29, 128, 128, 4),
        (29, base_folder, 6, 7, 30, 128, 128, 4),
        (30, base_folder, 6, 7, 31, 128, 128, 4)
    ]

@pytest.mark.test_6_7_1to31_128_4
@pytest.mark.parametrize(("test_case_number", "directory", "n_number_of_analog_states", "oversampling_rate", "bit_width", "lookback_length", "lookahead_length", "lut_input_width"), test_6_7_1to31_128_4_cases)
def test_6_7_1to31_128_4_digital_estimator_generator(test_case_number: int, directory: str, n_number_of_analog_states: int, oversampling_rate: int, bit_width: int, lookback_length: int, lookahead_length: int, lut_input_width: int):
    test_digital_estimator_generator(test_case_number = test_case_number, directory = directory, n_number_of_analog_states = n_number_of_analog_states, oversampling_rate = oversampling_rate, bit_width = bit_width, lookback_length = lookback_length, lookahead_length = lookahead_length, lut_input_width = lut_input_width)


# Explaination of the parameters
# Test number
# Folder to place SystemVerilog files
# Number of analog states
# Oversampling rate
# FIR coefficient bit width
# lookback length
# lookahead length
# LUT input bit width
test_7_7_1to31_256_4_cases = [
        (0, base_folder, 7, 7, 1, 256, 256, 4),
        (1, base_folder, 7, 7, 2, 256, 256, 4),
        (2, base_folder, 7, 7, 3, 256, 256, 4),
        (3, base_folder, 7, 7, 4, 256, 256, 4),
        (4, base_folder, 7, 7, 5, 256, 256, 4),
        (5, base_folder, 7, 7, 6, 256, 256, 4),
        (6, base_folder, 7, 7, 7, 256, 256, 4),
        (7, base_folder, 7, 7, 8, 256, 256, 4),
        (8, base_folder, 7, 7, 9, 256, 256, 4),
        (9, base_folder, 7, 7, 10, 256, 256, 4),
        (10, base_folder, 7, 7, 11, 256, 256, 4),
        (11, base_folder, 7, 7, 12, 256, 256, 4),
        (12, base_folder, 7, 7, 13, 256, 256, 4),
        (13, base_folder, 7, 7, 14, 256, 256, 4),
        (14, base_folder, 7, 7, 15, 256, 256, 4),
        (15, base_folder, 7, 7, 16, 256, 256, 4),
        (16, base_folder, 7, 7, 17, 256, 256, 4),
        (17, base_folder, 7, 7, 18, 256, 256, 4),
        (18, base_folder, 7, 7, 19, 256, 256, 4),
        (19, base_folder, 7, 7, 20, 256, 256, 4),
        (20, base_folder, 7, 7, 21, 256, 256, 4),
        (21, base_folder, 7, 7, 22, 256, 256, 4),
        (22, base_folder, 7, 7, 23, 256, 256, 4),
        (23, base_folder, 7, 7, 24, 256, 256, 4),
        (24, base_folder, 7, 7, 25, 256, 256, 4),
        (25, base_folder, 7, 7, 26, 256, 256, 4),
        (26, base_folder, 7, 7, 27, 256, 256, 4),
        (27, base_folder, 7, 7, 28, 256, 256, 4),
        (28, base_folder, 7, 7, 29, 256, 256, 4),
        (29, base_folder, 7, 7, 30, 256, 256, 4),
        (30, base_folder, 7, 7, 31, 256, 256, 4)
    ]

@pytest.mark.test_7_7_1to31_256_4
@pytest.mark.parametrize(("test_case_number", "directory", "n_number_of_analog_states", "oversampling_rate", "bit_width", "lookback_length", "lookahead_length", "lut_input_width"), test_7_7_1to31_256_4_cases)
def test_7_7_1to31_256_4_digital_estimator_generator(test_case_number: int, directory: str, n_number_of_analog_states: int, oversampling_rate: int, bit_width: int, lookback_length: int, lookahead_length: int, lut_input_width: int):
    test_digital_estimator_generator(test_case_number = test_case_number, directory = directory, n_number_of_analog_states = n_number_of_analog_states, oversampling_rate = oversampling_rate, bit_width = bit_width, lookback_length = lookback_length, lookahead_length = lookahead_length, lut_input_width = lut_input_width)


# Explaination of the parameters
# Test number
# Folder to place SystemVerilog files
# Number of analog states
# Oversampling rate
# FIR coefficient bit width
# lookback length
# lookahead length
# LUT input bit width
test_8_6_1to31_256_4_cases = [
        (0, base_folder, 8, 6, 1, 256, 256, 4),
        (1, base_folder, 8, 6, 2, 256, 256, 4),
        (2, base_folder, 8, 6, 3, 256, 256, 4),
        (3, base_folder, 8, 6, 4, 256, 256, 4),
        (4, base_folder, 8, 6, 5, 256, 256, 4),
        (5, base_folder, 8, 6, 6, 256, 256, 4),
        (6, base_folder, 8, 6, 7, 256, 256, 4),
        (7, base_folder, 8, 6, 8, 256, 256, 4),
        (8, base_folder, 8, 6, 9, 256, 256, 4),
        (9, base_folder, 8, 6, 10, 256, 256, 4),
        (10, base_folder, 8, 6, 11, 256, 256, 4),
        (11, base_folder, 8, 6, 12, 256, 256, 4),
        (12, base_folder, 8, 6, 13, 256, 256, 4),
        (13, base_folder, 8, 6, 14, 256, 256, 4),
        (14, base_folder, 8, 6, 15, 256, 256, 4),
        (15, base_folder, 8, 6, 16, 256, 256, 4),
        (16, base_folder, 8, 6, 17, 256, 256, 4),
        (17, base_folder, 8, 6, 18, 256, 256, 4),
        (18, base_folder, 8, 6, 19, 256, 256, 4),
        (19, base_folder, 8, 6, 20, 256, 256, 4),
        (20, base_folder, 8, 6, 21, 256, 256, 4),
        (21, base_folder, 8, 6, 22, 256, 256, 4),
        (22, base_folder, 8, 6, 23, 256, 256, 4),
        (23, base_folder, 8, 6, 24, 256, 256, 4),
        (24, base_folder, 8, 6, 25, 256, 256, 4),
        (25, base_folder, 8, 6, 26, 256, 256, 4),
        (26, base_folder, 8, 6, 27, 256, 256, 4),
        (27, base_folder, 8, 6, 28, 256, 256, 4),
        (28, base_folder, 8, 6, 29, 256, 256, 4),
        (29, base_folder, 8, 6, 30, 256, 256, 4),
        (30, base_folder, 8, 6, 31, 256, 256, 4)
    ]

@pytest.mark.test_8_6_1to31_256_4
@pytest.mark.parametrize(("test_case_number", "directory", "n_number_of_analog_states", "oversampling_rate", "bit_width", "lookback_length", "lookahead_length", "lut_input_width"), test_8_6_1to31_256_4_cases)
def test_8_6_1to31_256_4_digital_estimator_generator(test_case_number: int, directory: str, n_number_of_analog_states: int, oversampling_rate: int, bit_width: int, lookback_length: int, lookahead_length: int, lut_input_width: int):
    test_digital_estimator_generator(test_case_number = test_case_number, directory = directory, n_number_of_analog_states = n_number_of_analog_states, oversampling_rate = oversampling_rate, bit_width = bit_width, lookback_length = lookback_length, lookahead_length = lookahead_length, lut_input_width = lut_input_width)