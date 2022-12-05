import pytest
from pathlib import Path

import DigitalEstimatorGenerator


base_folder: str = "/df/sim/TestCase"


# Explaination of the parameters
# Test number
# Folder to place SystemVerilog files
# Number of analog states
# Oversampling rate
# FIR coefficient bit width
# lookback length
# lookahead length
# LUT input bit width
def test_digital_estimator_generator(test_case_number: int, directory: str, n_number_of_analog_states: int, oversampling_rate: int, bit_width: int, lookback_length: int, lookahead_length: int, lut_input_width: int):
    current_working_directory = Path.cwd()
    directory = str(current_working_directory.parent) + directory + "_" + str(n_number_of_analog_states) + "_" + str(oversampling_rate) + "_" + str(bit_width) + "_" + str(lookback_length) + "_" + str(lookahead_length) + "_" + str(lut_input_width)
    Path(directory).mkdir(parents = True, exist_ok = True)
    assert Path.exists(Path(directory))
    digital_estimator_generator: DigitalEstimatorGenerator.DigitalEstimatorGenerator = DigitalEstimatorGenerator.DigitalEstimatorGenerator()
    digital_estimator_generator.path = directory
    digital_estimator_generator.configuration_n_number_of_analog_states = n_number_of_analog_states
    digital_estimator_generator.configuration_over_sample_rate = oversampling_rate
    digital_estimator_generator.configuration_fir_data_width = bit_width
    digital_estimator_generator.configuration_lookback_length = lookback_length
    digital_estimator_generator.configuration_lookahead_length = lookahead_length
    digital_estimator_generator.configuration_fir_lut_input_width = lut_input_width
    
    digital_estimator_generator.generate()
    error: int = 0
    error_message: str = ""
    error, error_message = digital_estimator_generator.simulate()
    if error:
        pytest.fail(reason = error_message)



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