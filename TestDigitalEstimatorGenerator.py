import pytest
from pathlib import Path

import DigitalEstimatorGenerator


base_folder: str = "/df/sim/TestCase"

# Explaination of the parameters
# Folder to place SystemVerilog files
#
test_cases = [
        (0, base_folder, 16),
        (1, base_folder, 31)
    ]


@pytest.mark.parametrize(("test_case_number", "directory", "bit_width"), test_cases)
def test_digital_estimator_generator(test_case_number: int, directory: str, bit_width: int):
    current_working_directory = Path.cwd()
    directory = str(current_working_directory.parent) + directory + str(test_case_number)
    Path(directory).mkdir(parents = True, exist_ok = True)
    assert Path.exists(Path(directory))
    digital_estimator_generator: DigitalEstimatorGenerator.DigitalEstimatorGenerator = DigitalEstimatorGenerator.DigitalEstimatorGenerator()
    digital_estimator_generator.path = directory
    digital_estimator_generator.configuration_fir_data_width = bit_width
    digital_estimator_generator.generate()
