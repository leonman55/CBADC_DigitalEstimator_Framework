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


def main():
    path: str = ""
    if platform.system() == "Linux":
        #path = "./GeneratedSystemVerilogFiles/"
        path = "../df/sim/SystemVerilogFiles/"
        sim_folder = "../df/sim/"
    elif platform.system() == "Windows":
        #path = ".\\GeneratedSystemVerilogFiles\\"
        path = "..\\df\\sim\\SystemVerilogFiles\\"
        sim_folder = "..\\df\\sim\\"

    #name = "DigitalEstimatorWrapper"
    #digital_estimator_wrapper: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorModules.DigitalEstimatorWrapper.DigitalEstimatorWrapper(path, name)
    #digital_estimator_wrapper.generate()

    name = "DigitalEstimatorTestbench"
    digital_estimator_testbench: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorVerificationModules.DigitalEstimatorTestbench.DigitalEstimatorTestbench(path, name)
    digital_estimator_testbench.generate()

    options: list[str] = list[str]()
    options.append("-top " + digital_estimator_testbench.name)
    write_options_file(path, "xrun_options", options)
    #sim_xrun: subprocess.CompletedProcess = subprocess.run(sim_folder + "sim.sh", shell = True)
    sim_xrun = subprocess.Popen([sim_folder + "sim.sh"], stdout = PIPE, text = True, shell = True)
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
    if pass_count == digital_estimator_testbench.syntax_generator.assertion_count:
        print(f"All {digital_estimator_testbench.syntax_generator.assertion_count} assertions met!")
    else:
        print(f"{pass_count} out of {digital_estimator_testbench.syntax_generator.assertion_count} assertions were met.")
        print(f"{fail_count} assertions failed!")


def write_options_file(path: str, name: str, options: list[str]):
    options_file: FileGenerator.FileGenerator = FileGenerator.FileGenerator()
    options_file.set_path(path)
    options_file.set_name(name)
    options_file.open_output_file()
    for line in options:
        options_file.write_line_linebreak(line)
    options_file.close_output_file()


if __name__ == '__main__':
    main()
