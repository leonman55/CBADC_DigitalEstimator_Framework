from asyncio.subprocess import PIPE
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
        path = "../df/sim/"
    elif platform.system() == "Windows":
        path = "..\\df\\sim\\"
        #path = ".\\GeneratedSystemVerilogFiles\\"

    #name = "DigitalEstimatorWrapper"
    #digital_estimator_wrapper: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorModules.DigitalEstimatorWrapper.DigitalEstimatorWrapper(path, name)
    #digital_estimator_wrapper.generate()

    name = "DigitalEstimatorTestbench"
    digital_estimator_testbench: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorVerificationModules.DigitalEstimatorTestbench.DigitalEstimatorTestbench(path, name)
    digital_estimator_testbench.generate()

    #sim_xrun: subprocess.CompletedProcess = subprocess.run(path + "sim.sh", shell = True)
    sim_xrun = subprocess.Popen([path + "sim.sh"], stdout = PIPE, text = True, shell = True)
    sim_xrun.wait()
    while True:
        line: str = sim_xrun.stdout.readline().removesuffix("\n")
        if line == "":
            break
        else:
            print(line)


if __name__ == '__main__':
    main()
