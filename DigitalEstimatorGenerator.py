import os
import platform

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
        path = "../df/src/"
    elif platform.system() == "Windows":
        path = "..\\df\\src\\"
        #path = ".\\GeneratedSystemVerilogFiles\\"

    #name = "DigitalEstimatorWrapper"
    #digital_estimator_wrapper: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorModules.DigitalEstimatorWrapper.DigitalEstimatorWrapper(path, name)
    #digital_estimator_wrapper.generate()

    name = "DigitalEstimatorTestbench"
    digital_estimator_testbench: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorVerificationModules.DigitalEstimatorTestbench.DigitalEstimatorTestbench(path, name)
    digital_estimator_testbench.generate()


if __name__ == '__main__':
    main()
