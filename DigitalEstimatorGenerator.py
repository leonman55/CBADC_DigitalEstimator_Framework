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


def main():
    path: str = ""
    if platform.system() == "Linux":
        path = "./GeneratedSystemVerilogFiles/"
    elif platform.system() == "Windows":
        path = ".\\GeneratedSystemVerilogFiles\\"

    name = "DigitalEstimatorWrapper"
    digital_estimator_wrapper: SystemVerilogModule.SystemVerilogModule = DigitalEstimatorModules.DigitalEstimatorWrapper.DigitalEstimatorWrapper(path, name)
    digital_estimator_wrapper.generate()


if __name__ == '__main__':
    main()
