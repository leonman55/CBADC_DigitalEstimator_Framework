import os
import numpy as np
import scipy.signal
import cbadc
import calib_python

os.chdir("/local_work/leonma/calib_filter_generation/")
cwd: str = os.getcwd()
print("Current working directory: ", cwd, "\n")

bandwidth: float = 0.1
nyquist_frequency_samples: float = 1.0
number_of_digital_control_signals_m: int = 8
filter_taps_k: int = 512
kappa_scale_k0s: float = 0.1

cbadc_digital_estimation_filter_initial = cbadc.digital_estimator.initial_filter(
    [
        -scipy.signal.firwin2(
            int(filter_taps_k),
            np.array([0.0, 0.99 * float(bandwidth), 1.01 * float(bandwidth), float(nyquist_frequency_samples)]),
            np.array([1.0, 1.0, 0.0, 0.0]),
            )
        * float(kappa_scale_k0s)
    ],
    [int(filter_taps_k) for _ in range(int(number_of_digital_control_signals_m))],
    [0],
)

np.save("filter.npy", cbadc_digital_estimation_filter_initial)

filter = np.load("filter.npy")
print(filter)