from math import ceil

import cbadc
import matplotlib.mlab
import matplotlib.pyplot as plt
import numpy as np


def convert_coefficient_matrix_to_lut_entries(coefficient_matrix: np.ndarray, lut_input_width: int) -> np.ndarray:
    """Converts the FIR coefficient matrix given by the cbadc Python package
        into the LUT entries.

    The cbadc package creates the coefficients for the single elements of
        the digital control vectors. When using a FIR filter with LUTs,
        the LUTs can have an input width greater than 1. This leads to the
        need of combined coefficients.
    """
    total_number_of_elements: int = 1
    for dimension_index in range(len(coefficient_matrix.shape)):
        total_number_of_elements *= coefficient_matrix.shape[dimension_index]
    print("Total number of elements: ", total_number_of_elements)
    coefficient_matrix_flattened = coefficient_matrix.reshape(total_number_of_elements)
    lut_entry_array: list[int] = list[int]()
    for coefficient_index in range(ceil(total_number_of_elements / lut_input_width) - 1):
        for lut_entry_index in range(2**lut_input_width):
            entry: int = 0
            for shift_offset in range(lut_input_width):
                if lut_entry_index & (0b1 << shift_offset):
                    entry += coefficient_matrix_flattened[coefficient_index * lut_input_width + shift_offset]
                else:
                    entry -= coefficient_matrix_flattened[coefficient_index * lut_input_width + shift_offset]
            #lut_entry_array.append(entry)
            lut_entry_array.insert(0, entry)
    last_lut_width: int = total_number_of_elements % lut_input_width
    if last_lut_width == 0:
        last_lut_width = lut_input_width
    coefficient_index: int = ceil(total_number_of_elements / lut_input_width) - 1
    for lut_entry_index in range(2**last_lut_width):
        entry: int = 0
        for shift_offset in range(last_lut_width):
            if lut_entry_index & (0b1 << shift_offset):
                entry += coefficient_matrix_flattened[coefficient_index * lut_input_width + shift_offset]
            else:
                entry -= coefficient_matrix_flattened[coefficient_index * lut_input_width + shift_offset]
        #lut_entry_array.append(entry)
        lut_entry_array.insert(0, entry)
    print("LUT entry count:", len(lut_entry_array))
    print("LUT entry list:\n", lut_entry_array)
    return lut_entry_array


def plot_results(path: str = "../df/sim/SystemVerilogFiles", k1: int = 512, k2: int = 512, size: int = 1 << 14, T: float = 2.0e-8, OSR: int = 1, down_sample_rate: int = 1):
    """Plots the results of the cbadc Python package simulation and the RTL simulation.
    
    This function plots the following graphs:
        The cbadc high level estimation.
        The RTL estimation.
        cbadc and RTL estimation combined.
        The difference between the cbadc and RTL estimation.
        The PSD graphs of the cbadc and RTL estimations.
    Additionally the SNR of the RTL estimation will be calculated from the PSD.
    """
    digital_estimation_results: list[float] = list[float]()
    try:
        with open(path + "/digital_estimation.csv", "r") as system_verilog_simulation_csv_file:
            for line in system_verilog_simulation_csv_file.readlines():
                digital_estimation_results.append(float(line.rsplit(",")[0]))
            system_verilog_simulation_csv_file.close()
            plt.xlabel("sample number")
            plt.ylabel("signal amplitude")
            plt.plot(digital_estimation_results, linewidth = 0.25)
            plt.savefig(path + "/digital_estimation.pdf")
            plt.clf()
    except:
        print("No RTL simulation results available!")
    digital_estimation_high_level_results: list[float] = list[float]()
    try:
        with open(path + "/digital_estimation_high_level.csv", "r") as high_level_simulation_csv_file:
            for line in high_level_simulation_csv_file.readlines():
                digital_estimation_high_level_results.append(float(line.rsplit(",")[0]))
            high_level_simulation_csv_file.close()
            plt.xlabel("sample number")
            plt.ylabel("signal amplitude")
            plt.plot(digital_estimation_high_level_results, linewidth = 0.25)
            plt.savefig(path + "/digital_estimation_high_level.pdf")
            plt.clf()
    except:
        print("No cbadc Python simulation results available!")
    if len(digital_estimation_results) != 0 and len(digital_estimation_high_level_results) != 0:
        plt.xlabel("sample number")
        plt.ylabel("signal amplitude")
        plt.plot(digital_estimation_results, linewidth = 0.25)
        plt.plot(digital_estimation_high_level_results, linewidth = 0.25)
        plt.savefig(path + "/digital_estimation_system_verilog_&_high_level.pdf")
        plt.clf()
    try:
        with open(path + "/digital_estimation_high_level_self_programmed_integer.csv", "r") as high_level_simulation_self_programmed_integer_csv_file:
            digital_estimation_high_level_self_programmed_results_integer: list[int] = list[int]()
            for line in high_level_simulation_self_programmed_integer_csv_file.readlines():
                digital_estimation_high_level_self_programmed_results_integer.append(float(line.rsplit(",")[0]))
            high_level_simulation_self_programmed_integer_csv_file.close()
            plt.xlabel("sample number")
            plt.ylabel("signal amplitude")
            plt.plot(digital_estimation_high_level_self_programmed_results_integer, linewidth = 0.25)
            plt.savefig(path + "/digital_estimation_high_level_self_programmed_integer.pdf")
            plt.clf()
    except:
        pass
    try:
        with open(path + "/digital_estimation_high_level_self_programmed_float.csv", "r") as high_level_simulation_self_programmed_float_csv_file:
            digital_estimation_high_level_self_programmed_results_float: list[float] = list[float]()
            for line in high_level_simulation_self_programmed_float_csv_file.readlines():
                digital_estimation_high_level_self_programmed_results_float.append(float(line.rsplit(",")[0]))
            high_level_simulation_self_programmed_float_csv_file.close()
            plt.xlabel("sample number")
            plt.ylabel("signal amplitude")
            plt.plot(digital_estimation_high_level_self_programmed_results_float, linewidth = 0.25)
            plt.savefig(path + "/digital_estimation_high_level_self_programmed_float.pdf")
            plt.clf()
    except:
        pass
    try:
        with open(path + "/digital_estimation_snr_db.csv", "w") as digital_estimation_snr_db_csv_file:
            """digital_estimation_fft = np.fft.rfft(digital_estimation_results[-(size >> 1) : ], n = size >> 1)
            plt.xscale("log")
            plt.plot(digital_estimation_fft.real)
            plt.plot(digital_estimation_fft.imag)
            plt.savefig(path + "/digital_estimation_fft.pdf")
            plt.clf()
            digital_estimation_fft_abs = np.sqrt(digital_estimation_fft.real ** 2 + digital_estimation_fft.imag ** 2)
            plt.xscale("log")
            plt.plot(digital_estimation_fft_abs)
            plt.savefig(path + "/digital_estimation_fft_abs.pdf")
            plt.clf()
            digital_estimation_fft_abs_dB = 20 * np.log10(digital_estimation_fft_abs)
            plt.xscale("log")
            plt.plot(digital_estimation_fft_abs_dB)
            plt.savefig(path + "/digital_estimation_fft_abs_dB.pdf")
            plt.clf()
            maximum = np.max(digital_estimation_fft_abs)
            print(maximum)
            maximum_index = np.where(digital_estimation_fft_abs == maximum)
            print(maximum_index[0][0])
            noise = digital_estimation_fft_abs[np.where(digital_estimation_fft_abs != maximum)]
            noise_sum = np.sum(np.abs(noise))
            snr = maximum / noise_sum
            snr_dB = 20 * np.log10(snr)
            print(snr, "\tdB: ", snr_dB)
            digital_estimation_fft_abs_psd = digital_estimation_fft_abs**2
            plt.xscale("log")
            plt.plot(digital_estimation_fft_abs_psd)
            plt.savefig(path + "/digital_estimation_fft_abs_psd.pdf")
            plt.clf()"""

            plt.xscale("log")
            #plt.psd(digital_estimation_results[k1 + k2 : ], Fs = (1.0 / T), window = matplotlib.mlab.window_none)
            digital_estimation_results_psd = plt.psd(digital_estimation_results[-(size >> 1) : ], Fs = (1.0 / (T * down_sample_rate)), window = matplotlib.mlab.window_none, NFFT = size >> 1)
            maximum = np.max(digital_estimation_results_psd[0])
            noise = digital_estimation_results_psd[0][np.where(digital_estimation_results_psd[0] != maximum)]
            noise_sum = np.sum(noise)
            snr = maximum / noise_sum
            snr_dB = 10 * np.log10(snr)
            print("\tSNR dB: ", snr_dB)
            digital_estimation_snr_db_csv_file.write(str(snr_dB))
            digital_estimation_snr_db_csv_file.close()
            #plt.psd(digital_estimation_high_level_results[k1 + k2 : ], Fs = (1.0 / T), window = matplotlib.mlab.window_none)
            plt.psd(digital_estimation_high_level_results[-(size >> 1) : ], Fs = (1.0 / (T * down_sample_rate)), window = matplotlib.mlab.window_none, NFFT = size >> 1)
            plt.savefig(path + "/psd_log.pdf")
            plt.clf()
            # Maybe skip windows, make only 1 bin
            # Find signal in psd, calculate SNR
            #plt.psd(digital_estimation_results[k1 + k2 : ], Fs = (1.0 / T), window = matplotlib.mlab.window_none)
            plt.psd(digital_estimation_results[-(size >> 1) : ], Fs = (1.0 / (T * down_sample_rate)), window = matplotlib.mlab.window_none, NFFT = size >> 1)
            #plt.psd(digital_estimation_high_level_results[k1 + k2 : ], Fs = (1.0 / T), window = matplotlib.mlab.window_none)
            plt.psd(digital_estimation_high_level_results[-(size >> 1) : ], Fs = (1.0 / (T * down_sample_rate)), window = matplotlib.mlab.window_none, NFFT = size >> 1)
            plt.savefig(path + "/psd_linear.pdf")
            plt.clf()
    except:
        print("No RTL simulation PSD information available!")
    try:
        with open(path + "/digital_estimation_system_verilog_vs_high_level.csv") as digital_estimation_system_verilog_vs_high_level_csv_file:
            digital_estimation_system_verilog_vs_high_level_results: list[float] = list[float]()
            for line in digital_estimation_system_verilog_vs_high_level_csv_file.readlines():
                digital_estimation_system_verilog_vs_high_level_results.append(float(line.rsplit(",")[0]))
            plt.xlabel("sample number")
            plt.ylabel("signal difference")
            plt.plot(digital_estimation_system_verilog_vs_high_level_results, linewidth = 0.25)
            plt.savefig(path + "/digital_estimation_system_verilog_vs_high_level.pdf")
            plt.clf()
    except:
        pass


class DigitalEstimatorParameterGenerator():
    """This class is used to instantiate simulators from the cbadc python package,
        create the wanted input stimuli, calculate the filter parameters and
        plot the results to files.

    First the class should be parametrized according to the wanted system
        specifications and the wanted input stimuli for the simulation.
    Then the class can be used to simulate the analog system and generate
        a sequence of digital control vectors for digital estimation simulation.
    Furthermore the digital estimator coefficients can be generated.
    Also the digital estimator simulation is done with this class.
    After the simulations are done, the results can be plotted to files.

    Attributes
    ----------
        path: str
            The path, where the simulation outputs and plots are saved
        n_number_of_analog_states: int
            The number N of analog states of the analog system
        m_number_of_digital_states: int
            The number M of digital states of the digital control
        eta2: float
            -
        k1: int
            Lookback batch size.
        k2: int
            Lookahead batch size.
        data_width: int
            Data width of the filter coefficients, the LUTs, adders, etc.
        down_sample_rate: int
            Downsample rate of the digital estimator.
        T: float
            Sampling time of the digital control module. Calculated from the
                chosen bandwidth and the oversampling rate.
        amplitude: float
            Amplitude of the input signal.
        OSR: int
            The oversampling rate of the digital control unit.
        phase: float
            Phase of the chosen input signal.
        offset: float
            Offset of the chosen input signal
        size: int
            Size/length of the input stimulation signal.
        bandwidth: float = 1e6
            The bandwidth of the analog system
        fir_h_matrix: np.ndarray
            The FIR coefficient matrix as returned by the cbadc simulator
        fir_hb_matrix: np.ndarray
            The lookback part of the coefficient matrix
        fir_hf_matrix: np.ndarray
            The lookahead part of the coefficient matrix
    """
    path: str = "../df/sim/SystemVerilogFiles"
    # Set the number of analog states
    n_number_of_analog_states: int = 6
    # Set the number of digital states
    m_number_of_digital_states: int = n_number_of_analog_states
    # Set the amplification factor
    #beta: float = 6250.0
    #rho: float = -1e-2
    #kappa: float = -1.0
    eta2 = 1e7
    # Set the batch size of lookback
    #K1 = sequence_length
    k1 = 5
    # Set the batch size of lookahead
    k2 = 1
    # Digital estimator data width
    data_width: int = 64

    down_sample_rate: int = 1

    # Sampling time
    T: float = 0

    # Values for input stimulus generation
    # amplitude of the signal
    amplitude = 0.5
    # Choose the sinusoidal frequency via an oversampling ratio (OSR).
    OSR = 25
    # We also specify a phase an offset these are hovewer optional.
    phase = np.pi / 3
    offset = 0.0

    # Length of the simulation
    size: int = 1 << 12

    # Set the bandwidth of the estimator
    bandwidth = 1e6

    # FIR filter coefficients
    fir_h_matrix: np.ndarray
    fir_hb_matrix: np.ndarray
    fir_hf_matrix: np.ndarray

    def __init__(self,
            path: str = "../df/sim/SystemVerilogFiles",
            n_number_of_analog_states: int = 6,
            m_number_of_digital_states: int = n_number_of_analog_states,
            #beta: float = 6250.0,
            #rho: float = -1e-2,
            #kappa: float = -1.0,
            eta2: float = 1e7,
            k1: int = 5,
            k2: int = 1,
            data_width: int = 64,
            down_sample_rate: int = 1,
            amplitude: float = 0.7,
            OSR: int = 25,
            phase: float = np.pi / 3.0,
            offset: float = 0.0,
            size: int = 1 << 12,
            bandwidth: int = 1e6) -> None:
        self.path = path
        self.n_number_of_analog_states = n_number_of_analog_states
        self.m_number_of_digital_states = m_number_of_digital_states
        #self.beta = beta
        #self.rho = rho
        #self.kappa = kappa
        self.eta2 = eta2
        self.k1 = k1
        self.k2 = k2
        self.data_width = data_width
        self.down_sample_rate = down_sample_rate
        self.amplitude = amplitude
        self.OSR = OSR
        self.phase = phase
        self.offset = offset
        self.size = size
        self.bandwidth = bandwidth

    def write_control_signal_to_csv_file(self, values: np.ndarray):
        """Writes the digital control vector stream to a file
        """
        with open(self.path + "/control_signal.csv", "w") as csv_file:
            for single_control_signal in values:
                single_control_signal_string: str = ""
                for bit_index in range(self.m_number_of_digital_states - 1, -1, -1):
                    single_control_signal_string = single_control_signal_string + str(int(single_control_signal[bit_index]))
                csv_file.write(single_control_signal_string + ",\n")
            csv_file.close()

    def write_digital_estimation_fir_to_csv_file(self, values: np.ndarray):
        """Writes the cbadc high level estimation to a file
        """
        np.set_printoptions(floatmode = "fixed", precision = 18)
        with open(self.path + "/digital_estimation_high_level.csv", "w") as csv_file:
            for single_estimation_value in values:
                single_estimation_value_string: str = str(single_estimation_value)
                single_estimation_value_string = single_estimation_value_string.lstrip("[")
                single_estimation_value_string = single_estimation_value_string.lstrip(" ")
                single_estimation_value_string = single_estimation_value_string.rstrip("],")
                csv_file.write(single_estimation_value_string + ",\n")
            csv_file.close()


    def get_eta2(self, AF, BW: int):
        """Calculates the eta2 parameter from a defined analog system
        """
        eta2 = (
        np.linalg.norm(
        AF.analog_system.transfer_function_matrix(
        np.array([2 * np.pi * BW])
        )) ** 2 )
        return eta2

    def get_input_freq(self, sim_len, n_cycles, fs):
        """Calculates an appropriate input frequence for the input stimulation signal.
        
        This is dependant on the chosen simulation length, the wanted complete cycles
            in the stimulation signal and the sample frequence of the digital control.
        """
        samples_per_period = sim_len/n_cycles
        fi = fs / samples_per_period
        return fi

    def simulate_analog_system(self):
        """Simulates the analog system and digital control and saves the digital
            control vector stream to a file.
        """
        # Setup the analog System.
        # We fix the number of analog states.
        #N = 6
        #N = n_number_of_analog_states
        # Set the amplification factor.
        #beta = 6250.0
        #rho = -1e-2
        #kappa = -1.0

        # In this example, each nodes amplification and local feedback will be set
        # identically.
        #betaVec = self.beta * np.ones(self.n_number_of_analog_states)
        #rhoVec = betaVec * self.rho
        #kappaVec = self.kappa * self.beta * np.eye(self.n_number_of_analog_states)
        # Instantiate a chain-of-integrators analog system.
        #analog_system = cbadc.analog_system.ChainOfIntegrators(betaVec, rhoVec, kappaVec)

        # LeapFrog analog system
        #gamma = self.OSR / np.pi
        #omega_3dB = 2 * np.pi * self.eta2
        #beta_vec = gamma * (omega_3dB / 2) * np.ones(self.n_number_of_analog_states)
        #alpha_vec = -(omega_3dB / 2) / gamma * np.ones(self.n_number_of_analog_states - 1)
        #rho_vec = np.zeros(self.n_number_of_analog_states)
        #T = 1.0 / (2.0 * beta_vec[0])
        #self.T = T
        #Gamma = np.diag(-beta_vec)
        #analog_system = cbadc.analog_system.LeapFrog(beta_vec, alpha_vec, rho_vec, Gamma)

        analog_frontend = cbadc.synthesis.get_leap_frog(OSR = self.OSR, N = self.n_number_of_analog_states, BW = self.bandwidth)
        #analog_frontend = cbadc.synthesis.get_leap_frog(ENOB = 15.2, N = self.n_number_of_analog_states, BW = self.bandwidth)
        #over_sample_rate = 5e-7 / analog_frontend.digital_control.clock.T
        eta2 = self.get_eta2(analog_frontend, self.bandwidth)
        self.T = analog_frontend.digital_control.clock.T

        # print the analog system such that we can very it being correctly initalized.
        print(analog_frontend.analog_system, "\n")
        print(analog_frontend.digital_control)


        # Setup the digital control.
        # Set the time period which determines how often the digital control updates.
        #T = 1.0 / (2 * self.beta)
        # Instantiate a corresponding clock.
        #clock = cbadc.analog_signal.Clock(T)
        # Set the number of digital controls to be same as analog states.
        #M = N
        # Initialize the digital control.
        #digital_control = cbadc.digital_control.DigitalControl(clock, self.m_number_of_digital_states)
        # print the digital control to verify proper initialization.

        #clock = cbadc.analog_signal.Clock(T)
        #digital_control = cbadc.digital_control.DigitalControl(clock, self.m_number_of_digital_states)

        #print(digital_control)


        n_cycles = 1<<3
        # Setup the analog stimulation signal
        # Set the peak amplitude.
        #amplitude = 0.5
        # Choose the sinusoidal frequency via an oversampling ratio (OSR).
        #OSR = 1 << 9
        #frequency = 1.0 / (T * self.OSR)
        #frequency = self.eta2 / 32.0
        #frequency = 1.0 / (T * 1024)
        #frequency = self.bandwidth / 128
        #frequency = 1 / (self.T * 1000)
        #frequency = self.get_input_freq(self.size, n_cycles, 1 / self.T)
        frequency = self.get_input_freq(self.size, n_cycles, 1 / (self.T * self.OSR))

        # We also specify a phase an offset these are hovewer optional.
        #phase = np.pi / 3
        #offset = 0.0

        # Instantiate the analog signal
        analog_signal = cbadc.analog_signal.Sinusoidal(self.amplitude, frequency, self.phase, self.offset)
        #analog_signal = cbadc.analog_signal.ConstantSignal(0.2)
        # print to ensure correct parametrization.
        #print(analog_signal)
        print(analog_signal)


        # Setup the simulation of the system
        # Simulate for 2^18 control cycles.
        #size = 1 << 18
        #size = 1 << 12
        #end_time = T * self.size
        #end_time = (self.size * 2 * self.T * self.OSR) + (self.k1 + self.k2) * self.T
        #end_time = self.size * self.T
        end_time = self.size * self.T * self.OSR

        # Instantiate the simulator.
        simulator = cbadc.simulator.get_simulator(
            #analog_system,
            analog_system = analog_frontend.analog_system,
            #digital_control,
            digital_control = analog_frontend.digital_control,
            input_signal = [analog_signal],
            #clock = clock,
            clock = analog_frontend.digital_control.clock,
            t_stop = end_time,
        )
        # Depending on your analog system the step above might take some time to
        # compute as it involves precomputing solutions to initial value problems.

        # To verify the simulation parametrization we can
        print(simulator)

        # Let's print the first 20 control decisions.
        """index = 0
        for s in cbadc.utilities.show_status(simulator):
            if index > 19:
                break
            print(f"step:{index} -> s:{np.array(s)}")
            index += 1"""

        # To verify the simulation parametrization we can
        #print(simulator)


        # Setup extended simulator
        # Repeating the steps above we now get for the following
        # ten control cycles.
        #ext_simulator = cbadc.simulator.extended_simulation_result(simulator)
        """for res in cbadc.utilities.show_status(ext_simulator):
            if index > 29:
                break
            print(f"step:{index} -> s:{res['control_signal']}, x:{res['analog_state']}")
            index += 1"""

        
        # Safe control signal to file
        # Instantiate a new simulator and control.
        """simulator = cbadc.simulator.get_simulator(
            analog_system,
            digital_control,
            [analog_signal],
            clock = clock,
            t_stop=end_time
        )"""

        return simulator

    def simulate_digital_estimator_batch(self) -> cbadc.digital_estimator.BatchEstimator:
        """Simulates the batch version of the digital estimator.

        Currently not used.
        """
        # Setup analog system and digital control
        N = 6
        M = N
        beta = 6250.0
        rho = -1e-2
        kappa = -1.0
        A = [
            [beta * rho, 0, 0, 0, 0, 0],
            [beta, beta * rho, 0, 0, 0, 0],
            [0, beta, beta * rho, 0, 0, 0],
            [0, 0, beta, beta * rho, 0, 0],
            [0, 0, 0, beta, beta * rho, 0],
            [0, 0, 0, 0, beta, beta * rho],
        ]
        B = [[beta], [0], [0], [0], [0], [0]]
        CT = np.eye(N)
        Gamma = [
            [kappa * beta, 0, 0, 0, 0, 0],
            [0, kappa * beta, 0, 0, 0, 0],
            [0, 0, kappa * beta, 0, 0, 0],
            [0, 0, 0, kappa * beta, 0, 0],
            [0, 0, 0, 0, kappa * beta, 0],
            [0, 0, 0, 0, 0, kappa * beta],
        ]
        Gamma_tildeT = np.eye(N)
        T = 1.0 / (2 * beta)
        clock = cbadc.analog_signal.Clock(T)

        analog_system = cbadc.analog_system.AnalogSystem(A, B, CT, Gamma, Gamma_tildeT)
        digital_control = cbadc.digital_control.DigitalControl(clock, M)

        # Summarize the analog system, digital control, and digital estimator.
        print(analog_system, "\n")
        print(digital_control)


        # Setup placeholder dummy control signal
        # Another way would be to use a random control signal. Such a generator
        # is already provided in the :func:`cbadc.utilities.random_control_signal`
        # function. Subsequently, a random (random 1-0 valued M tuples) control signal
        # of length
        #sequence_length = 10
        #sequence_length = 11

        # can conveniently be created as

        control_signal_sequences = cbadc.utilities.random_control_signal(
            self.m_number_of_digital_states, stop_after_number_of_iterations=self.size, random_seed=42
        )
        # where random_seed and stop_after_number_of_iterations are fully optional


        # Setup the digital estimator
        # Set the bandwidth of the estimator
        #eta2 = 1e7

        # Set the batch size
        #K1 = sequence_length
        #K1 = 5

        # Set lookahead
        #K2 = 1

        # Instantiate the digital estimator (this is where the filter coefficients are
        # computed).
        digital_estimator_batch = cbadc.digital_estimator.BatchEstimator(
            analog_system, digital_control, self.eta2, self.k1, self.k2
        )

        print("Batch estimator:\n\n", digital_estimator_batch, "\n")

        return digital_estimator_batch


    def simulate_digital_estimator_fir(self) -> cbadc.digital_estimator.FIRFilter:
        """Simulates the FIR version of the digital estimator.

        This includes the calculation of the FIR filter coefficients and
            creating an estimation of the input signal.
        """
        # Setup analog system and digital control
        #N = 6
        #M = N
        #beta = 6250.0
        #rho = -1e-2
        #kappa = -1.0

        """A = [
            [beta * rho, 0, 0, 0, 0, 0],
            [beta, beta * rho, 0, 0, 0, 0],
            [0, beta, beta * rho, 0, 0, 0],
            [0, 0, beta, beta * rho, 0, 0],
            [0, 0, 0, beta, beta * rho, 0],
            [0, 0, 0, 0, beta, beta * rho],
        ]"""
        """a: list[list[float]] = list[list[float]]()
        for row_index_a in range(self.n_number_of_analog_states):
            a.append(list[float]())
            for column_index_a in range(self.n_number_of_analog_states):
                if row_index_a == 0:
                    if column_index_a == 0:
                        a[row_index_a].append(self.beta * self.rho)
                    else:
                        a[row_index_a].append(float(0.0))
                else:
                    if column_index_a == row_index_a - 1:
                        a[row_index_a].append(self.beta)
                    elif column_index_a == row_index_a:
                        a[row_index_a].append(self.beta * self.rho)
                    else:
                        a[row_index_a].append(float(0.0))"""

        #B = [[beta], [0], [0], [0], [0], [0]]
        """b: list[float] = list[float]()
        for index_b in range(self.n_number_of_analog_states):
            if index_b == 0:
                b.append(self.beta)
            else:
                b.append(0.0)"""

        #CT = np.eye(N)
        ct = np.eye(self.n_number_of_analog_states)

        """Gamma = [
            [kappa * beta, 0, 0, 0, 0, 0],
            [0, kappa * beta, 0, 0, 0, 0],
            [0, 0, kappa * beta, 0, 0, 0],
            [0, 0, 0, kappa * beta, 0, 0],
            [0, 0, 0, 0, kappa * beta, 0],
            [0, 0, 0, 0, 0, kappa * beta],
        ]"""
        """gamma: list[list[float]] = list[list[float]]()
        for row_index_gamma in range(self.n_number_of_analog_states):
            gamma.append(list[float]())
            for column_index_gamma in range(self.n_number_of_analog_states):
                if column_index_gamma == row_index_gamma:
                    gamma[row_index_gamma].append(self.kappa * self.beta)
                else:
                    gamma[row_index_gamma].append(float(0.0))"""

        #Gamma_tildeT = np.eye(N)
        #gamma_tildeT = np.eye(self.n_number_of_analog_states)

        #analog_system = cbadc.analog_system.AnalogSystem(a, b, ct, gamma, gamma_tildeT)
        #gamma = self.OSR / np.pi
        #omega_3dB = 2 * np.pi * self.eta2
        #beta_vec = gamma * (omega_3dB / 2) * np.ones(self.n_number_of_analog_states)
        #alpha_vec = -(omega_3dB / 2) / gamma * np.ones(self.n_number_of_analog_states - 1)
        #rho_vec = np.zeros(self.n_number_of_analog_states)
        #T = 1.0 / (2.0 * beta_vec[0])
        #self.T = T
        #Gamma = np.diag(-beta_vec)
        #analog_system = cbadc.analog_system.LeapFrog(beta_vec, alpha_vec, rho_vec, Gamma)
        #betaVec = self.beta * np.ones(self.n_number_of_analog_states)
        #rhoVec = betaVec * self.rho
        #kappaVec = self.kappa * self.beta * np.eye(self.n_number_of_analog_states)
        # Instantiate a chain-of-integrators analog system.
        #analog_system = cbadc.analog_system.ChainOfIntegrators(betaVec, rhoVec, kappaVec)

        #T = 1.0 / (2 * self.beta)
        #clock = cbadc.analog_signal.Clock(T)

        #digital_control = cbadc.digital_control.DigitalControl(clock, self.m_number_of_digital_states)

        # Summarize the analog system, digital control, and digital estimator.
        #print(analog_system, "\n")
        #print(digital_control)


        # Setup placeholder dummy control signal
        # Another way would be to use a random control signal. Such a generator
        # is already provided in the :func:`cbadc.utilities.random_control_signal`
        # function. Subsequently, a random (random 1-0 valued M tuples) control signal
        # of length
        #sequence_length = 10
        #sequence_length = 11

        # can conveniently be created as

        #control_signal_sequences = cbadc.utilities.random_control_signal(self.m_number_of_digital_states,
        #    stop_after_number_of_iterations=sequence_length, random_seed=42
        #)
        # where random_seed and stop_after_number_of_iterations are fully optional


        # Setup the digital estimator
        # Set the bandwidth of the estimator
        #eta2 = 1e7

        # Set the batch size of lookback
        #K1 = sequence_length
        #K1 = 5

        # Set the batch size of lookahead
        #K2 = 1

        # Redo for ENOB = 10 and ENOB = 12
        # Instantiate LeapFrog analog_system and digital_control
        analog_frontend = cbadc.synthesis.get_leap_frog(OSR = self.OSR, N = self.n_number_of_analog_states, BW = self.bandwidth)
        eta2 = self.get_eta2(analog_frontend, self.bandwidth)
        self.eta2 = eta2
        self.T = analog_frontend.digital_control.clock.T

        # Instantiate the digital estimator (this is where the filter coefficients are
        # computed).
        #digital_estimator_fir: cbadc.digital_estimator.FIRFilter = cbadc.digital_estimator.FIRFilter(analog_system, digital_control, self.eta2, self.k1, self.k2, fixed_point = cbadc.utilities.FixedPoint(self.data_width, 1.0), downsample = self.down_sample_rate)
        #digital_estimator_fir: cbadc.digital_estimator.FIRFilter = cbadc.digital_estimator.FIRFilter(analog_system, digital_control, self.eta2, self.k1, self.k2, fixed_point = cbadc.utilities.FixedPoint(self.data_width, 4.0))
        #digital_estimator_fir: cbadc.digital_estimator.FIRFilter = cbadc.digital_estimator.FIRFilter(analog_system, digital_control, self.eta2, self.k1, self.k2)
        digital_estimator_fir: cbadc.digital_estimator.FIRFilter = cbadc.digital_estimator.FIRFilter(analog_frontend.analog_system, analog_frontend.digital_control, self.eta2, self.k1, self.k2, fixed_point = cbadc.utilities.FixedPoint(self.data_width, 1.0), downsample = self.down_sample_rate)
        #digital_estimator_fir: cbadc.digital_estimator.FIRFilter = cbadc.digital_estimator.FIRFilter(analog_frontend.analog_system, analog_frontend.digital_control, self.eta2, self.k1, self.k2, downsample = self.down_sample_rate)
        print("FIR estimator\n\n", digital_estimator_fir, "\n")
        self.fir_h_matrix = digital_estimator_fir.h
        #self.fir_hb_matrix = digital_estimator_fir.h[0 : self.n_number_of_analog_states - 1, 0 : self.k1]
        
        tmp_fir_hb_matrix = digital_estimator_fir.h[0 : self.n_number_of_analog_states - 1, 0 : self.k1]
        self.fir_hb_matrix = tmp_fir_hb_matrix.copy()
        for index in range(self.fir_hb_matrix.shape[1]):
            self.fir_hb_matrix[0][index] = tmp_fir_hb_matrix[0][tmp_fir_hb_matrix.shape[1] - index - 1]

        self.fir_hf_matrix = digital_estimator_fir.h[0 : self.n_number_of_analog_states - 1, self.k1 : self.k1 + self.k2]
        print("FIR hb matrix:\n", self.fir_hb_matrix)
        print("FIR hf matrix:\n", self.fir_hf_matrix)

        # Set the frequency of the analog simulation signal
        #frequency = 1.0 / (T * self.OSR)
        #frequency = 1.0 / (T * 1024)
        #frequency = self.bandwidth / 128
        n_cycles = 1<<3
        #frequency = self.get_input_freq(self.size, n_cycles, 1 / self.T)
        frequency = self.get_input_freq(self.size, n_cycles, 1 / (self.T * self.OSR))

        #frequency = 1 / (self.T * 1000)

        # Instantiate the analog signal
        analog_signal = cbadc.analog_signal.Sinusoidal(self.amplitude, frequency, self.phase, self.offset)
        #analog_signal = cbadc.analog_signal.ConstantSignal(0.2)
        # print to ensure correct parametrization.
        #print(analog_signal)
        print(analog_signal)

        # Setup the simulation time of the system
        #end_time = T * self.size
        #end_time = self.size * self.T
        #end_time = (self.size * 2 * self.T * self.OSR) + (self.k1 + self.k2) * self.T
        #end_time = self.size * self.T
        end_time = self.size * self.T * self.OSR

        # Instantiate the simulator.
        simulator = cbadc.simulator.get_simulator(
            #analog_system,
            analog_system = analog_frontend.analog_system,
            #digital_control,
            digital_control = analog_frontend.digital_control,
            input_signal = [analog_signal],
            #clock = clock,
            clock = analog_frontend.digital_control.clock,
            t_stop = end_time,
            
        )

        digital_estimator_fir(simulator)

        return digital_estimator_fir


    def get_fir_lookback_coefficient_matrix(self) -> np.ndarray:
        """Returns the FIR lookback coefficients.
        """
        return self.fir_hb_matrix


    def get_fir_lookahead_coefficient_matrix(self) -> np.ndarray:
        """Returns the FIR lookahead coefficients.
        """
        return self.fir_hf_matrix


    def simulate_fir_filter_self_programmed(self, number_format: str = "both"):
        """This is a self programmed version of the FIR filter.

        Highly inefficient. Not intended for extensive use. Was mainly programmed
            as playground during the creation of the SystemVerilog code.
        """
        lookback: list[int] = list[int]()
        for lookback_index in range(self.k1):
            lookback.append(0)
        lookahead: list[int] = list[int]()
        for lookahead_index in range(self.k2):
            lookahead.append(0)
        with open(self.path + "/control_signal.csv", "r") as input_csv_file:
            output_csv_file_integer = None
            output_csv_file_float = None
            if number_format == "integer" or number_format == "both":
                output_csv_file_integer = open(self.path + "/digital_estimation_high_level_self_programmed_integer.csv", "w")
            if number_format == "float" or number_format == "both":
                output_csv_file_float = open(self.path + "/digital_estimation_high_level_self_programmed_float.csv", "w")

            lines: list[str] = input_csv_file.readlines()
            startup_count: int = 0
            startup_counter: int = 0
            downsample_cycle: int = 0
            for line in lines:
                line = line.rstrip(",\n")
                line_int: int = int(line, base = 2)                    
                lookahead.append(line_int)
                lookback.insert(0, lookahead.pop(0))
                lookback.pop(self.k1)
                
                downsample_cycle = (downsample_cycle + 1) % self.down_sample_rate
                if downsample_cycle != 0:
                    continue

                lookback_result: int = 0
                for lookback_index in range(self.k1):
                    for value_index in range(self.m_number_of_digital_states):
                        if (lookback[lookback_index] >> value_index) & 1:
                            lookback_result += self.fir_hb_matrix[0][lookback_index][value_index]
                        else:
                            lookback_result -= self.fir_hb_matrix[0][lookback_index][value_index]
                lookahead_result: int = 0
                for lookahead_index in range(self.k2):
                    for value_index in range(self.m_number_of_digital_states):
                        if (lookahead[lookahead_index] >> value_index) & 1:
                            lookahead_result += self.fir_hf_matrix[0][lookahead_index][value_index]
                        else:
                            lookahead_result -= self.fir_hf_matrix[0][lookahead_index][value_index]
                if startup_counter < startup_count:
                    startup_counter += 1
                    continue
                if number_format == "integer" or number_format == "both":
                    output_csv_file_integer.write(str(lookback_result + lookahead_result) + ", " + str(lookback_result) + ", " + str(lookahead_result) + ",\n")
                if number_format == "float" or number_format == "both":
                    lookback_result_float: float = float(lookback_result) / (2.0**(self.data_width - 1))
                    lookahead_result_float: float = float(lookahead_result) / (2.0**(self.data_width - 1))
                    output_csv_file_float. write(str(lookback_result_float + lookahead_result_float) + "," + str(lookback_result_float) + "," + str(lookahead_result_float) + ",\n")
            output_csv_file_integer.close()
            output_csv_file_float.close()
            input_csv_file.close()


    def compare_simulation_system_verilog_to_high_level(self, fixed_point: bool = False, fixed_point_mantissa_bits: int = 0, offset: int = 0):
        """Compares the RTL simulation output to the cbadc golden sample.

        Watches if the RTL simulation differs from the golden sample more
            than the set SNR. This is used for the automated quality
            assurance.
        """
        with open(self.path + "/digital_estimation.csv", "r") as system_verilog_simulation_csv_file:
            with open(self.path + "/digital_estimation_high_level.csv", "r") as high_level_simulation_csv_file:
                with open(self.path + "/digital_estimation_system_verilog_vs_high_level.csv", "w") as comparison_csv_file:
                    system_verilog_simulation_results = system_verilog_simulation_csv_file.readlines()
                    system_verilog_simulation_results = system_verilog_simulation_results[offset : ]
                    high_level_simulation_results = high_level_simulation_csv_file.readlines()

                    difference_below_snr: bool = True

                    for index in range(len(system_verilog_simulation_results)):
                        if index >= len(high_level_simulation_results):
                            break
                        system_verilog_simulation_result = system_verilog_simulation_results[index].rsplit(",")[0]
                        high_level_simulation_result = high_level_simulation_results[index].rsplit(",")[0]

                        if fixed_point:
                            system_verilog_simulation_float: float = float(system_verilog_simulation_result) / (2**(fixed_point_mantissa_bits - 1))
                        else:
                            system_verilog_simulation_float: float = float(system_verilog_simulation_result)
                        high_level_simulation_float: float = float(high_level_simulation_result)
                        difference: float = system_verilog_simulation_float - high_level_simulation_float
                        if index > (self.k1 + self.k2) / self.down_sample_rate and difference > 3.87e-5:
                            difference_below_snr = False
                        comparison_csv_file.write(str(difference) + ",\n")
                    system_verilog_simulation_csv_file.close()
                    high_level_simulation_csv_file.close()
                    comparison_csv_file.close()


                    """offset_counter: int = 0
                    while True:
                        while offset_counter < offset:
                            system_verilog_simulation_csv_file.readline()
                            offset_counter += 1
                        system_verilog_simulation_line: str = system_verilog_simulation_csv_file.readline().rsplit(",")[0]
                        if system_verilog_simulation_line == "":
                            break
                        if fixed_point:
                            system_verilog_simulation_float: float = float(system_verilog_simulation_line) / (2**(fixed_point_mantissa_bits - 1))
                        else:
                            system_verilog_simulation_float: float = float(system_verilog_simulation_line)
                        high_level_simulation_line: str = high_level_simulation_csv_file.readline().rsplit(",")[0]
                        if high_level_simulation_line == "":
                            break
                        high_level_simulation_float: float = float(high_level_simulation_line)
                        comparison_csv_file.write(str(system_verilog_simulation_float - high_level_simulation_float) + ",\n")
                    system_verilog_simulation_csv_file.close()
                    high_level_simulation_csv_file.close()
                    comparison_csv_file.close()"""

                    return difference_below_snr


    def plot_results(self):
        """Dummy class function for the automated execution.

        This was split into a class and non-class version to be able
            to execute the plot_results function manually and automated.
        """
        plot_results(path = self.path, k1 = self.k1, k2 = self.k2, size = self.size, T = self.T, OSR = self.OSR, down_sample_rate = self.down_sample_rate)
                

if __name__ == '__main__':
    """Main function for debugging purposes of the simulation class.
    """
    #high_level_simulation: DigitalEstimatorParameterGenerator = DigitalEstimatorParameterGenerator(k1 = 512, k2 = 128, data_width = 31)
    #simulation: cbadc.simulator.PreComputedControlSignalsSimulator = high_level_simulation.simulate_analog_system()
    #high_level_simulation.write_control_signal_to_csv_file(simulation)

    #estimator = high_level_simulation.simulate_digital_estimator_fir()
    #SystemVerilogSyntaxGenerator.ndarray_to_system_verilog_array(np.array(convert_coefficient_matrix_to_lut_entries(high_level_simulation.fir_hb_matrix, 4)))
    #SystemVerilogSyntaxGenerator.ndarray_to_system_verilog_array(np.array(convert_coefficient_matrix_to_lut_entries(high_level_simulation.fir_hf_matrix, 4)))
    """high_level_simulation.simulate_digital_estimator_fir()
    lut_entry_list: list[int] = convert_coefficient_matrix_to_lut_entries(high_level_simulation.fir_hb_matrix, lut_input_width = 4)
    lut_entry_array: np.ndarray = np.array(lut_entry_list)
    SystemVerilogSyntaxGenerator.ndarray_to_system_verilog_array(lut_entry_array)
    lut_entry_list = convert_coefficient_matrix_to_lut_entries(high_level_simulation.fir_hf_matrix, lut_input_width = 4)
    lut_entry_array = np.array(lut_entry_list)
    SystemVerilogSyntaxGenerator.ndarray_to_system_verilog_array(lut_entry_array)"""

    plot_results()