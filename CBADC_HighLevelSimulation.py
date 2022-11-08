from array import array
from curses import KEY_A1
from math import ceil
from operator import index
from tokenize import Double
from turtle import shape
from typing import Generator, Iterator, Union
import matplotlib.pyplot as plt
import cbadc
import numpy as np
from sqlalchemy import true


import SystemVerilogModule
import SystemVerilogSyntaxGenerator


def convert_coefficient_matrix_to_lut_entries(coefficient_matrix: np.ndarray, lut_input_width: int) -> np.ndarray:
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
        last_lut_width = 4
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


def plot_results():
    with open("../df/sim/SystemVerilogFiles/digital_estimation.csv", "r") as system_verilog_simulation_csv_file:
        with open("../df/sim/SystemVerilogFiles/digital_estimation_high_level.csv", "r") as high_level_simulation_csv_file:
            pass
        system_verilog_simulation_csv_file.close()
        high_level_simulation_csv_file.close()


class DigitalEstimatorParameterGenerator():
    path: str = "../df/sim/SystemVerilogFiles"
    # Set the number of analog states
    n_number_of_analog_states: int = 6
    # Set the number of digital states
    m_number_of_digital_states: int = n_number_of_analog_states
    # Set the amplification factor
    beta: float = 6250.0
    rho: float = -1e-2
    kappa: float = -1.0
    # Set the bandwidth of the estimator
    eta2 = 1e7
    # Set the batch size of lookback
    #K1 = sequence_length
    k1 = 5
    # Set the batch size of lookahead
    k2 = 1
    # Digital estimator data width
    data_width: int = 64

    # Values for input stimulus generation
    # amplitude of the signal
    amplitude = 0.5
    # Choose the sinusoidal frequency via an oversampling ratio (OSR).
    OSR = 1 << 9
    # We also specify a phase an offset these are hovewer optional.
    phase = np.pi / 3
    offset = 0.0
    # Length of the simulation
    size: int = 1 << 12

    # FIR filter coefficients
    fir_h_matrix: np.ndarray
    fir_hb_matrix: np.ndarray
    fir_hf_matrix: np.ndarray

    def __init__(self,
            path: str = "../df/sim/SystemVerilogFiles",
            n_number_of_analog_states: int = 6,
            m_number_of_digital_states: int = n_number_of_analog_states,
            beta: float = 6250.0,
            rho: float = -1e-2,
            kappa: float = -1.0,
            eta2: float = 1e7,
            k1: int = 5,
            k2: int = 1,
            data_width: int = 64,
            amplitude: float = 0.5,
            OSR: int = 1 << 9,
            phase: float = np.pi / 3.0,
            offset: float = 0.0,
            size: int = 1 << 12) -> None:
        self.path = path
        self.n_number_of_analog_states = n_number_of_analog_states
        self.m_number_of_digital_states = m_number_of_digital_states
        self.beta = beta
        self.rho = rho
        self.kappa = kappa
        self.eta2 = eta2
        self.k1 = k1
        self.k2 = k2
        self.data_width = data_width
        self.amplitude = amplitude
        self.OSR = OSR
        self.phase = phase
        self.offset = offset
        self.size = size

    def write_control_signal_to_csv_file(self, values: np.ndarray):
        with open(self.path + "/control_signal.csv", "w") as csv_file:
            for single_control_signal in values:
                single_control_signal_string: str = ""
                for bit_index in range(self.m_number_of_digital_states - 1, -1, -1):
                    single_control_signal_string = single_control_signal_string + str(int(single_control_signal[bit_index]))
                csv_file.write(single_control_signal_string + ",\n")
            csv_file.close()

    def write_digital_estimation_fir_to_csv_file(self, values: np.ndarray):
        with open(self.path + "/digital_estimation_high_level.csv", "w") as csv_file:
            for single_estimation_value in values:
                single_estimation_value_string: str = str(single_estimation_value)
                single_estimation_value_string = single_estimation_value_string.lstrip("[")
                single_estimation_value_string = single_estimation_value_string.lstrip(" ")
                single_estimation_value_string = single_estimation_value_string.rstrip("],")
                csv_file.write(single_estimation_value_string + ",\n")
            csv_file.close()


    def simulate_analog_system(self):
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
        betaVec = self.beta * np.ones(self.n_number_of_analog_states)
        rhoVec = betaVec * self.rho
        kappaVec = self.kappa * self.beta * np.eye(self.n_number_of_analog_states)

        # Instantiate a chain-of-integrators analog system.
        analog_system = cbadc.analog_system.ChainOfIntegrators(betaVec, rhoVec, kappaVec)
        # print the analog system such that we can very it being correctly initalized.
        print(analog_system)


        # Setup the digital control.
        # Set the time period which determines how often the digital control updates.
        T = 1.0 / (2 * self.beta)
        # Instantiate a corresponding clock.
        clock = cbadc.analog_signal.Clock(T)
        # Set the number of digital controls to be same as analog states.
        #M = N
        # Initialize the digital control.
        digital_control = cbadc.digital_control.DigitalControl(clock, self.m_number_of_digital_states)
        # print the digital control to verify proper initialization.
        print(digital_control)


        # Setup the analog stimulation signal
        # Set the peak amplitude.
        #amplitude = 0.5
        # Choose the sinusoidal frequency via an oversampling ratio (OSR).
        #OSR = 1 << 9
        frequency = 1.0 / (T * self.OSR)

        # We also specify a phase an offset these are hovewer optional.
        #phase = np.pi / 3
        #offset = 0.0

        # Instantiate the analog signal
        analog_signal = cbadc.analog_signal.Sinusoidal(self.amplitude, frequency, self.phase, self.offset)
        # print to ensure correct parametrization.
        print(analog_signal)


        # Setup the simulation of the system
        # Simulate for 2^18 control cycles.
        #size = 1 << 18
        #size = 1 << 12
        end_time = T * self.size

        # Instantiate the simulator.
        simulator = cbadc.simulator.get_simulator(
            analog_system,
            digital_control,
            [analog_signal],
            clock = clock,
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

        # Construct byte stream.
        #byte_stream = cbadc.utilities.control_signal_2_byte_stream(simulator, self.m_number_of_digital_states)

        #cbadc.utilities.write_byte_stream_to_file(
        #    "sinusoidal_simulation.dat", self.print_next_10_bytes(byte_stream, size, index)
        #)


        """# Analog state evaluation
        # Set sampling time two orders of magnitude smaller than the control period
        Ts = T / 100.0
        # Instantiate a corresponding clock
        observation_clock = cbadc.analog_signal.Clock(Ts)

        # Simulate for 65536 control cycles.
        size = 1 << 16

        # Initialize a new digital control.
        new_digital_control = cbadc.digital_control.DigitalControl(clock, M)

        # Instantiate a new simulator with a sampling time.
        simulator = cbadc.simulator.AnalyticalSimulator(
            analog_system, new_digital_control, [analog_signal], observation_clock
        )

        # Create data containers to hold the resulting data.
        time_vector = np.arange(size) * Ts / T
        states = np.zeros((N, size))
        control_signals = np.zeros((M, size), dtype=np.int8)

        # Iterate through and store states and control_signals.
        simulator = cbadc.simulator.extended_simulation_result(simulator)
        for index in cbadc.utilities.show_status(range(size)):
            res = next(simulator)
            states[:, index] = res["analog_state"]
            control_signals[:, index] = res["control_signal"]

        # Plot all analog state evolutions.
        plt.figure()
        plt.title("Analog state vectors")
        for index in range(N):
            plt.plot(time_vector, states[index, :], label=f"$x_{index + 1}(t)$")
        plt.grid(visible=True, which="major", color="gray", alpha=0.6, lw=1.5)
        plt.xlabel("$t/T$")
        plt.xlim((0, 10))
        plt.legend()

        # reset figure size and plot individual results.
        plt.rcParams["figure.figsize"] = [6.40, 6.40 * 2]
        fig, ax = plt.subplots(N, 2)
        for index in range(N):
            color = next(ax[0, 0]._get_lines.prop_cycler)["color"]
            ax[index, 0].grid(visible=True, which="major", color="gray", alpha=0.6, lw=1.5)
            ax[index, 1].grid(visible=True, which="major", color="gray", alpha=0.6, lw=1.5)
            ax[index, 0].plot(time_vector, states[index, :], color=color)
            ax[index, 1].plot(time_vector, control_signals[index, :], "--", color=color)
            ax[index, 0].set_ylabel(f"$x_{index + 1}(t)$")
            ax[index, 1].set_ylabel(f"$s_{index + 1}(t)$")
            ax[index, 0].set_xlim((0, 15))
            ax[index, 1].set_xlim((0, 15))
            ax[index, 0].set_ylim((-1, 1))
        fig.suptitle("Analog state and control contribution evolution")
        ax[-1, 0].set_xlabel("$t / T$")
        ax[-1, 1].set_xlabel("$t / T$")
        fig.tight_layout()"""

        return simulator


    def print_next_10_bytes(self, stream, size, index):
        for byte in cbadc.utilities.show_status(stream, size):
            if index < 40:
                print(f"{index} -> {byte}")
                index += 1
            yield byte
        

    def simulate_digital_estimator_batch(self) -> cbadc.digital_estimator.BatchEstimator:
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


        """# Set control signal iterator
        digital_estimator(control_signal_sequences)


        # Producing Estimates
        for i in digital_estimator:
            print(i)


        # Load control signal from file
        byte_stream = cbadc.utilities.read_byte_stream_from_file("sinusoidal_simulation.dat", M)
        control_signal_sequences = cbadc.utilities.byte_stream_2_control_signal(byte_stream, M)


        # Estimating the input
        stop_after_number_of_iterations = 1 << 17
        u_hat = np.zeros(stop_after_number_of_iterations)
        K1 = 1 << 10
        K2 = 1 << 11
        digital_estimator = cbadc.digital_estimator.BatchEstimator(
            analog_system,
            digital_control,
            eta2,
            K1,
            K2,
            stop_after_number_of_iterations=stop_after_number_of_iterations,
        )
        # Set control signal iterator
        digital_estimator(control_signal_sequences)
        for index, u_hat_temp in enumerate(digital_estimator):
            u_hat[index] = u_hat_temp"""

        """t = np.arange(u_hat.size)
        plt.plot(t, u_hat)
        plt.xlabel("$t / T$")
        plt.ylabel("$\hat{u}(t)$")
        plt.title("Estimated input signal")
        plt.grid()
        plt.xlim((0, 1500))
        plt.ylim((-1, 1))
        plt.tight_layout()"""

        return digital_estimator_batch


    def simulate_digital_estimator_fir(self) -> cbadc.digital_estimator.FIRFilter:
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
        a: list[list[float]] = list[list[float]]()
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
                        a[row_index_a].append(float(0.0))

        #B = [[beta], [0], [0], [0], [0], [0]]
        b: list[float] = list[float]()
        for index_b in range(self.n_number_of_analog_states):
            if index_b == 0:
                b.append(self.beta)
            else:
                b.append(0.0)

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
        gamma: list[list[float]] = list[list[float]]()
        for row_index_gamma in range(self.n_number_of_analog_states):
            gamma.append(list[float]())
            for column_index_gamma in range(self.n_number_of_analog_states):
                if column_index_gamma == row_index_gamma:
                    gamma[row_index_gamma].append(self.kappa * self.beta)
                else:
                    gamma[row_index_gamma].append(float(0.0))

        #Gamma_tildeT = np.eye(N)
        gamma_tildeT = np.eye(self.n_number_of_analog_states)

        T = 1.0 / (2 * self.beta)
        clock = cbadc.analog_signal.Clock(T)

        analog_system = cbadc.analog_system.AnalogSystem(a, b, ct, gamma, gamma_tildeT)
        digital_control = cbadc.digital_control.DigitalControl(clock, self.m_number_of_digital_states)

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

        # Instantiate the digital estimator (this is where the filter coefficients are
        # computed).
        digital_estimator_fir: cbadc.digital_estimator.FIRFilter = cbadc.digital_estimator.FIRFilter(analog_system, digital_control, self.eta2, self.k1, self.k2, fixed_point = cbadc.utilities.FixedPoint(self.data_width, 1.0))
        #digital_estimator_fir: cbadc.digital_estimator.FIRFilter = cbadc.digital_estimator.FIRFilter(analog_system, digital_control, self.eta2, self.k1, self.k2, fixed_point = cbadc.utilities.FixedPoint(self.data_width, 4.0))
        print("FIR estimator\n\n", digital_estimator_fir, "\n")
        self.fir_h_matrix = digital_estimator_fir.h
        self.fir_hb_matrix = digital_estimator_fir.h[0 : self.n_number_of_analog_states - 1, 0 : self.k1]
        self.fir_hf_matrix = digital_estimator_fir.h[0 : self.n_number_of_analog_states - 1, self.k1 : self.k1 + self.k2]
        print("FIR hb matrix:\n", self.fir_hb_matrix)
        print("FIR hf matrix:\n", self.fir_hf_matrix)

        # Set the frequency of the analog simulation signal
        frequency = 1.0 / (T * self.OSR)

        # Instantiate the analog signal
        analog_signal = cbadc.analog_signal.Sinusoidal(self.amplitude, frequency, self.phase, self.offset)
        # print to ensure correct parametrization.
        print(analog_signal)

        # Setup the simulation time of the system
        end_time = T * self.size

        # Instantiate the simulator.
        simulator = cbadc.simulator.get_simulator(
            analog_system,
            digital_control,
            [analog_signal],
            clock = clock,
            t_stop = end_time,
        )

        digital_estimator_fir(simulator)


        """# Set control signal iterator
        digital_estimator(control_signal_sequences)


        # Producing Estimates
        for i in digital_estimator:
            print(i)


        # Load control signal from file
        byte_stream = cbadc.utilities.read_byte_stream_from_file("sinusoidal_simulation.dat", M)
        control_signal_sequences = cbadc.utilities.byte_stream_2_control_signal(byte_stream, M)


        # Estimating the input
        stop_after_number_of_iterations = 1 << 17
        u_hat = np.zeros(stop_after_number_of_iterations)
        K1 = 1 << 10
        K2 = 1 << 11
        digital_estimator = cbadc.digital_estimator.BatchEstimator(
            analog_system,
            digital_control,
            eta2,
            K1,
            K2,
            stop_after_number_of_iterations=stop_after_number_of_iterations,
        )
        # Set control signal iterator
        digital_estimator(control_signal_sequences)
        for index, u_hat_temp in enumerate(digital_estimator):
            u_hat[index] = u_hat_temp"""

        """t = np.arange(u_hat.size)
        plt.plot(t, u_hat)
        plt.xlabel("$t / T$")
        plt.ylabel("$\hat{u}(t)$")
        plt.title("Estimated input signal")
        plt.grid()
        plt.xlim((0, 1500))
        plt.ylim((-1, 1))
        plt.tight_layout()"""

        return digital_estimator_fir


    def get_fir_lookback_coefficient_matrix(self) -> np.ndarray:
        return self.fir_hb_matrix


    def get_fir_lookahead_coefficient_matrix(self) -> np.ndarray:
        return self.fir_hf_matrix


    def simulate_fir_filter(self, number_format_float: bool = False):
        lookback: list[int] = list[int]()
        for lookback_index in range(self.k1):
            lookback.append(0)
        lookahead: list[int] = list[int]()
        for lookahead_index in range(self.k2):
            lookahead.append(0)
        with open(self.path + "/control_signal.csv", "r") as input_csv_file:
            with open(self.path + "/digital_estimation_high_level_self_programmed.csv", "w") as output_csv_file:
                lines: list[str] = input_csv_file.readlines()
                startup_count: int = 0
                startup_counter: int = 0
                for line in lines:
                    line = line.rstrip(",\n")
                    line_int: int = int(line, base = 2)                    
                    lookback.insert(0, line_int)
                    lookahead.insert(0, lookback.pop(self.k1))
                    lookahead.pop(self.k2)

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
                    if number_format_float:
                        lookback_result_float: float = float(lookback_result) / (2.0**(self.data_width - 1))
                        lookahead_result_float: float = float(lookahead_result) / (2.0**(self.data_width - 1))
                        output_csv_file. write(str(lookback_result_float + lookahead_result_float) + "," + str(lookback_result_float) + "," + str(lookahead_result_float) + ",\n")
                    else:
                        output_csv_file.write(str(lookback_result + lookahead_result) + ", " + str(lookback_result) + ", " + str(lookahead_result) + ",\n")
                output_csv_file.close()
            input_csv_file.close()


    def compare_simulation_system_verilog_to_high_level(self, fixed_point: bool = False, fixed_point_mantissa_bits: int = 0, offset: int = 2):
        with open(self.path + "/digital_estimation.csv", "r") as system_verilog_simulation_csv_file:
            with open(self.path + "/digital_estimation_high_level.csv", "r") as high_level_simulation_csv_file:
                with open(self.path + "/digital_estimation_system_verilog_vs_high_level.csv", "w") as comparison_csv_file:
                    offset_counter: int = 0
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
                    comparison_csv_file.close()
                

if __name__ == '__main__':
    high_level_simulation: DigitalEstimatorParameterGenerator = DigitalEstimatorParameterGenerator(k1 = 32, k2 = 32)
    simulation: cbadc.simulator.PreComputedControlSignalsSimulator = high_level_simulation.simulate_analog_system()
    high_level_simulation.write_control_signal_to_csv_file(simulation)

    #high_level_simulation.simulate_digital_estimator_batch()
    """high_level_simulation.simulate_digital_estimator_fir()
    lut_entry_list: list[int] = convert_coefficient_matrix_to_lut_entries(high_level_simulation.fir_hb_matrix, lut_input_width = 4)
    lut_entry_array: np.ndarray = np.array(lut_entry_list)
    SystemVerilogSyntaxGenerator.ndarray_to_system_verilog_array(lut_entry_array)
    lut_entry_list = convert_coefficient_matrix_to_lut_entries(high_level_simulation.fir_hf_matrix, lut_input_width = 4)
    lut_entry_array = np.array(lut_entry_list)
    SystemVerilogSyntaxGenerator.ndarray_to_system_verilog_array(lut_entry_array)"""
