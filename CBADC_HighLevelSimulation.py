import matplotlib.pyplot as plt
import cbadc
import numpy as np

def simulate_analog_system():
    # Setup the analog System.
    # We fix the number of analog states.
    N = 6
    # Set the amplification factor.
    beta = 6250.0
    rho = -1e-2
    kappa = -1.0
    # In this example, each nodes amplification and local feedback will be set
    # identically.
    betaVec = beta * np.ones(N)
    rhoVec = betaVec * rho
    kappaVec = kappa * beta * np.eye(N)

    # Instantiate a chain-of-integrators analog system.
    analog_system = cbadc.analog_system.ChainOfIntegrators(betaVec, rhoVec, kappaVec)
    # print the analog system such that we can very it being correctly initalized.
    print(analog_system)


    # Setup the digital control.
    # Set the time period which determines how often the digital control updates.
    T = 1.0 / (2 * beta)
    # Instantiate a corresponding clock.
    clock = cbadc.analog_signal.Clock(T)
    # Set the number of digital controls to be same as analog states.
    M = N
    # Initialize the digital control.
    digital_control = cbadc.digital_control.DigitalControl(clock, M)
    # print the digital control to verify proper initialization.
    print(digital_control)


    # Setup the ananlog stimulation signal
    # Set the peak amplitude.
    amplitude = 0.5
    # Choose the sinusoidal frequency via an oversampling ratio (OSR).
    OSR = 1 << 9
    frequency = 1.0 / (T * OSR)

    # We also specify a phase an offset these are hovewer optional.
    phase = np.pi / 3
    offset = 0.0

    # Instantiate the analog signal
    analog_signal = cbadc.analog_signal.Sinusoidal(amplitude, frequency, phase, offset)
    # print to ensure correct parametrization.
    print(analog_signal)


    # Setup the simulation of the system
    # Simulate for 2^18 control cycles.
    size = 1 << 18
    end_time = T * size

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

    # Let's print the first 20 control decisions.
    index = 0
    for s in cbadc.utilities.show_status(simulator):
        if index > 19:
            break
        print(f"step:{index} -> s:{np.array(s)}")
        index += 1

    # To verify the simulation parametrization we can
    print(simulator)


    # Setup extended simulator
    # Repeating the steps above we now get for the following
    # ten control cycles.
    ext_simulator = cbadc.simulator.extended_simulation_result(simulator)
    for res in cbadc.utilities.show_status(ext_simulator):
        if index > 29:
            break
        print(f"step:{index} -> s:{res['control_signal']}, x:{res['analog_state']}")
        index += 1

    
    # Safe control signal to file
    # Instantiate a new simulator and control.
    simulator = cbadc.simulator.get_simulator(
        analog_system,
        digital_control,
        [analog_signal],
        clock = clock,
        t_stop=end_time
    )

    # Construct byte stream.
    byte_stream = cbadc.utilities.control_signal_2_byte_stream(simulator, M)

    cbadc.utilities.write_byte_stream_to_file(
        "sinusoidal_simulation.dat", print_next_10_bytes(byte_stream, size, index)
    )


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


def print_next_10_bytes(stream, size, index):
    for byte in cbadc.utilities.show_status(stream, size):
        if index < 40:
            print(f"{index} -> {byte}")
            index += 1
        yield byte
    

def simulate_digital_estimator_batch() -> cbadc.digital_estimator.BatchEstimator:
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
    sequence_length = 11

    # can conveniently be created as

    control_signal_sequences = cbadc.utilities.random_control_signal(
        M, stop_after_number_of_iterations=sequence_length, random_seed=42
    )
    # where random_seed and stop_after_number_of_iterations are fully optional


    # Setup the digital estimator
    # Set the bandwidth of the estimator
    eta2 = 1e7

    # Set the batch size
    #K1 = sequence_length
    K1 = 5

    # Set lookahead
    K2 = 1

    # Instantiate the digital estimator (this is where the filter coefficients are
    # computed).
    digital_estimator_batch = cbadc.digital_estimator.BatchEstimator(
        analog_system, digital_control, eta2, K1, K2
    )

    print("Batch estimator:\n\n", digital_estimator_batch, "\n")

    #digital_estimator_fir = cbadc.digital_estimator.FIRFilter(analog_system, digital_control, eta2, K1, K2)

    #print("FIR estimator\n\n", digital_estimator_fir, "\n")

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


def simulate_digital_estimator_fir() -> cbadc.digital_estimator.FIRFilter:
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
    sequence_length = 11

    # can conveniently be created as

    control_signal_sequences = cbadc.utilities.random_control_signal(
        M, stop_after_number_of_iterations=sequence_length, random_seed=42
    )
    # where random_seed and stop_after_number_of_iterations are fully optional


    # Setup the digital estimator
    # Set the bandwidth of the estimator
    eta2 = 1e7

    # Set the batch size
    #K1 = sequence_length
    K1 = 5

    # Set lookahead
    K2 = 1

    # Instantiate the digital estimator (this is where the filter coefficients are
    # computed).
    #digital_estimator_batch = cbadc.digital_estimator.BatchEstimator(
    #    analog_system, digital_control, eta2, K1, K2
    #)

    #print("Batch estimator:\n\n", digital_estimator_batch, "\n")

    digital_estimator_fir = cbadc.digital_estimator.FIRFilter(analog_system, digital_control, eta2, K1, K2)

    print("FIR estimator\n\n", digital_estimator_fir, "\n")

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


if __name__ == '__main__':
    #simulate_analog_system()
    simulate_digital_estimator_batch()
    simulate_digital_estimator_fir()
    