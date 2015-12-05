''' Run the 16.322 estimator.

Matt Vernacchia
2015 Nov 30
'''

from matplotlib import pyplot as plt
import numpy as np
import transforms3d.quaternions as quat
import argparse

from estimators.sensor_models.sensor_interface import KalmanSensors
from estimators.sensor_models.magnetometer import Magnetometer
from estimators.sensor_models.rate_gyro import RateGyro
from estimators.sensor_models.accelerometer import Accelerometer

from estimators.kraft_quat_ukf import KraftQautUKF
from estimators.utils import quat_utils
from estimators.utils.plot_utils_16322 import plot_single_state_vs_time


def rotation_dynamics(x, u, dt=0.01):
    # Attitude quaternion
    q = x[0:4]
    # Angular rate vector
    w = x[4:7]
    
    q_next = quat_utils.quat_propagate(q, w, dt)
    w_next = w + u * dt

    x_next = np.hstack((q_next, w_next))
    return x_next


def create_estimator(dt):
    # Create the sensors for the Kalman filter estimator (known bias parameters).
    magneto_est = Magnetometer(h_bias_ned=[0, 0, 0], h_bias_sensor=[0, 0, 0])
    magneto_est.is_stateful = False
    gyro_est = RateGyro(constant_bias=[0,0,0],
        dt=dt)
    accel_est = Accelerometer(a_bias_sensor=[0, 0, 0])
    accel_est.is_stateful = False

    # System process noise covariance
    process_std_dev = np.hstack((np.deg2rad([10, 10, 10])*dt, 
        np.deg2rad([10, 10, 10])*dt))
    W = np.diag(process_std_dev)**2

    # Number of system states.
    n_system_states = 7
    # Number of sensor bias states.
    n_sensor_states = 3

    est_sensors = KalmanSensors([gyro_est, magneto_est, accel_est],
        [[4, 5, 6], [0, 1, 2, 3], [0, 1, 2, 3]], n_system_states,
        [[7, 8, 9], [], []], n_sensor_states,
        lambda x, u: rotation_dynamics(x, u, dt),
        W,
        1)

    # Initial state estimate. Set the sensor bias states
    # to an initial estimate of zero.
    x_est_init = np.concatenate(([0.76, 0.76, 0., 0., 0., 0., 0.],
        np.zeros(n_sensor_states)))

    # Initial estimate covariance.
    # The std. dev. uncertainty of the initial system state
    # estimate
    system_state_init_std_dev = np.hstack((np.deg2rad([30., 30., 30.]), 
        np.deg2rad([0.1, 0.1, 0.1])))
    # The std. dev. uncertainty of the sensor bias states.
    sensor_state_init_std_dev = np.deg2rad([5., 5., 5.])
    Q_init = np.diag(np.concatenate((
        system_state_init_std_dev,
        sensor_state_init_std_dev
        )))**2

    # Create the Kalman Filter
    est = KraftQautUKF(
        x_est_init,
        Q_init,
        est_sensors.augmented_transition_function,
        est_sensors.augmented_process_covariance,
        est_sensors.measurement_function,
        est_sensors.noise_cov,
        )

    return est


def run(est, meas_source, n_steps, dt):
    # Set up the measurement source
    if meas_source == 'sim':
        # Create the sensors for the simulation (unknown, random bias parameters). 
        gyro_sim = RateGyro(dt=dt)
        magneto_sim = Magnetometer(h_bias_ned=[0, 0, 0], h_bias_sensor=[0, 0, 0])
        magneto_sim.is_stateful = False
        accel_sim = Accelerometer()
        accel_sim.is_stateful = False
        # Number of system states.
        n_system_states = 7
        sim_sensors = KalmanSensors([gyro_sim, magneto_sim, accel_sim],
            [[4, 5, 6], [0, 1, 2, 3], [0, 1, 2, 3]], n_system_states)

        # Initial true state
        x_init = np.array([1., 0., 0., 0., 0., 0., 0.])

        # True state trajectory
        x_traj = np.zeros((n_steps, len(x_init)))
        x_traj[0] = x_init

        # Control trajectory
        u_traj = np.zeros((n_steps, 3))
        for i in xrange(100, 110):
            u_traj[i] = np.deg2rad([10.0, 0, 0])

        # Sensor bias trajectories
        gyro_bias_traj = np.zeros((n_steps, 3))
    else:
        raise ValueError

    # Create trajectories for storing data
    x_est_traj = np.zeros((n_steps, len(est.x_est)))
    x_est_traj[0] = est.x_est
    t_traj = np.zeros(n_steps)
    Q_traj = np.zeros((n_steps, len(est.x_est)-1, len(est.x_est)-1))
    Q_traj[0] = est.Q
    y_traj = np.zeros((n_steps, len(sim_sensors.measurement_function(x_init))))

    # Run the fitler
    for i in xrange(1, n_steps):
        # Get measurements
        if meas_source == 'sim':
            # Update sensor state
            sim_sensors.update_sensor_state(x_traj[i-1])
            gyro_bias_traj[i] = gyro_sim.bias + gyro_sim.constant_bias
            y_traj[i] = sim_sensors.add_noise(sim_sensors.measurement_function(x_traj[i-1]))
            # Simulate the true dynamics.
            x_traj[i] = rotation_dynamics(x_traj[i-1], u_traj[i], dt)
        else:
            raise ValueError
        # Update filter estimate.
        est.propagate_dynamics(np.array([0, 0, 0]))
        est.update_measurement(y_traj[i])
        # Record the estimates.
        x_est_traj[i] = est.x_est
        Q_traj[i] = est.Q        
        t_traj[i] = t_traj[i-1] + dt

    print 'Final state est = '
    print est.x_est
    print 'Final estimate covariance Q = '
    print est.Q

    if meas_source == 'sim':
        return (t_traj, x_est_traj, Q_traj, y_traj, x_traj, gyro_bias_traj)
    else:
        raise ValueError


def plot_traj(t_traj, x_est_traj, Q_traj, y_traj, x_traj, gyro_bias_traj):
    ax = plt.subplot(3, 2, 1)
    colors = ['black', 'red', 'green', 'blue']
    for i in xrange(4):
        plt.plot(t_traj, x_traj[:, i], color=colors[i], linestyle='-',
            label='q[{:d}] true'.format(i))
        plt.plot(t_traj, x_est_traj[:, i], color=colors[i], linestyle='--',
            label='q[{:d}] est'.format(i), marker='x')
    plt.xlabel('Time [s]')
    plt.ylabel('Quaternion')
    plt.legend(framealpha=0.5)
    
    ax2 = plt.subplot(3, 2, 2, sharex=ax)
    Q_traj_padded = np.concatenate((
        np.zeros((len(t_traj), len(x_est_traj[0]), 1)),
        np.concatenate((
            np.zeros((len(t_traj), 1, len(x_est_traj[0])-1)),
            Q_traj), axis=1)
        ), axis=2)
    for i in [0, 1, 2]:
        plt.plot(t_traj, x_traj[:, i+4], color=colors[i+1], linestyle='-',
            label='w[{:d}] true'.format(i))
        plot_single_state_vs_time(ax2, t_traj, x_est_traj, Q_traj_padded, i+4,
            color=colors[i+1], label='w[{:d}] est'.format(i),
            linestyle='--')
    plt.xlabel('Time [s]')
    plt.ylabel('Angular rate [rad / s]')
    plt.legend(framealpha=0.5)

    ax3 = plt.subplot(3, 2, 3, sharex=ax)
    for i in xrange(3):
        plt.plot(t_traj, y_traj[:, i + 3], color=colors[i+1], marker='x',
            label='mag[{:d}]'.format(i))
    plt.xlabel('Time [s]')
    plt.ylabel('Mag Field [uT]')
    plt.legend(framealpha=0.5)

    ax4 = plt.subplot(3, 2, 4, sharex=ax)
    for i in [0, 1, 2]:
        plt.plot(t_traj, gyro_bias_traj[:, i], color=colors[i+1], linestyle='-',
            label='b[{:d}] true'.format(i))
        plot_single_state_vs_time(ax4, t_traj, x_est_traj, Q_traj_padded, i+7,
            color=colors[i+1], label='b[{:d}] est'.format(i),
            linestyle='--')
    plt.xlabel('Time [s]')
    plt.ylabel('Gyro bias [rad / s]')
    plt.legend(framealpha=0.5)

    ax5 = plt.subplot(3, 2, 5, sharex=ax)
    for i in xrange(3):
        plt.plot(t_traj, y_traj[:, i + 6], color=colors[i+1],
            label='accel[{:d}]'.format(i))
    plt.xlabel('Time [s]')
    plt.ylabel('Accel [m / s**2]')
    plt.legend(framealpha=0.5)


def main(args):
    np.set_printoptions(precision=3)
    dt=0.1
    
    # Create the estimator
    est = create_estimator(dt)

    # Run the estimator
    n_steps = 500
    t_traj, x_est_traj, Q_traj, y_traj, x_traj, gyro_bias_traj = \
        run(est, args.meas_source, n_steps, dt)

    # Plot the results
    plot_traj(t_traj, x_est_traj, Q_traj, y_traj, x_traj, gyro_bias_traj)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the state estimator.')
    parser.add_argument('--meas_source', type=str, choices=['sim'])
    args = parser.parse_args()
    main(args)