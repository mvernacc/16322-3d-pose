''' Run the 16.322 estimator.

Matt Vernacchia
2015 Nov 30
'''

from matplotlib import pyplot as plt
import numpy as np
import transforms3d.quaternions as quat
import transforms3d.taitbryan as tb
import argparse
import cPickle as pickle

from estimators.sensor_models.sensor_interface import KalmanSensors
from estimators.sensor_models.magnetometer import Magnetometer, calibrate_data
from estimators.sensor_models.rate_gyro import RateGyro
from estimators.sensor_models.accelerometer import Accelerometer

from estimators.kraft_quat_ukf import KraftQautUKF
from estimators.utils import quat_utils
from estimators.utils.plot_utils_16322 import plot_single_state_vs_time


def rotation_dynamics(x, u, dt=0.01):
    # Attitude quaternion
    q_world2body = x[0:4]
    # Angular rate vector
    w_body = x[4:7]

    w_world = quat_utils.rotate_frame(w_body, q_world2body)
    
    q_next = quat_utils.quat_propagate(q_world2body, w_world, dt)
    w_next = w_body + u * dt

    x_next = np.hstack((q_next, w_next))
    return x_next


def create_estimator(dt, use_mag, mag_cal):
    # Create the sensors for the Kalman filter estimator (known bias parameters).
    if mag_cal is None:
        magneto_est = Magnetometer(noise_std_dev=3,
            b=[0, 0, 0], D=np.zeros((3,3)))
    else:
        # Load the magnetometer calibration parameters.
        with open(mag_cal, 'rb') as f:
            mag_cal_data = pickle.load(f)
        magneto_est = Magnetometer(noise_std_dev=3,
            b=mag_cal_data['b'], D=mag_cal_data['D'])
        print 'calibrated magnetometer with b, D ='
        print magneto_est.b
        print magneto_est.D
    magneto_est.is_stateful = False

    gyro_est = RateGyro(rate_noise_std_dev=np.deg2rad(0.02),
        constant_bias=[0,0,0],
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

    if use_mag:
        est_sensors = KalmanSensors([gyro_est, accel_est, magneto_est],
            [[4, 5, 6], [0, 1, 2, 3], [0, 1, 2, 3]], n_system_states,
            [[7, 8, 9], [], []], n_sensor_states,
            lambda x, u: rotation_dynamics(x, u, dt),
            W,
            1)
    else:
        est_sensors = KalmanSensors([gyro_est, accel_est],
            [[4, 5, 6], [0, 1, 2, 3]], n_system_states,
            [[7, 8, 9], []], n_sensor_states,
            lambda x, u: rotation_dynamics(x, u, dt),
            W,
            1)

    # Initial state estimate. Set the sensor bias states
    # to an initial estimate of zero.
    q_init = quat.axangle2quat([0,0,1], np.deg2rad(-45))
    body_rate_init = [0., 0., 0.]
    x_est_init = np.concatenate(( q_init, body_rate_init,
        np.zeros(n_sensor_states)))

    # Initial estimate covariance.
    # The std. dev. uncertainty of the initial system state
    # estimate
    yaw_init_std_dev = np.deg2rad(10.)
    pitch_init_std_dev = np.deg2rad(1.0)
    roll_init_std_dev = np.deg2rad(1.0)
    body_rate_init_std_dev = np.deg2rad([0.1, 0.1, 0.1])
    system_state_init_std_dev = np.hstack((
        [roll_init_std_dev, pitch_init_std_dev, yaw_init_std_dev],
        body_rate_init_std_dev
        ))
    # The std. dev. uncertainty of the sensor bias states.
    # [units: radian second**-1]
    gyro_bias_std_dev = np.deg2rad([5., 5., 5.])
    sensor_state_init_std_dev = gyro_bias_std_dev
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

    return (est, est_sensors)


def run(est, meas_source, n_steps, dt, t_traj=None, y_traj=None, use_mag=False,
    sim_type='static'):
    # Set up the measurement source
    if meas_source == 'sim':
        # Time
        t_traj = np.linspace(0, (n_steps - 1)*dt, n_steps)

        # Create the sensors for the simulation (unknown, random bias parameters). 
        gyro_sim = RateGyro(dt=dt)
        magneto_sim = Magnetometer(b=[10., 10., 10.], D=0.1*np.ones((3,3)))
        magneto_sim.is_stateful = False
        accel_sim = Accelerometer(a_bias_sensor=[0, 0, 0])
        accel_sim.is_stateful = False
        # Number of system states.
        n_system_states = 7

        if use_mag:
            sim_sensors = KalmanSensors([gyro_sim, accel_sim, magneto_sim],
                [[4, 5, 6], [0, 1, 2, 3], [0, 1, 2, 3]], n_system_states)
        else:
            sim_sensors = KalmanSensors([gyro_sim, accel_sim],
                [[4, 5, 6], [0, 1, 2, 3]], n_system_states)

        # Initial true state
        q_init = quat.axangle2quat([0,0,1], np.deg2rad(-45))
        body_rate_init = [0., 0., 0.]
        x_init = np.concatenate(( q_init, body_rate_init))

        # True state trajectory
        x_traj = np.zeros((n_steps, len(x_init)))
        x_traj[0] = x_init

        # Control trajectory
        u_traj = np.zeros((n_steps, 3))
        if sim_type == 'static':
            # Keep zero control
            pass
        elif sim_type == 'xyz90':
            t_x = 5.
            t_y = 20.
            t_z = 35.
            dur = 5.
            u = np.deg2rad(90) / dur / dt
            u_traj[int(t_x/dt)] = u * np.array([1,0,0])
            u_traj[int((t_x + dur)/dt)] = -2*u * np.array([1,0,0])
            u_traj[int((t_x + 2*dur)/dt)] = u * np.array([1,0,0])

            u_traj[int(t_y/dt)] = u * np.array([0,1,0])
            u_traj[int((t_y + dur)/dt)] = -2*u * np.array([0,1,0])
            u_traj[int((t_y + 2*dur)/dt)] = u * np.array([0,1,0])

            u_traj[int(t_z/dt)] = u * np.array([0,0,1])
            u_traj[int((t_z + dur)/dt)] = -2*u * np.array([0,0,1])
            u_traj[int((t_z + 2*dur)/dt)] = u * np.array([0,0,1])

        elif sim_type == 'tumble':
            for i in xrange(n_steps):
                if np.random.rand() < 2e-2:
                    u_traj[i] = 1000 * dt**2 * (np.random.rand(3) - 0.5)
        else:
            raise ValueError

        # for i in xrange(100, 110):
        #     u_traj[i] = np.deg2rad([10.0, 0, 0])

        # Sensor bias trajectories
        gyro_bias_traj = np.zeros((n_steps, 3))

        # Measurement trajectory
        y_traj = np.zeros((n_steps,
                len(sim_sensors.measurement_function(x_init))))

        

    elif meas_source == 'pickle':
        assert t_traj is not None
        assert y_traj is not None
    else:
        raise ValueError

    # Create trajectories for storing data
    x_est_traj = np.zeros((n_steps, len(est.x_est)))
    x_est_traj[0] = est.x_est
    Q_traj = np.zeros((n_steps, len(est.x_est)-1, len(est.x_est)-1))
    Q_traj[0] = est.Q

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
        elif meas_source == 'pickle':
            pass
        else:
            raise ValueError
        # Update filter estimate.
        est.propagate_dynamics(np.array([0, 0, 0]))
        est.update_measurement(y_traj[i])
        # Record the estimates.
        x_est_traj[i] = est.x_est
        Q_traj[i] = est.Q        

    print 'Final state est = '
    print est.x_est
    print 'Final estimate covariance Q = '
    print est.Q

    if meas_source == 'sim':
        return (t_traj, x_est_traj, Q_traj, y_traj, x_traj, gyro_bias_traj)
    elif meas_source == 'pickle':
        return (t_traj, x_est_traj, Q_traj, y_traj, None, None)
    else:
        raise ValueError


def plot_traj(t_traj, x_est_traj, Q_traj, y_traj, x_traj, gyro_bias_traj, use_mag, est_sensors):
    # Radian to degree conversion
    r2d = 180.0 / np.pi

    ax = plt.subplot(3, 2, 1)
    colors = ['black', 'red', 'green', 'blue']
    for i in xrange(4):
        if x_traj is not None:
            plt.plot(t_traj, x_traj[:, i], color=colors[i], linestyle='-',
                label='q[{:d}] true'.format(i))
        plt.plot(t_traj, x_est_traj[:, i], color=colors[i], linestyle='--',
            label='q[{:d}] est'.format(i))
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
        if x_traj is not None:
            plt.plot(t_traj, x_traj[:, i+4] * r2d, color=colors[i+1], linestyle='-',
                label='w[{:d}] true'.format(i))
        plot_single_state_vs_time(ax2, t_traj, x_est_traj * r2d, Q_traj_padded * r2d**2, i+4,
            color=colors[i+1], label='w[{:d}] est'.format(i),
            linestyle='--')
    plt.xlabel('Time [s]')
    plt.ylabel('Angular rate [deg / s]')
    plt.legend(framealpha=0.5)

    ax3 = plt.subplot(3, 2, 3, sharex=ax)
    if use_mag:
        # Calibrated mag data.
        mag_data_post_cal = calibrate_data(y_traj[:, 6:9], est_sensors.sensors[2].b,
            est_sensors.sensors[2].D)
        for i in xrange(3):
            plt.plot(t_traj, y_traj[:, i + 6], color=colors[i+1],
                label='mag[{:d}] raw'.format(i))
            plt.plot(t_traj, mag_data_post_cal[:, i], color=colors[i+1],
                label='mag[{:d}] calib'.format(i),
                linestyle='--')
        plt.xlabel('Time [s]')
        plt.ylabel('Mag Field [uT]')
        plt.legend(framealpha=0.5)

    ax4 = plt.subplot(3, 2, 4, sharex=ax)
    for i in [0, 1, 2]:
        if gyro_bias_traj is not None:
            plt.plot(t_traj, gyro_bias_traj[:, i], color=colors[i+1], linestyle='-',
                label='b[{:d}] true'.format(i))
        plot_single_state_vs_time(ax4, t_traj, x_est_traj * r2d, Q_traj_padded * r2d**2, i+7,
            color=colors[i+1], label='b[{:d}] est'.format(i),
            linestyle='--')
    plt.xlabel('Time [s]')
    plt.ylabel('Gyro bias [deg / s]')
    plt.legend(framealpha=0.5)

    ax5 = plt.subplot(3, 2, 5, sharex=ax)
    for i in xrange(3):
        plt.plot(t_traj, y_traj[:, i + 3], color=colors[i+1],
            label='accel[{:d}]'.format(i))
    plt.xlabel('Time [s]')
    plt.ylabel('Accel [m / s**2]')
    plt.legend(framealpha=0.5)

    ax6 = plt.subplot(3, 2, 6)
    angle_names = ['yaw', 'pitch', 'roll']
    angle_colors = ['blue', 'green', 'red']
    # Find yaw, pitch and roll
    ypr_est = np.zeros((len(t_traj), 3))
    for i in xrange(len(t_traj)):
        ypr_est[i] = tb.quat2euler(x_est_traj[i, :4])
    if x_traj is not None:
        ypr_true = np.zeros((len(t_traj), 3))
        for i in xrange(len(t_traj)):
            ypr_true[i] = tb.quat2euler(x_traj[i, :4])
    # Find yaw, pitch, and roll covariance
    Q_ypr = np.zeros((len(t_traj), 3, 3))
    # yaw about z
    Q_ypr[:,0,0] = Q_traj[:,2,2]
    # pitch about y
    Q_ypr[:,1,1] = Q_traj[:,1,1]
    # roll about x
    Q_ypr[:,2,2] = Q_traj[:,0,0]
    for i in xrange(3):
        if x_traj is not None:
            plt.plot(t_traj, ypr_true[:,i] * r2d, 
                color=angle_colors[i], label=angle_names[i] + ' true')
        plot_single_state_vs_time(ax6, t_traj, ypr_est * r2d, Q_ypr * r2d**2,
            i, color=angle_colors[i], label=angle_names[i] + ' est',
            linestyle='--')
    plt.xlabel('Time [s]')
    plt.ylabel('Attitude [deg]')
    plt.legend(framealpha=0.5)


def main(args):
    np.set_printoptions(precision=3)

    # Parse measurement file if needed
    if args.meas_source == 'pickle':
        with open(args.pkl_file, 'rb') as f:
            data = pickle.load(f)
        t_traj = data['time']
        n_steps = len(t_traj)
        t_diff = np.diff(t_traj)
        dt = np.mean(t_diff)
        print 'Time step = {:.3f} +/- {:.3f} second'.format(
            dt, np.std(t_diff))
        # MPU data is in g, conver to meter second**-1
        data['accel_data'] = data['accel_data'] * 9.81
        # MPU gyro data is in degree second**-1, convert to radian second**-1
        data['gyro_data'] = np.deg2rad(data['gyro_data'])
        if args.use_mag:
            y_traj = np.array([np.hstack((w, a, h)) for (w, a, h) in \
                zip(data['gyro_data'], data['accel_data'], data['mag_data'])])
        else:
            y_traj = np.array([np.hstack((w, a)) for (w, a) in \
                zip(data['gyro_data'], data['accel_data'])])
        # Correct length pf y_traj if we got more sensor readings than
        # time steps. This happens based on when the 
        # logger was interrupted.
        if y_traj.shape[0] > n_steps:
            y_traj = y_traj[:n_steps]
        assert y_traj.shape[0] == n_steps
    elif args.meas_source == 'sim':
        dt=0.1
        n_steps = 500
        t_traj = None
        y_traj = None
    else:
        raise ValueError
    
    # Create the estimator
    est, est_sensors = create_estimator(dt, args.use_mag, args.mag_cal)

    # Run the estimator
    t_traj, x_est_traj, Q_traj, y_traj, x_traj, gyro_bias_traj = \
        run(est, args.meas_source, n_steps, dt, t_traj, y_traj,
            args.use_mag, args.sim_type)

    # Plot the results
    plt.figure(figsize=(4*4, 3*4))
    plot_traj(t_traj, x_est_traj, Q_traj, y_traj, x_traj, gyro_bias_traj,
        args.use_mag,est_sensors)
    name = ''
    if args.sim_type is not None:
        name = args.sim_type
    if args.pkl_file is not None:
        if 'static' in args.pkl_file:
            name = 'static'
        if 'xyz90' in args.pkl_file:
            name = 'xyz90'
    plt.savefig('est_result_{:s}_{:s}.png'.format(args.meas_source, name))
    plt.savefig('est_result_{:s}_{:s}.pdf'.format(args.meas_source, name))
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the state estimator.')
    parser.add_argument('--meas_source', type=str, choices=['sim', 'pickle'],
        required=True)
    parser.add_argument('--pkl_file', type=str, required=False,
        help='The pickle file containing the measurement data. Required if --meas_source is "pickle".')
    parser.add_argument('--sim_type', type=str, choices=['static', 'xyz90', 'tumble'],
        required=False,
        help='Which simulation to run. Required if --meas_source is "sim".')
    parser.add_argument('--use_mag', help='Use magnetometer data',
                    action='store_true')
    parser.add_argument('--mag_cal', type=str, required=False,
        help='Pickle file containing the magnetometer calibration parameters.')
    args = parser.parse_args()
    if args.meas_source == 'pickle' and args.pkl_file is None:
        parser.error('--pkl_file is required if --meas_source is "pickle"')
    if args.meas_source == 'sim' and args.sim_type is None:
        parser.error('--sim_type is required if --meas_source is "sim"')
    main(args)