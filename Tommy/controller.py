# Import any necessary libraries
import numpy as np

max_vel = 8  # m/s
mass = 0.23  # kg (230 grams)

class PID:
    def __init__(self, Kp, Ki, Kd, Kisat):
        # Define variables
        self.Kp = np.array(Kp)  # Proportional gain
        self.Ki = np.array(Ki)  # Integral gain
        self.Kd = np.array(Kd)  # Derivative gain
        self.Kisat = np.array(Kisat)  # Saturation of integral error - to prevent integral windup

        # Initialise Variables
        self.integral_error = np.zeros_like(Kp, dtype=float)
        self.derivative_error = np.zeros_like(Kp)
        self.previous_error = np.zeros_like(Kp)

    # Reset method
    def reset(self):
        self.integral_error = np.zeros_like(self.integral_error)
        self.previous_error = np.zeros_like(self.previous_error)

    # Get PID output
    def getPID_output(self, error, dt):
        # Define the proportional error
        self.proportional_error = error

        # Find the integral error
        # Sum of error and previous error times time-step
        self.integral_error += error * dt

        # Prevent integral wind up by clipping the integral error
        min_integral_error = np.negative(self.Kisat)
        max_integral_error = np.positive(self.Kisat)
        self.integral_error = np.clip(self.integral_error, min_integral_error, max_integral_error)

        # Find the derivative error
        self.derivative_error = (error - self.previous_error) / dt

        # Compute the PID output
        PID_output = (self.Kp * self.proportional_error +
                      self.Ki * self.integral_error +
                      self.Kd * self.derivative_error)

        self.previous_error = error
        return PID_output


# Define Kp, Ki, Kd, Ksat for the class and assigning the class a new name
Kp_jer = 1
Ki_jer = 0.05
Kd_jer = 0.1
Kisat_jer = 0.5

Kp_acc = 3
Ki_acc = 0.05
Kd_acc = 0.4
Kisat_acc = 0.5

Kp_vel = 10
Ki_vel = 0.05
Kd_vel = 0.8
Kisat_vel = 0.5

Kp_pos = 10
Ki_pos = 0.05
Kd_pos = 0.8
Kisat_pos = 0.5

Kp_yaw = 1
Ki_yaw = 0.05
Kd_yaw = 0.1
Kisat_yaw = 0.5

jer_PID = PID(
    Kp=[Kp_jer, Kp_jer, Kp_jer],
    Ki=[Ki_jer, Ki_jer, Ki_jer],
    Kd=[Kd_jer, Kd_jer, Kd_jer],
    Kisat=[Kisat_jer, Kisat_jer, Kisat_jer])

acc_PID = PID(
    Kp=[Kp_acc, Kp_acc, Kp_acc],
    Ki=[Ki_acc, Ki_acc, Ki_acc],
    Kd=[Kd_acc, Kd_acc, Kd_acc],
    Kisat=[Kisat_acc, Kisat_acc, Kisat_acc])

vel_PID = PID(
    Kp=[Kp_vel, Kp_vel, Kp_vel],
    Ki=[Ki_vel, Ki_vel, Ki_vel],
    Kd=[Kd_vel, Kd_vel, Kd_vel],
    Kisat=[Kisat_vel, Kisat_vel, Kisat_vel])

pos_PID = PID(
    Kp=[Kp_pos, Kp_pos, Kp_pos],
    Ki=[Ki_pos, Ki_pos, Ki_pos],
    Kd=[Kd_pos, Kd_pos, Kd_pos],
    Kisat=[Kisat_pos, Kisat_pos, Kisat_pos])

yaw_PID = PID(Kp=Kp_yaw,
              Ki=Ki_yaw,
              Kd=Kd_yaw,
              Kisat=Kisat_yaw)

prev_target_pos = None
vel_global = np.zeros(3)
prev_acc = np.zeros(3)
prev_vel = np.zeros_like(vel_global)
prev_pos = np.zeros(3)


def controller(state, target, dt):
    global prev_target_pos, vel_PID, yaw_PID, max_vel, vel_global, prev_acc, prev_vel, prev_pos

    # Extract the values we need from the state list
    pos = np.array(state[0:3])
    yaw = state[5]

    # Extract the values from the target list
    target_pos = np.array(target[0:3])
    target_yaw = target[3]

    if prev_target_pos is None or not np.allclose(target_pos, prev_target_pos):
        print("[INFO] The target has changed, resetting PID controllers and last target")
        print(f"[INFO] Target changed to: {target}")
        prev_target_pos = np.copy(target_pos)
        vel_PID.reset()
        yaw_PID.reset()
        return (0.0, 0.0, 0.0, 0.0)

    # Calculate the positional error
    pos_error = target_pos - pos

    pos_error_from_PID = pos_PID.getPID_output(pos_error, dt)

    # Get velocity from PID
    vel_global = vel_PID.getPID_output(pos_error_from_PID, dt)

    # Constrain to the drones known maximum velocity

    # calculate the actual velocity
    vel_measured = (pos - prev_pos) / dt

    # calculate the velocity error
    vel_error = vel_global - vel_measured

    # update the pre_pos
    prev_pos = pos

    acc_global = acc_PID.getPID_output(vel_error, dt)
    acc_measured = (vel_global - prev_vel) / dt
    acc_error = acc_global - acc_measured

    jer_correction = jer_PID.getPID_output(acc_error, dt)

    # Update acceleration and velocity
    acc_global += jer_correction * dt
    vel_global += acc_global * dt
    vel_global = np.clip(vel_global, -max_vel, max_vel)

    prev_vel = vel_global
    prev_acc = acc_global
    # Calculate the yaw error
    yaw_error = target_yaw - yaw

    # Such that the yaw error stays between -180 and 180 degrees - ie -pi and +pi
    yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi

    # Get yaw rate from PID
    yaw_rate = yaw_PID.getPID_output(yaw_error, dt)

    # Constrain the drone to a reasonable maximum yaw rate
    max_yaw_rate = np.pi
    yaw_rate = np.clip(yaw_rate, -max_yaw_rate, max_yaw_rate)

    cos_yaw = np.cos(-yaw)
    sin_yaw = np.sin(-yaw)

    vx = vel_global[0]
    vy = vel_global[1]
    vz = vel_global[2]

    v1 = cos_yaw * vx - sin_yaw * vy
    v2 = sin_yaw * vx + cos_yaw * vy
    v3 = vz

    # Print outputs to help debug
    print(f"[STATE] pos: {pos}, yaw: {yaw:.2f} | [TARGET] pos: {target_pos}, yaw_target: {target_yaw:.2f}")
    print(f"[GLOBAL VEL] vx: {vx:.2f}, vy: {vy:.2f}, vz: {vz:.2f}, yaw_rate: {yaw_rate:.2f}")
    print(f"[COMMAND VEL] v1: {v1:.2f}, v2: {v2:.2f}, v3: {v3:.2f}, yaw_rate: {yaw_rate:.2f}")

    # The output of this controller should be:
    # [x velocity, y velocity, z velocity, yaw rate]
    return (v1, v2, v3, yaw_rate)

