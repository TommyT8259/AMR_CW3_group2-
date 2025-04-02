import numpy as np

def controller(state, target_pos, dt):
    # === Controller Gain Initialization ===
    # Base gains (will be auto-tuned)
    Kp_base = np.array([0.4, 0.4, 0.6])
    Ki_base = np.array([0.01, 0.01, 0.01])
    Kd_base = np.array([0.2, 0.2, 0.3])

    Kp_yaw = 1.0
    Kd_yaw = 0.2
    Ki_yaw = 0.01

    # === Unpack state ===
    current_pos = np.array(state[0:3])
    current_yaw = state[5]
    current_vel = np.array(state[6:9])

    # === Position Error (Layer 1) ===
    pos_error = target_pos[0:3] - current_pos

    # === Auto-Tune Gains Based on Error Magnitude ===
    scale = np.clip(np.linalg.norm(pos_error) / 4.0, 0.5, 2.0)
    Kp = Kp_base * scale
    Ki = Ki_base * scale
    Kd = Kd_base * scale

    # === Layer 1: Position PID → Desired Velocity ===
    # Add basic integrator memory
    if not hasattr(controller, "pos_error_integral"):
        controller.pos_error_integral = np.zeros(3)
        controller.prev_pos_error = np.zeros(3)

    controller.pos_error_integral += pos_error * dt
    pos_error_derivative = (pos_error - controller.prev_pos_error) / dt
    controller.prev_pos_error = pos_error

    desired_velocity = (
        Kp * pos_error
        + Ki * controller.pos_error_integral
        + Kd * pos_error_derivative
    )

    # === Layer 2: Velocity PID → Desired Acceleration ===
    vel_error = desired_velocity - current_vel
    if not hasattr(controller, "vel_error_integral"):
        controller.vel_error_integral = np.zeros(3)
        controller.prev_vel_error = np.zeros(3)

    controller.vel_error_integral += vel_error * dt
    vel_error_derivative = (vel_error - controller.prev_vel_error) / dt
    controller.prev_vel_error = vel_error

    # Cascade gain for velocity PID
    Kp_vel = np.array([1.0, 1.0, 1.0])
    Ki_vel = np.array([0.05, 0.05, 0.05])
    Kd_vel = np.array([0.2, 0.2, 0.3])

    desired_accel = (
        Kp_vel * vel_error
        + Ki_vel * controller.vel_error_integral
        + Kd_vel * vel_error_derivative
    )

    # === Layer 3: Acceleration to Final Velocity Output ===
    # For simulation purposes we’ll treat this as final command (i.e. skipping thrust level)
    output_velocity = desired_accel

    # === Yaw Control (with basic PID) ===
    yaw_error = target_pos[3] - current_yaw
    yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi  # Normalize

    if not hasattr(controller, "yaw_error_integral"):
        controller.yaw_error_integral = 0
        controller.prev_yaw_error = 0

    controller.yaw_error_integral += yaw_error * dt
    yaw_error_derivative = (yaw_error - controller.prev_yaw_error) / dt
    controller.prev_yaw_error = yaw_error

    yaw_rate_setpoint = (
        Kp_yaw * yaw_error
        + Ki_yaw * controller.yaw_error_integral
        + Kd_yaw * yaw_error_derivative
    )

    # === Safety: Clip output velocities ===
    output_velocity = np.clip(output_velocity, -1, 1)
    yaw_rate_setpoint = np.clip(yaw_rate_setpoint, -1.74533, 1.74533)

    output = (
        output_velocity[0],
        output_velocity[1],
        output_velocity[2],
        yaw_rate_setpoint
    )

    return output
