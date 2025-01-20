# pylint: disable=C0103,C0111,C0301
import numpy as np
import casadi as ca
from typing import Dict, List, Tuple


from config import *
from toolbox import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec


def create_visualization(eval_result, total_time):
    """
    Create an animated visualization of the trajectory optimization.

    Args:
        eval_result: Dictionary containing 'positions', 'velocities', 'accelerates' from eval_traj
        total_time: Total time duration for the trajectory
    """
    # Convert casadi DM objects to numpy arrays and reshape
    pos = np.array(eval_result['positions'])
    vel = np.array(eval_result['velocities'])
    # vel = np.array(eval_result['jerks'])
    acc = np.array(eval_result['accelerates'])
    # acc = np.array(eval_result['snaps'])
    cur = np.array(eval_result['curvatures_sq'])

    # Reshape arrays from (n,2) to (2,n)
    n_points = pos.shape[0]
    pos = pos.reshape(n_points, 2).T
    vel = vel.reshape(n_points, 2).T
    acc = acc.reshape(n_points, 2).T

    # Create time vector
    time = np.linspace(0, total_time, n_points)

    # Calculate derived quantities
    # speed_magnitude = np.max(np.abs(vel), axis=0)
    speed_magnitude = np.sum(vel**2, axis=0)
    # accel_magnitude = np.max(np.abs(acc), axis=0)
    accel_magnitude = np.sum(acc**2, axis=0)

    # Create figure and subplots
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(3, 2, figure=fig)

    # Trajectory subplot
    ax_traj = fig.add_subplot(gs[:, 0])
    ax_traj.grid(True)
    ax_traj.set_aspect('equal')
    ax_traj.set_xlabel('X Position')
    ax_traj.set_ylabel('Y Position')
    ax_traj.set_title('Motion Trajectory')

    # Set axis limits with some padding
    x_min, x_max = pos[0].min(), pos[0].max()
    y_min, y_max = pos[1].min(), pos[1].max()
    padding = 0.2 * max(x_max - x_min, y_max - y_min)
    ax_traj.set_xlim(x_min - padding, x_max + padding)
    ax_traj.set_ylim(y_min - padding, y_max + padding)

    # Plot full trajectory
    ax_traj.plot(pos[0, :], pos[1, :], 'b-', alpha=0.3, label='Trajectory')

    # Create moving objects
    particle, = ax_traj.plot([], [], 'bo', markersize=10, label='Particle')
    vel_arrow = ax_traj.quiver([], [], [], [], color='r', scale=20, label='Velocity')
    acc_arrow = ax_traj.quiver([], [], [], [], color='g', scale=20, label='Acceleration')
    ax_traj.legend()

    # Speed subplot
    ax_speed = fig.add_subplot(gs[0, 1])
    ax_speed.grid(True)
    ax_speed.set_ylabel('Speed Magnitude')
    ax_speed.set_title('Speed vs Time')
    ax_speed.plot(time, speed_magnitude, 'b-', alpha=0.3)
    speed_point, = ax_speed.plot([], [], 'r.', markersize=10)
    speed_line = ax_speed.axvline(x=time[0], color='k', linestyle='--')

    # Acceleration subplot
    ax_accel = fig.add_subplot(gs[1, 1])
    ax_accel.grid(True)
    ax_accel.set_ylabel('Acceleration Magnitude')
    ax_accel.plot(time, accel_magnitude, 'b-', alpha=0.3)
    accel_point, = ax_accel.plot([], [], 'g.', markersize=10)
    accel_line = ax_accel.axvline(x=time[0], color='k', linestyle='--')

    # Curvature subplot
    ax_curv = fig.add_subplot(gs[2, 1])
    ax_curv.grid(True)
    ax_curv.set_xlabel('Time (s)')
    ax_curv.set_ylabel('Curvature')
    ax_curv.plot(time, cur, 'b-', alpha=0.3)
    curv_point, = ax_curv.plot([], [], 'g.', markersize=10)
    curv_line = ax_curv.axvline(x=time[0], color='k', linestyle='--')

    def init():
        particle.set_data([], [])
        speed_point.set_data([], [])
        accel_point.set_data([], [])
        curv_point.set_data([], [])
        return particle, speed_point, accel_point, curv_point

    def animate(i):
        # Update particle position - 修正这里
        particle.set_data([pos[0, i]], [pos[1, i]])  # 使用列表包装单个值
        # print(pos[:,i])

        # Update velocity arrow
        vel_arrow.set_offsets(np.column_stack((pos[0, i], pos[1, i])))
        vel_arrow.set_UVC(vel[0, i], vel[1, i])

        # Update acceleration arrow
        acc_arrow.set_offsets(np.column_stack((pos[0, i], pos[1, i])))
        acc_arrow.set_UVC(acc[0, i], acc[1, i])

        # Update speed plot
        speed_point.set_data(time[:i+1], speed_magnitude[:i+1])
        speed_line.set_xdata([time[i], time[i]])

        # Update acceleration plot
        accel_point.set_data(time[:i+1], accel_magnitude[:i+1])
        accel_line.set_xdata([time[i], time[i]])

        # Update curvature plot
        curv_point.set_data(time[:i+1], cur[:i+1])
        curv_line.set_xdata([time[i], time[i]])

        return particle, vel_arrow, acc_arrow, speed_point, accel_point, curv_point

    anim = FuncAnimation(fig, animate, init_func=init, frames=n_points,
                        interval=50, blit=True, repeat=False)

    plt.tight_layout()
    plt.show()




