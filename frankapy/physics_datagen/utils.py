import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.signal import sawtooth

# ============================================= plot ============================================= #

def plt_sim_real_comparison(sim_time, 
                            sim_pos, 
                            sim_pos_ctrl_signal,
                            real_time=None, 
                            real_pos=None, 
                            output_filename="comparison.png"
                            ):

    plt.figure(figsize=(8, 5))
    if real_time is not None and real_pos is not None:
        plt.plot(real_time, real_pos, label='Real Position', color='r')

    plt.plot(sim_time, sim_pos, label='Simulation Position', color='g')
    plt.plot(sim_time, sim_pos_ctrl_signal, label='Sim Pos Ctrl Signal', color='k', linestyle='--')
    plt.title('After System Identification')
    plt.xlabel('Time (s)')
    plt.ylabel('Position')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    json_filename = os.path.splitext(output_filename)[0] + ".json"
    data = {
        "sim_time": sim_time.tolist(),
        "sim_pos": sim_pos.tolist(),
        "sim_pos_ctrl_signal": sim_pos_ctrl_signal.tolist(),
    }

    if real_time is not None and real_pos is not None:
        data["real_time"] = real_time.tolist()
        data["real_pos"] = real_pos.tolist()

    with open(json_filename, "w") as json_file:
        json.dump(data, json_file, indent=4)

def plot_trajectory(t_history, actual_position, goal_position, actual_vel, filename, output, show_plot):
        # if show_plot:
        #     plt.ion()
        if filename != None:
            fig, axs = plt.subplots(2, 1, figsize=(5, 6))  # 2 rows, 1 column
            axs[0].plot(t_history, actual_position, label='Actual Position', color='r')
            axs[0].plot(t_history, goal_position, label='Goal Position (Control Signal)', color='b', linestyle='--', linewidth=1)

            axs[0].set_title('Position Over Time')
            axs[0].set_xlabel('Time (s)')
            axs[0].set_ylabel('Position')
            axs[0].legend()

            # Plot for velocity on the second subplot
            axs[1].plot(t_history, actual_vel, label='Velocity', color='k')
            axs[1].set_title('Velocity Over Time')
            axs[1].set_xlabel('Time (s)')
            axs[1].set_ylabel('Velocity')
            axs[1].legend()
            plt.tight_layout()
            if show_plot:
                plt.show()

        data_file_root = output
        os.makedirs(data_file_root, exist_ok=True)
        image_filename = f'{filename}.jpeg'
        fig.savefig(os.path.join(data_file_root, image_filename))
        print(f"Image saved to {image_filename}")


def plot_loss_surface(kp_kv_div, kv, loss_values, joint_id, savename, loss_range=None):
    """
    Plot a 3D surface of the loss values, with optional filtering by loss range.
    
    Parameters:
        kp (2D array): Kp grid values (x-coordinates).
        kv (2D array): Kv grid values (y-coordinates).
        loss_values (2D array): Loss values (z-coordinates).
        joint_id (int): Joint ID for labeling the plot.
        savename (str): Directory to save the plot.
        loss_range (tuple, optional): A tuple (min_loss, max_loss) to filter loss values.
    """
    # Filter loss values if a range is provided
    # if loss_range is not None:
    #     min_loss, max_loss = loss_range
    #     mask = (loss_values >= min_loss) & (loss_values <= max_loss)
    #     loss_values = np.where(mask, loss_values, np.nan)  # Mask out values outside the range
    loss_values = np.clip(loss_values, loss_range[0], loss_range[1])
    loss_values = loss_values.reshape(kp_kv_div.shape)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Surface plot for grid data
    surf = ax.plot_surface(kp_kv_div, kv, loss_values, cmap='viridis', edgecolor='none')
    
    ax.set_title(f"Loss Surface for Joint {joint_id}")
    ax.set_xlabel("kp_kv_div")
    ax.set_ylabel("Kv")
    ax.set_zlabel("Loss")
    fig.colorbar(surf, shrink=0.5, aspect=10)
    
    # Save the plot
    # plt.show()
    plt.savefig(f"{savename}/joint_{joint_id}_loss_surface.png")
    plt.close()
    
# ============================================= traj ============================================= #

def sine_traj(lower, upper, num_wave=3, sample_rate=60, min_freq=0.5, max_freq=2.0):
    t = np.linspace(0, num_wave, int(num_wave * sample_rate))
    freq = np.linspace(min_freq, max_freq, len(t))
    sine_wave = np.sin(2 * np.pi * freq * t)
    goal_pos_list = lower + (sine_wave + 1) * (upper - lower) / 2
    return goal_pos_list

def step_traj(lower, upper, length=80):
    quad_length = length // 4
    goal_pos_list =  [lower] * (quad_length) +[(upper+lower)/2.0] * quad_length +[upper] * quad_length + [(upper+lower)/2.0] * quad_length
    return goal_pos_list

def step_sine_traj(lower, upper, length=80, num_wave=3, sample_rate=60, min_freq=0.5, max_freq=1.5):
    goal_pos_list1 = step_traj(lower, upper, length)
    goal_pos_list2 = sine_traj(lower, upper, num_wave, sample_rate, min_freq, max_freq)
    return np.concatenate([goal_pos_list1, goal_pos_list2])

def triangle_traj(lower, upper, num_wave=2, sample_rate=60, freq=0.5):
    t = np.linspace(0, num_wave, int(num_wave * sample_rate))
    triangle_wave = sawtooth(2 * np.pi * freq * t, width=0.5)
    goal_pos_list = lower + (triangle_wave + 1) * (upper - lower) / 2
    return goal_pos_list

def get_traj(trajectory_type, lower, upper, **kwargs):
    if trajectory_type == "sine":
        return sine_traj(lower, upper, **kwargs)
    elif trajectory_type == "step":
        return step_traj(lower, upper, **kwargs)
    elif trajectory_type == "step_sine":
        return step_sine_traj(lower, upper, **kwargs)
    elif trajectory_type == "triangle":
        return triangle_traj(lower, upper, **kwargs)
    elif trajectory_type == "combined":
        return np.concatenate((step_traj(lower, upper, **kwargs), sine_traj(lower, upper, **kwargs)), axis=0)
    else:
        raise NotImplementedError(f"Unknown type {trajectory_type}")