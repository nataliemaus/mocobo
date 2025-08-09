

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_2d_rover(ax, roverdomain):
    # Create a Rectangle  
    delta = roverdomain.obstacle_delta
    for i in range(len(roverdomain.obstacle_l)):
        rect = patches.Rectangle(
            (roverdomain.obstacle_l[i, 0], roverdomain.obstacle_l[i, 1]),
            delta,
            delta,
            linewidth=1,
            edgecolor="darkred",
            facecolor="darkred",
        )
        ax.add_patch(rect) 


def plot_rover(domain, x, save_plot_path, plot_title=None, trajectory_color=None, linewidth=None):
    _, ax = plt.subplots(1, 1, figsize=(8, 6))
    plot_2d_rover(ax, domain)
    trajectory = domain.trajectory(x.cpu().numpy())
    if trajectory_color is None:
        trajectory_color = "blue"
    if linewidth is None:
        linewidth = 5.0 
    plt.plot(trajectory[:, 0], trajectory[:, 1], color=trajectory_color, linewidth=linewidth)
    plt.xlim([0.0, 1.0])                  
    plt.ylim([0.0, 1.0])
    plt.plot(0.05, 0.05, ".g", ms=25) # 16)  # Start
    plt.plot(0.95, 0.95, ".r", ms=25) # 16)  # Goal
    plt.grid(True)
    # Add grid lines
    ax.grid(True)

    # Turn off axis labels (numbers) but keep ticks (and thus the grid lines)
    ax.set_xticklabels([])  # Remove x-axis tick labels
    ax.set_yticklabels([])  # Remove y-axis tick labels
    if plot_title is not None:
        plt.title(plot_title)
    plt.savefig(save_plot_path, bbox_inches='tight', pad_inches=0)
