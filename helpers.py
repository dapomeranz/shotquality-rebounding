import matplotlib as mpl
import matplotlib.pyplot as plt


def plot_basketball_court():
    color = "black"
    fig = plt.figure(figsize=(4, 3.76))
    ax = fig.add_axes([0, 0, 1, 1])
    # Short corner 3PT lines
    ax.plot([-220, -220], [0, 140], linewidth=2, color=color)
    ax.plot([220, 220], [0, 140], linewidth=2, color=color)

    # 3PT Arc
    ax.add_artist(
        mpl.patches.Arc(
            (0, 140),
            440,
            315,
            theta1=0,
            theta2=180,
            facecolor="none",
            edgecolor=color,
            lw=2,
        )
    )

    # Lane and Key
    ax.plot([-80, -80], [0, 190], linewidth=2, color=color)
    ax.plot([80, 80], [0, 190], linewidth=2, color=color)
    ax.plot([-60, -60], [0, 190], linewidth=2, color=color)
    ax.plot([60, 60], [0, 190], linewidth=2, color=color)
    ax.plot([-80, 80], [190, 190], linewidth=2, color=color)
    ax.add_artist(
        mpl.patches.Circle((0, 190), 60, facecolor="none", edgecolor=color, lw=2)
    )

    # Rim
    ax.add_artist(
        mpl.patches.Circle((0, 60), 15, facecolor="none", edgecolor=color, lw=2)
    )

    # Backboard
    ax.plot([-30, 30], [40, 40], linewidth=2, color=color)

    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Set axis limits
    ax.set_xlim(-250, 250)
    ax.set_ylim(0, 470)
