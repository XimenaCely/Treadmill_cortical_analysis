import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Palatino Linotype", "Palatino", "DejaVu Serif"],
    "font.size": 22,
})

participants = {
    "P01": {    
        "FAC": [1, 1],
        "NIHSS": [10, 7],
        "FIM": [74, 90],
        "FMA": [13, 16],
    },
    "P02": {   
        "FAC": [1, 3],
        "NIHSS": [10, 8],
        "FIM": [76, 86],
        "FMA": [24, 28],
    },
    "P03": {    
        "FAC": [1, 1],
        "NIHSS": [14, 8],
        "FIM": [53, 70],
        "FMA": [22, 25],
    },
    "P04": {    
            "FAC": [3, 4],
            "NIHSS": [6, 3],
            "FIM": [94, 113],
            "FMA": [25, 28],
        },
    "P05": {    
            "FAC": [4, 5],     
            "NIHSS": [8, 5],
            "FIM": [95, 105],
            "FMA": [22, 22],
        },
}

colors = {
    "P01": "#003366",  # Dark navy blue
    "P02": "#1F5A94",  # Deep blue
    "P03": "#2F6FA5",  # Medium blue
    "P04": "#4D8BBE",  # Light medium blue
    "P05": "#6AA4CF",  # Light blue
}

# Participant-specific markers:
# P01 = triangle, P02 = circle, P03 = rectangle
markers = {
    "P01": "^",
    "P02": "o",
    "P03": "s",
    "P04": "D",
    "P05": "P",
}

metrics = [
    ("FAC", "FAC score", (0.8, 5.4)),
    ("NIHSS", "NIHSS score", (2.5, 15.5)),
    ("FIM", "FIM score", (48, 115)),
    ("FMA", "FMA motor score", (10, 31)),
]

fig, axes = plt.subplots(1, 4, figsize=(24, 6.5))

# The left and right positions represent pre and post, respectively.
x = [0, 1]

participant_offsets = {
    "P01": -0.045,
    "P02": 0.0,
    "P03": 0.045,
    "P04": -0.045,
    "P05": 0.0,
}

for ax, (metric, ylabel, ylim) in zip(axes, metrics):
    for participant, participant_data in participants.items():
        color = colors[participant]
        marker = markers[participant]
        values = participant_data[metric]

        x_participant = [
            value + participant_offsets[participant] for value in x
        ]

        ax.plot(
            x_participant,
            values,
            linestyle="--",
            linewidth=2.5,
            color=color,
            marker=marker,
            markersize=10,
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=0.8,
            zorder=3,
        )

    ax.set_title(metric, pad=12)

    # Retain the two x positions without displaying Pre/Post labels.
    ax.set_xticks(x)
    ax.set_xticklabels([])
    ax.tick_params(axis="x", length=0)

    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

participant_handles = [
    Line2D(
        [0],
        [0],
        color=colors[participant],
        linestyle="--",
        marker=markers[participant],
        markerfacecolor=colors[participant],
        markeredgecolor="black",
        markersize=10,
        linewidth=2.5,
        label=participant,
    )
    for participant in participants
]

fig.legend(
    handles=participant_handles,
    title="Participant",
    loc="lower center",
    bbox_to_anchor=(0.5, -0.02),
    ncol=3,
    frameon=False,
    fontsize=22,
    title_fontsize=22,
)

fig.tight_layout(rect=(0, 0.16, 1, 1), w_pad=2.2)

output_path = (
    r"A:\\ARTICLES\\MDPI 2026\\Complete_code\\participant_outcomes.png"
)

fig.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)

plt.show()