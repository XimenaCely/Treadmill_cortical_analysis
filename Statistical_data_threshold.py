
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

rows = [
    # Participant, Session, Condition, Mean, SD, Median, Above_1_4, Below_1_4
    ("P01", 1, "Baseline",       1.371, None, 1.385, 48.2, 51.8),
    ("P01", 1, "Active walking", 0.687, None, 0.623,  3.4, 96.6),
    ("P01", 1, "KMI + PF",       1.106, None, 1.136, 29.3, 70.7),
    ("P01", 2, "Baseline",       1.511, None, 1.535, 73.5, 26.5),
    ("P01", 2, "Active walking", 1.427, None, 1.438, 56.5, 43.5),
    ("P01", 2, "KMI + PF",       1.443, None, 1.458, 59.3, 40.7),
    ("P01", 3, "Baseline",       1.446, None, 1.476, 63.3, 36.7),
    ("P01", 3, "Active walking", 1.430, None, 1.441, 55.9, 44.1),
    ("P01", 3, "KMI + PF",       1.460, None, 1.475, 63.6, 36.4),

    ("P02", 1, "Baseline",       1.508, None, 1.542, 73.8, 26.2),
    ("P02", 1, "Active walking", 1.432, None, 1.471, 61.3, 38.7),
    ("P02", 1, "KMI + PF",       1.431, None, 1.450, 58.3, 41.7),
    ("P02", 2, "Baseline",       1.330, None, 1.342, 40.7, 59.3),
    ("P02", 2, "Active walking", 1.461, None, 1.485, 62.2, 37.8),
    ("P02", 2, "KMI + PF",       1.469, None, 1.498, 65.4, 34.6),
    ("P02", 3, "Baseline",       1.194, None, 1.201, 20.3, 79.7),
    ("P02", 3, "Active walking", 1.313, None, 1.331, 39.0, 61.0),
    ("P02", 3, "KMI + PF",       1.322, None, 1.337, 39.3, 60.7),

    ("P03", 1, "Baseline",       1.411, None, 1.425, 54.6, 45.4),
    ("P03", 1, "Active walking", 1.630, None, 1.637, 83.8, 16.2),
    ("P03", 1, "KMI + PF",       1.633, None, 1.651, 84.0, 16.0),
    ("P03", 2, "Baseline",       1.262, None, 1.271, 30.7, 69.3),
    ("P03", 2, "Active walking", 1.219, None, 1.233, 25.7, 74.3),
    ("P03", 2, "KMI + PF",       1.219, None, 1.231, 23.8, 76.2),
    ("P03", 3, "Baseline",       1.424, None, 1.440, 55.9, 44.1),
    ("P03", 3, "Active walking", 1.352, None, 1.366, 43.1, 56.9),
    ("P03", 3, "KMI + PF",       1.409, None, 1.414, 52.1, 47.9),

    ("P04", 1, "Baseline",       1.368, None, 1.376, 45.6, 54.4),
    ("P04", 1, "Active walking", 1.317, None, 1.334, 41.2, 58.8),
    ("P04", 1, "KMI + PF",       1.362, None, 1.382, 47.7, 52.3),
    ("P04", 2, "Baseline",       1.376, None, 1.369, 46.0, 54.0),
    ("P04", 2, "Active walking", 1.300, None, 1.312, 37.5, 62.5),
    ("P04", 2, "KMI + PF",       1.286, None, 1.302, 33.6, 66.4),
    ("P04", 3, "Baseline",       1.308, None, 1.322, 36.9, 63.1),
    ("P04", 3, "Active walking", 1.204, None, 1.217, 26.6, 73.4),
    ("P04", 3, "KMI + PF",       1.238, None, 1.237, 28.6, 71.4),

    ("P05", 1, "Baseline",       1.685, None, 1.721, 88.2, 11.8),
    ("P05", 1, "Active walking", 1.647, None, 1.673, 85.8, 14.2),
    ("P05", 1, "KMI + PF",       1.664, None, 1.709, 84.1, 15.9),
    ("P05", 2, "Baseline",       1.539, None, 1.560, 74.2, 25.8),
    ("P05", 2, "Active walking", 1.516, None, 1.538, 71.7, 28.3),
    ("P05", 2, "KMI + PF",       1.505, None, 1.516, 70.6, 29.4),
    ("P05", 3, "Baseline",       1.473, None, 1.485, 63.0, 37.0),
    ("P05", 3, "Active walking", 1.437, None, 1.445, 57.7, 42.3),
    ("P05", 3, "KMI + PF",       1.487, None, 1.502, 65.9, 34.1),
]

columns = ["Participant", "Session", "Condition", "Mean", "SD", "Median", "Above_1_4", "Below_1_4",]
df = pd.DataFrame(rows, columns=columns)

sns.set_theme(style="whitegrid", context="notebook")
# Global font settings
plt.rcParams.update({
    "font.family": "Palatino Linotype",
    "font.size": 22,
    "axes.labelsize": 22,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 22,
    "figure.titlesize": 22,
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "axes.edgecolor": "black",
    "savefig.dpi": 300,
})

condition_order = ["Baseline", "Active walking", "KMI + PF",]
participant_order = ["P01", "P02", "P03", "P04", "P05"]

palette = {
    "P01": "#08306B",  # Deep navy
    "P02": "#2171B5",  # Strong blue
    "P03": "#00A6D6",  # Cyan-blue
    "P04": "#4169E1",  # Royal blue
    "P05": "#7B2CBF",  # Blue-violet for stronger contrast
}

# Assign one blue to each experimental condition
colors = {
    "Baseline": palette["P01"],
    "Active walking": palette["P03"],
    "KMI + PF": palette["P05"],
}


fig1, axes = plt.subplots(1, 3, figsize=(24, 9), sharey=True)
for ax, session in zip(axes, [1, 2, 3]):
    subset = df[df["Session"] == session].copy()
    # Plot individual participant means using participant colors
    for participant in participant_order:
        participant_data = subset[subset["Participant"] == participant]
        ax.scatter([condition_order.index(condition) for condition in participant_data["Condition"]], participant_data["Mean"], color=palette[participant], s=120, alpha=0.9, zorder=3, 
        )

    # Mean across participants for each condition
    group_means = (subset.groupby("Condition")["Mean"].mean().reindex(condition_order)  )
    ax.scatter(
        range(len(condition_order)),
        group_means.values,
        marker="D",
        s=180,
        color="black",
        label="Mean across participants",
        zorder=5,
    )
    
    # Condition labels: smaller and horizontal
    ax.set_xticks( range(len(condition_order)), condition_order, rotation=45, ha="center",  )
    ax.tick_params(axis="x", labelsize=16)

    # Keep the threshold line, but do not label the mean diamonds
    ax.axhline( y=1.4, color="#4D8BBE", linestyle="--", linewidth=2, label="Threshold = 1.4",  )
    legend_elements = [ Line2D( [0], [0], color="black", linestyle="--", linewidth=2, label="Threshold = 1.4" )    ]

    # Add one entry for each participant
    for participant, color in palette.items():
        legend_elements.append(
            Line2D(
                [0], [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                markersize=9,
                label=participant
            )
        )

    axes[-1].legend(
        handles=legend_elements,
        loc="lower right",
        fontsize=12,
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=1,
        borderpad=0.6,
        labelspacing=0.5,
        handletextpad=0.6
    )


    # Shared legend: threshold only
    legend_items = [
        Line2D(
            [0], [0],
            color="#4D8BBE",
            linestyle="--",
            linewidth=2,
            label="Threshold = 1.4",
        )
    ]

    fig1.legend(
        handles=legend_items,
        loc="lower center",
        ncol=1,
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
        fontsize=22,
    )

    # No titles
    ax.set_title("")
    fig1.suptitle("")

    plt.tight_layout(rect=[0, 0.08, 1, 1])

plt.show()

# HEATMAP: PERCENTAGE BELOW THRESHOLD 1.4
df["Session_Condition"] = ( "S" + df["Session"].astype(str) + " - " + df["Condition"])

heatmap_column_order = [ f"S{session} - {condition}" for session in [1, 2, 3] for condition in condition_order]
heatmap_data = df.pivot( index="Participant", columns="Session_Condition", values="Below_1_4")
heatmap_data = heatmap_data.reindex( index=participant_order, columns=heatmap_column_order)

# Average across participants
average_row = heatmap_data.mean(axis=0)
heatmap_data.loc["Average"] = average_row

# Custom five-color blue gradient
blue_cmap = LinearSegmentedColormap.from_list(
    "custom_blue_palette",
    [
         "#003366",  # 0%: Dark navy blue
        "#1F5A94",
        "#2F6FA5",
        "#4D8BBE",
        "#6AA4CF",  # 100%: Lightest blue
    ],
)

fig2, ax = plt.subplots(figsize=(24, 10))

sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".1f",
    cmap="Blues",
    vmin=0,
    vmax=100,
    linewidths=0.8,
    linecolor="white",
    annot_kws={
        "fontsize": 22,
        "color": "black",
    },
    cbar_kws={
        "label": "Values below threshold 1.4 (%)",
    },
    ax=ax,
)

ax.set_xlabel("Session and Condition", labelpad=12)
ax.set_ylabel("Participant", labelpad=12)

ax.tick_params(axis="x", labelrotation=40)
ax.tick_params(axis="y", labelrotation=0)

# Match colorbar typography
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=22)
cbar.set_label(
    "Values below threshold 1.4 (%)",
    fontsize=22,
)

plt.tight_layout()

plt.savefig(
    "heatmap_below_threshold_1_4_with_average.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print("\nAverage percentage below threshold 1.4:")
print(average_row.round(1))


# from pathlib import Path

# # Output folder
# output_folder = Path(r"A:\ARTICLES\MDPI 2026\Complete_code")
# output_folder.mkdir(parents=True, exist_ok=True)

# # Save the first plot (dot plot)
# fig1.savefig(
#     output_folder / "participant_means.eps",
#     format="eps",
#     bbox_inches="tight"
# )

# # Save the second plot (heatmap)
# fig2.savefig(
#     output_folder / "percentage_below_threshold.eps",
#     format="eps",
#     bbox_inches="tight"
# )

