
import numpy as np

# --------------------------------------------------
# 1. ERD Hilbert values
# Participant order: P01, P02, P03, P04, P05
# --------------------------------------------------

erd_initial = {
    "P01": [-6.61396587, -3.30117003, -3.71765349, -4.27682492,
            -10, -10, -5.57185106, -10, -10, 2.89608747,
            -8.36826449, -10, 2.81392606, 2.89300733, -10, -10],

    "P02": [-0.02994463, -1.51939731, 1.10839867, 0.43918674,
            5.41857507, 1.21151543, 0.93253485, 0.78567709,
            1.62007138, 3.39352203, 3.99759855, 4.16618569,
            1.4164671, 1.52409164, 3.28248277, 1.39615502],

    "P03": [1.64231281e-35, 1.64231281e-35, -10, -10, -10, -10,
            -10, -10, -10, 1.64231281e-35, 1.64231281e-35,
            -10, -10, -10, -10, -10],

    "P04": [-1.69698011, -1.91674073, -0.58919366, -0.3680726,
            0.07208135, -0.0653034, -0.48370471, -0.29420384,
            2.71259738, 2.87063206, 3.05748287, 3.09828929,
            2.51670161, 2.46407898, 2.95022957, 1.97835712],

    "P05": [-10, -10, -10, -10, -10, -10, -10, 0.40734109,
            0.55799476, -10, -10, -10, 2.56483986, -10, -10, -10]
}

erd_final = {
    "P01": [7.84342401, 4.55757095, 10, 10, 7.66980776,
            6.81970491, 9.7067812, 10, -5.10667329, 6.12282128,
            6.79065923, 0.42598563, 1.71270577, 0.07680391,
            1.79657224],

    "P02": [-0.18336366, -8.06763188, 2.83371545, -0.41095946,
            1.99357912, -4.79120404, 2.95394046, 4.27246142,
            -4.32812311, 0.94521376, 3.36024944, -2.53253156,
            1.8556199, 4.77156423],

    "P03": [-6.37837173, -3.68747069, -6.57192485, -10,
            1.5432189, 3.01533005, -1.8879691, -4.50597551,
            -2.05704469, 4.73956004, -6.08647697, -6.57240712,
            6.6201865, 3.61815418, -3.85694452, -6.18578984],

    "P04": [10] * 16,

    "P05": [-5.98771902, -5.92623972, 5.11628236, -10,
            -1.52416702, -5.98189452, -10, 10, 2.73085074,
            -10, -5.10387358, -4.9906025, 5.04359725,
            3.56318305, 2.97334529, 5.00236547]
}

# --------------------------------------------------
# 2. Clinical scores: [initial, final]
# --------------------------------------------------

FIM = np.array([
    [74, 90],    # P01 Salvador
    [76, 86],    # P02 Joao
    [53, 70],    # P03 Maria Freitas
    [94, 113],   # P04 Gilcimar
    [95, 105]    # P05 Maria Goreth
], dtype=float)

FMA = np.array([
    [13, 16],
    [24, 28],
    [22, 25],
    [25, 28],
    [22, 22]
], dtype=float)

participant_ids = ["P01", "P02", "P03", "P04", "P05"]

# --------------------------------------------------
# 3. Reduce each ERD array to one mean per participant
# --------------------------------------------------

ERD_initial_mean = np.array([
    np.nanmean(erd_initial[pid]) for pid in participant_ids
])

ERD_final_mean = np.array([
    np.nanmean(erd_final[pid]) for pid in participant_ids
])

# Calculate changes: final - initial
ERD_change = ERD_final_mean - ERD_initial_mean
FIM_change = FIM[:, 1] - FIM[:, 0]
FMA_change = FMA[:, 1] - FMA[:, 0]

print("Participant-level data")
print("-" * 75)
print("ID     ERD initial   ERD final   ERD change   FIM change   FMA change")

for i, pid in enumerate(participant_ids):
    print(
        f"{pid:<6} "
        f"{ERD_initial_mean[i]:>12.3f} "
        f"{ERD_final_mean[i]:>11.3f} "
        f"{ERD_change[i]:>12.3f} "
        f"{FIM_change[i]:>12.1f} "
        f"{FMA_change[i]:>12.1f}"
    )

# --------------------------------------------------
# 4. Pearson correlation using NumPy only
# --------------------------------------------------

def pearson_r(x, y):
    """Pearson correlation coefficient, excluding invalid pairs."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) < 3:
        return np.nan, len(x)

    # Correlation is undefined if either variable is constant
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan, len(x)

    r = np.corrcoef(x, y)[0, 1]
    return r, len(x)


def report_correlation(label, x, y):
    r, n = pearson_r(x, y)

    if np.isnan(r):
        print(f"{label:<35} r = undefined (n = {n})")
    else:
        print(f"{label:<35} r = {r:+.4f} (n = {n})")


print("\n\nPEARSON CORRELATION RESULTS")
print("=" * 65)

# Initial ERD vs initial clinical scores
report_correlation(
    "Initial ERD vs initial FIM",
    ERD_initial_mean, FIM[:, 0]
)
report_correlation(
    "Initial ERD vs initial FMA",
    ERD_initial_mean, FMA[:, 0]
)

# Final ERD vs final clinical scores
report_correlation(
    "Final ERD vs final FIM",
    ERD_final_mean, FIM[:, 1]
)
report_correlation(
    "Final ERD vs final FMA",
    ERD_final_mean, FMA[:, 1]
)

# Changes in ERD vs changes in clinical scores
report_correlation(
    "ERD change vs FIM change",
    ERD_change, FIM_change
)
report_correlation(
    "ERD change vs FMA change",
    ERD_change, FMA_change
)
