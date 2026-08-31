######################################################################
# EEG data processing to determine Laplacian filter-based power,
# relative power changes and number of times Treadmill was activated.
# Author: Aura Ximena Gonzalez Cely
# V1.0 - 08.2026
# Please change the participant subject folder while executing, 
# depending on where you stored data folder as well as the participant
# you want to analyze.
######################################################################

import os
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path

# Change subject folder depending on where you store the data, as well as the participant P01, P02, P03
subject_folder = Path(r"A:\\ARTICLES\\MDPI 2026\\Complete_code\\P01")

eeg_fs=sampling_rate=125
pow_PM = []
RP=[]
window_seconds = 1.0
epoch_interval_seconds = 0.1
window_samples = round(window_seconds * sampling_rate)  # 125

def calculate_power_spectrum(X, Fs, freq_range):
    fft_df = np.fft.fft(X,axis=0)
    freqs = np.fft.fftfreq(len(X), 1 / Fs)
    powfft = np.abs(fft_df) ** 2
    Pm = np.mean(powfft[np.where((freqs >= freq_range[0]) & (freqs <= freq_range[1]))],axis=0)
    return Pm

def process_segments_calib(segments, true_label, segment_name):
    
    Ptot = []
    EEG_channels = [8,9,10,12,13,14]
    velocHigh_count = 0
    velocMed_count = 0
    velocLow_count = 0

    for trial_index, signal in enumerate(segments):

        if len(signal) < window_samples:
            print(f"Skipping {segment_name} segment {trial_index}: " f"it has only {len(signal)} samples.")
            continue

        window_index = 0
        while True:
            start_sample = math.floor(window_index * epoch_interval_seconds * sampling_rate + 0.5)
            end_sample = start_sample + window_samples

            if end_sample > len(signal):
                break

            window = signal[start_sample:end_sample, :]
            # print(window.shape)
            CAR = window[:, EEG_channels].mean(axis=1)  # shape: (126,)
            Cz = window[:, 9] - CAR
            Pmu = calculate_power_spectrum(Cz, eeg_fs, [8, 12])
            Pbeta = calculate_power_spectrum(Cz, eeg_fs, [15, 20])
            Pfull = calculate_power_spectrum(Cz, eeg_fs, [8, 30])
            Prel = ((Pmu+Pbeta)/2)/Pfull
            Ptot.append(Prel)

            if segment_name == "Active":
                pow_PM.append(Prel)
            elif segment_name == "Imagery":
                pow_PMr = np.array(pow_PM)
                pow_PMr = pow_PMr.mean()                                    
                RNFB = int(100 * abs(Prel - pow_PMr) / pow_PMr) if not any(np.isnan([Prel, pow_PMr])) and pow_PMr != 0 else None
                if RNFB is not None and RNFB < 20:
                    velocHigh_count += 1
                elif RNFB is not None and RNFB > 80:
                    velocMed_count += 1
                elif RNFB is not None:
                    velocLow_count += 1
                RP.append(RNFB)
            window_index += 1
    return velocHigh_count, velocMed_count, velocLow_count, Ptot, RP

##################################################################################################################################################################
#                                                                   CALIBRATION DATA LOADING
##################################################################################################################################################################

calibration_sessions = {csv_file.stem.lower(): csv_file for csv_file in subject_folder.glob("*.csv")}

for session_key in calibration_sessions:
    calibration_csv_path = calibration_sessions[session_key]

    # Example: S1, S2, S3
    session_name = calibration_csv_path.stem
    session_folder = subject_folder / session_name
    # Load CSV files
    calibration_data = pd.read_csv(calibration_csv_path)
    print(f"\nSession: {session_name}")
    print(f"Calibration CSV loaded: {calibration_csv_path.name}")
    ########################################################################################################################333
    #                                               Calibration Data Processing
    ########################################################################################################################333
    eeg_data = calibration_data.iloc[:, 2:18].to_numpy()
    markers = calibration_data.iloc[:, 18].to_numpy()
    print(f"EEG data shape: {eeg_data.shape}")

    labels_mapping = {
        32775: "base_start",
        32776: "base_end",
        768: "start",
        800: "end",
        32777: "MI",
    }

    # Find all sample indices for each event marker
    indexes = {label: np.flatnonzero(markers == marker_value) for marker_value, label in labels_mapping.items()}

    baseline_segments = []
    number_of_baselines = min(
        len(indexes["base_start"]),
        len(indexes["base_end"])
    )

    for baseline_index in range(number_of_baselines):
        start_sample = indexes["base_start"][baseline_index]
        end_sample = indexes["base_end"][baseline_index]

        if end_sample <= start_sample:
            print(f"Skipping invalid baseline {baseline_index}.")
            continue

        baseline_signal = eeg_data[start_sample:end_sample]
        baseline_segments.append(baseline_signal)

    # -------------------------------------------------------
    # 2. Extract trial segments: start -> MI -- PASSIVE
    # -------------------------------------------------------
    mi_starts = indexes["MI"]
    passive_starts = indexes["start"]
    end_starts = indexes["end"]

    number_of_trials = min(len(mi_starts), len(passive_starts))
    passive_segments = []
    for trial_index in range(number_of_trials):
        start_index = passive_starts[trial_index]
        end_index = mi_starts[trial_index]
        # Skip invalid segments
        if end_index <= start_index:
            print(f"Skipping trial {trial_index}: " f"PASSIVE index ({end_index}) is before MI index ({start_index}).")
            continue
        passive_signal = eeg_data[start_index:end_index]
        passive_segments.append(passive_signal)

    imagery_segments = []

    for trial_index in range(number_of_trials):

        # Skip trials without both a start and an end marker
        if (trial_index >= len(mi_starts) or trial_index >= len(end_starts)):
            print(f"Skipping trial {trial_index}: " "missing MI start or end index.")
            continue

        start_index = int(mi_starts[trial_index])
        end_index = int(end_starts[trial_index])

        # Skip invalid or incomplete trials
        if (start_index < 0 or end_index <= start_index or end_index > len(eeg_data)):
            print(f"Skipping trial {trial_index}: " f"invalid/incomplete interval " f"[{start_index}:{end_index}].")
            continue

        imagery_signal = eeg_data[start_index:end_index]
        imagery_segments.append(imagery_signal)

    print(f"Number of valid MI trials: {len(imagery_segments)}, " f"rest trials: {len(passive_segments)}")

    (velocHigh_countbase, velocMed_countbase, velocLow_countbase, Ptotbase,RPbase) = process_segments_calib(baseline_segments,true_label=1,segment_name="Baseline")
    print("Baseline: ", velocHigh_countbase, velocMed_countbase, velocLow_countbase, len(Ptotbase), np.isnan(Ptotbase))
    (velocHigh_countPass, velocMed_countPass, velocLow_countPass, PtotPass,RPpass) = process_segments_calib(passive_segments,true_label=0,segment_name="Active")
    print("Active walking: ", velocHigh_countPass, velocMed_countPass, velocLow_countPass, len(PtotPass), np.isnan(PtotPass))
    (velocHigh_countMI, velocMed_countMI, velocLow_countMI, PtotMI,RPtrue) = process_segments_calib(imagery_segments,true_label=1,segment_name="Imagery")
    print("Imagery: ", velocHigh_countMI, velocMed_countMI, velocLow_countMI, len(PtotMI), np.isnan(PtotMI))
    
    ########################################################################### Figure plot ###############################################################
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Palatino Linotype", "Palatino", "DejaVu Serif"],
        "font.size": 22
    })

    fig, axes = plt.subplots(1,3,figsize=(14, 4.5),gridspec_kw={"width_ratios": [3, 1.2, 2.5]})

    main_boxplot = axes[0].boxplot(
        [
            np.asarray(Ptotbase).flatten(),
            np.asarray(PtotPass).flatten(),
            np.asarray(PtotMI).flatten()
        ],
        patch_artist=True,
        boxprops={
            "facecolor": "#003366",
            "edgecolor": "black"
        },
        medianprops={
            "color": "#7FB3D5",
            "linewidth": 2
        }
    )

    axes[0].set_xticks([1, 2, 3])
    axes[0].set_xticklabels(["Baseline","Active\nwalking","MI + NF"])
    axes[0].set_ylim(0.3, 2.2)
    axes[0].set_ylabel("Normalized Power\n Laplacian filter")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    axes[0].set_axisbelow(True)

    # Separate RP boxplot 
    rp_boxplot = axes[1].boxplot(
        [np.asarray(RPtrue).flatten()],
        patch_artist=True,
        boxprops={
            "facecolor": "#001F3F",
            "edgecolor": "black"
        },
        medianprops={
            "color": "#7FB3D5",
            "linewidth": 2
        }
    )

    axes[1].set_xticks([1])
    axes[1].set_xticklabels(["RP"])
    axes[1].set_ylim(0, 125)
    axes[1].set_ylabel("Relative Power [%]")
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)
    axes[1].set_axisbelow(True)

    # Bar plot
    velocity_labels = ["High", "Medium", "Low"]

    velocity_counts = np.asarray([
        velocHigh_countMI,
        velocMed_countMI,
        velocLow_countMI
    ], dtype=float)

    bars = axes[2].bar(
        velocity_labels,
        velocity_counts,
        color=[
            "#2F6FA5",  # Medium blue
            "#4D8BBE",  # Lighter medium blue
            "#6AA4CF"   # Lightest, but still medium blue
        ],
        edgecolor="black",
        linewidth=1
    )

    axes[2].bar_label(bars,labels=[f"{value:g}" for value in velocity_counts],padding=3)
    axes[2].set_ylim(0, 1250)
    axes[2].set_ylabel("Number of correct values")
    axes[2].yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[2].grid(axis="y", linestyle="--", alpha=0.4)
    axes[2].set_axisbelow(True)

    fig.tight_layout()
    output_path = os.path.join(subject_folder, f"{session_name}_power_velocity_summary.eps")
    fig.savefig(output_path,format="eps",bbox_inches="tight")
    plt.show()
