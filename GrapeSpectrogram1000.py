#!/usr/bin/env python
# coding: utf-8

# # Grape2 Spectrogram Generator
#
# Author: Cuong Nguyen
#
# Date     | Version     | Author | Comments
# ---------|-------------|--------|-----------
# 01-04-24 | Ver 1.00.00 | KC3UAX | Initial commit
# 01-25-24 | Ver 1.01.00 | KC3UAX | Optimize the program with multiprocessing and clean up

# ## Imports and Setup
import os
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count


# ## Find the Indices of the Rows to be Skipped by CSV Reader


def find_rows_with_characters(file_path):
    row_numbers = []

    with open(file_path, "r") as file:
        for row_number, line in enumerate(file):
            if any(char in line for char in ["#", "T", "C"]):
                row_numbers.append(row_number)

    return row_numbers


# ## Read Datafiles


def process_file(file):
    dir_path = "Srawdata-2024-01-25"
    file_path = os.path.join(dir_path, file)
    df_raw = pd.read_csv(
        file_path,
        names=range(3),
        skiprows=find_rows_with_characters(file_path),
        header=None,
        converters={col: (lambda x: int(x, 16)) for col in range(3)},
    )
    return df_raw


def load_data(csv_files):
    with Pool(processes=min(cpu_count(), len(csv_files))) as pool:
        results = pool.map(process_file, csv_files)

    # Concatenate the results to get the final DataFrame
    df = pd.DataFrame()
    return pd.concat(results, ignore_index=True)


# df = load_data(csv_files)


# ## Raw Signals


def create_time_vector(fs):
    # Create time vector
    Ts = 1 / fs  # Convert to sampling period

    t_max = 24 * 60 * 60.0  # Time max in seconds.
    N = t_max * fs  # Total number of samples

    return np.arange(0, N) * Ts  # Time vector in seconds


def plot_sinusoid(t, df):
    # Plot the sinusoid.
    xlim = None
    # ylim = (-3, 3)
    ylim = None

    fig = plt.figure(figsize=(12, 10))

    ax = fig.add_subplot(3, 1, 1)
    ax.plot(t, df[0])
    ax.set_ylabel("Radio 0")
    ax.set_xticklabels([])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax = fig.add_subplot(3, 1, 2)
    ax.plot(t, df[1])
    ax.set_ylabel("Radio 1")
    ax.set_xticklabels([])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax = fig.add_subplot(3, 1, 3)
    ax.plot(t, df[2])
    ax.set_xlabel("t [s]")
    ax.set_ylabel("Radio 2")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    fig.tight_layout()
    plt.show()


# ## Spectrograms

if __name__ == "__main__":
    # Matplotlib settings to make the plots look a little nicer.
    plt.rcParams["font.size"] = 12
    # plt.rcParams['font.weight']    = 'bold'
    plt.rcParams["axes.grid"] = False
    plt.rcParams["axes.xmargin"] = 0
    # plt.rcParams['grid.linestyle'] = ':'
    plt.rcParams["figure.figsize"] = (20, 6)

    # ## User Inputs

    fs = 8000
    dir_path = "Srawdata-2024-01-25"

    # ## Get List of CSV Files

    csv_files = [file for file in os.listdir(dir_path) if file.endswith(".csv")]
    csv_files = sorted(csv_files, key=lambda x: x[:19])
    print(csv_files)

    df = load_data(csv_files)

    f, t_spec, Sxx = signal.spectrogram(df[1], fs=fs, window="hann")
    Sxx_db = 10 * np.log10(Sxx)
    xlim = (0, 60 * 60)
    fig = plt.figure(figsize=(20, 6))
    ax = fig.add_subplot(1, 1, 1)
    mpbl = ax.pcolormesh(t_spec, f, Sxx_db)
    cbar = fig.colorbar(mpbl, label="PSD [dB]")
    ax.set_title(dir_path)
    ax.set_xlabel("t [s]")
    ax.set_xlim(xlim)
    ax.set_ylabel("DopplerShift [Hz]")
    # plt.show()
    # Save the plot to a file (e.g., PNG)
    file_path = "spectrogram1.png"
    plt.savefig(file_path)

    f, t_spec, Sxx = signal.spectrogram(
        df[1], fs=fs, window="hann", nperseg=int(fs / 0.01)
    )
    Sxx_db = 10 * np.log10(Sxx)
    flim = (490, 510)
    fig = plt.figure(figsize=(20, 6))
    ax = fig.add_subplot(1, 1, 1)
    mpbl = ax.pcolormesh(t_spec, f, Sxx_db)
    cbar = fig.colorbar(mpbl, label="PSD [dB]")
    ax.set_title(dir_path)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("DopplerShift [Hz]")
    ax.set_ylim(flim)
    # plt.show()
    # Save the plot to a file (e.g., PNG)
    file_path = "spectrogram2.png"
    plt.savefig(file_path)
