"""IZV project part 1

This module plots parsed data from download.py using matplotlib.
"""

__author__ = "Martin Kostelník (xkoste12)"

import argparse
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import download as dl

def parse_arguments():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser()

    parser.add_argument("--fig_location", help="Figure location")
    parser.add_argument("--show_figure", help="Show plot in separate window", action='store_true')

    return parser.parse_args()

def plot_stat(data_source, fig_location = None, show_figure = False):
    """Plot parse data into nice charts

    Arguments:
    data_source -- parsed data to plot

    Keyword arguments:
    fig_location -- Save plotted charts into this folder. Name of the file is always 'crashes.pdf' (default None)
    show_figure -- Display plotted charts. (default None)
    """

    fig, axs = plt.subplots(5, figsize=(10, 20))

    data = {"2016": list(),
            "2017": list(),
            "2018": list(),
            "2019": list(),
            "2020": list(),}

    # Group data by year
    i = 0
    for d in data_source[1][3]:
        data[d[:4]].append(data_source[1][-1][i])
        i += 1

    # Plot a bar chart for each year
    i = 0
    for key in data:
        unique, counts = np.unique(np.array(data[key], dtype="U4"), return_counts=True)

        # Sort data by accident count
        s_counts = counts.copy()
        s_counts[::-1].sort()

        axs[i].title.set_text(f"Počet dopravních nehod v roce {key} (leden - září)" if key == "2020" else f"Počet dopravních nehod v roce {key}")
        axs[i].bar(unique, counts)

        axs[i].set_ylim([0, 25000])

        # Horizontal lines for better readability
        for j in range(5000, 25000, 5000):
            axs[i].axhline(y = j, color='gray', alpha=0.2, linestyle='--')

        # Display order of accidents above each bar
        for j, v in enumerate(counts):
            axs[i].text(j, v + 400, str(np.where(s_counts == v)[0][0] + 1) + '.', color='cornflowerblue', fontweight='bold', ha='center')

        i += 1

    fig.tight_layout()

    if fig_location is not None:
        if not os.path.isdir(fig_location):
            try:
                os.makedirs(fig_location)
            except OSError:
                print("ERROR: could not create directory, quitting", file=sys.stderr)
                sys.exit(1)

        plt.savefig(f'{fig_location}/crashes.pdf')
 
    if show_figure:
        plt.show()

    plt.close(fig)

if __name__ == "__main__":
    args = parse_arguments()

    data_source = dl.DataDownloader().get_list()
    plot_stat(data_source, args.fig_location, args.show_figure)
