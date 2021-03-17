"""IZV project part 3.3

This file is used to assist in creating doc.pdf containing various
statistics about car crashes in Czech Republic.
"""

__author__ = "Martin Kostelník (xkoste12)"

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


def get_df() -> pd.DataFrame:
    """Loads data from accidents.pkl.gz file

    Returns:
    Loaded dataframe
    """
    return pd.read_pickle("accidents.pkl.gz")


def plot_weather(df: pd.DataFrame):
    """Plot chart containing car crashes during different weather situations

    Keyword arguments:
    df -- existing dataframe containing car crashes data
    """
    plt.figure(figsize=(12, 5))
    ax = plt.gca()

    # Change values to represent weather situation
    bar_names = {0: "Jiné", 1: "Neztížené", 2: "Mlha", 3: "Slabý déšť", 4: "Déšť", 5: "Sněžení", 6: "Náledí", 7: "Nárazový vítr"}
    df["weather"] = df["p18"].replace(bar_names)

    sns.countplot(data=df, x="weather", ax=ax, order=df["weather"].value_counts().index, color="mediumblue", log=True)
    ax.set(xlabel="Povětrnostní podmínky", ylabel="Počet")

    ax.axhline(y = 1000, color='gray', alpha=0.4, linestyle='--')
    ax.axhline(y = 10000, color='gray', alpha=0.4, linestyle='--')
    ax.axhline(y = 100000, color='gray', alpha=0.4, linestyle='--')

    plt.tight_layout()

    plt.savefig("fig.pdf")


def create_table(df: pd.DataFrame):
    """Creates table containing data per weather per year and
    print it to standard output in LaTeX format with tabular environment

    Keyword arguments:
    df -- existing dataframe containing car crashes data
    """
    print(r"\begin{tabular}{ |c|c|c|c|c|c|c|c|c| }")
    print(r"\hline")
    print(r"Rok & Jiné & Nezatížené & Mlha & Slabý déšť & Déšť & Sněžení & Náledí & Nárazový vítr \\")
    print(r"\hline")

    years = ["2016", "2017", "2018", "2019", "2020"]

    for year in years:
        print(f"{year} & ", end='')

        for val in np.sort(df["p18"].unique()):
            count = df[df.p2a.str.startswith(year)]["p18"].value_counts()[val]

            if val != 7:
                print(f"{count} & ", end='')
            else:
                print(f"{count} \\\\")

        print(r"\hline")

    print(r"\end{tabular}")


def print_stats(df: pd.DataFrame):
    """Prints stats used in doc.pdf

    Keyword arguments:
    df -- existing dataframe containing car crashes data
    """
    print(f"Celkem nehod: {len(df)}")

    crashes_snow = (df["p18"].value_counts()[5] + df["p18"].value_counts()[6]) / len(df) * 100
    print(f"Nehody při sněžení/náledí: {crashes_snow:.02f} %")
    print(f"Počet nehod při ztížených podmínkách: {len(df[df.p18 != 1])}")
    print(f"Počet nehod při neztížených podmínkách: {len(df[df.p18 == 1])}")


if __name__ == "__main__":
    df = get_df()
    plot_weather(df)
    create_table(df)
    print_stats(df)
