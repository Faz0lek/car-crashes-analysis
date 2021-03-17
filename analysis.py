"""IZV project part 2

This module analyses car crashes in the Czech Republic.
"""

__author__ = "Martin Kostelník (xkoste12)"

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import sys


def get_dataframe(filename: str = "accidents.pkl.gz", verbose: bool = False) -> pd.DataFrame:
    """Create dataframe with car crashes data

    Keyword arguments:
    filename -- name of the file containing the data (default accidents.pkl.gz)
    verbose -- if True, function prints dataframe size before and after change to category data (default False)
    """
    try:
        df = pd.read_pickle(filename, "gzip")
    except FileNotFoundError:
        print(f"ERROR: File '{filename}' not found. Quitting.", file=sys.stderr)
        sys.exit(1)

    df["date"] = df["p2a"].astype("datetime64")

    if verbose:
        # 1 MB = 1048576 B
        print(f"orig_size={df.memory_usage(deep=True).sum() / 1048576:.1f} MB")

    # These columns will not be changed to category type
    exclude_cols = ["region", "p2a", "date", "p13a", "p13b", "p13c", "p12", "p53", "p16"]

    # Columns with less unique values than THRESHOLD will not be changed to category type
    THRESHOLD = 1000

    for col in df:
        if df[col].nunique() < THRESHOLD and col not in exclude_cols:
            df[col] = df[col].astype("category")

    if verbose:
        # 1 MB = 1048576 B
        print(f"new_size={df.memory_usage(deep=True).sum() / 1048576:.1f} MB")

    return df


def plot_conseq(df: pd.DataFrame, fig_location: str = None, show_figure: bool = False):
    """Plot car crashes data concerning injuries

    Keyword arguments:
    df -- dataframe containing data
    fig_location -- plots will be saved to this file (default None)
    show_figure -- if True, function displays the plots on screen
    """
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 15))

    sns.barplot(data=df, x="region", y="p13a", ax=ax1, estimator=np.sum, order=df["region"].value_counts().index, ci=None, color="mediumblue")
    sns.barplot(data=df, x="region", y="p13b", ax=ax2, estimator=np.sum, order=df["region"].value_counts().index, ci=None, color="mediumblue")
    sns.barplot(data=df, x="region", y="p13c", ax=ax3, estimator=np.sum, order=df["region"].value_counts().index, ci=None, color="mediumblue")
    sns.countplot(data=df, x="region", ax=ax4, order=df["region"].value_counts().index, color="mediumblue")

    ax1.set(xlabel="Kraj", ylabel="Počet", title="Úmrtí")
    ax1.set_fc("silver")
    ax2.set(xlabel="Kraj", ylabel="Počet", title="Těžce ranění")
    ax2.set_fc("silver")
    ax3.set(xlabel="Kraj", ylabel="Počet", title="Lehce ranění")
    ax3.set_fc("silver")
    ax4.set(xlabel="Kraj", ylabel="Počet", title="Celkem nehod")
    ax4.set_fc("silver")

    fig.tight_layout()

    if show_figure:
        plt.show()

    if fig_location:
        plt.savefig(fig_location)

    plt.close(fig)


def plot_damage(df: pd.DataFrame, fig_location: str = None, show_figure: bool = False):
    """Plot car crashes data concerning property damage

    Keyword arguments:
    df -- dataframe containing data
    fig_location -- plots will be saved to this file (default None)
    show_figure -- if True, function displays the plots on screen
    """
    regions = ["MSK", "JHM", "OLK", "ZLK"]
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
    axs = [ax1, ax2, ax3, ax4]

    i = 0
    for region in regions:
        tmp_df = df.loc[df["region"] == region].copy()
        tmp_df["p53"] /= 10

        cause_bins = [99, 200, 300, 400, 500, 600, 700]
        cause_labels = [
            "Nezaviněna řidičem", "Nepřiměřená rychlost jízdy",
            "Nesprávné předjíždění", "Nedání přednosti v jízdě",
            "Nesprávný způsob jízdy", "Technická závada vozidla",
        ]
        cause=pd.cut(tmp_df["p12"], bins=cause_bins, labels=cause_labels)

        damage_bins = [0, 50, 200, 500, 1000, float("inf")]
        damage_labels = ["< 50", "50 - 200", "200 - 500", "500 - 1000", "> 1000"]
        damage=pd.cut(tmp_df["p53"], bins=damage_bins, labels=damage_labels, include_lowest=True)

        sns.countplot(ax=axs[i], x=damage, hue=cause)
        axs[i].set(title=region, yscale="log", xlabel="Škoda (tisíc Kč)", ylabel="Počet")
        axs[i].legend(loc="upper right", frameon=False, fontsize=8)

        for val in range(0, 5):
            axs[i].axhline(y = 10**val, color='gray', alpha=0.2, linestyle='--')

        i += 1

    fig.tight_layout()

    if show_figure:
        plt.show()

    if fig_location:
        plt.savefig(fig_location)

    plt.close(fig)


def plot_surface(df: pd.DataFrame , fig_location: str = None, show_figure: bool = False):
    """Plot car crashes data based on road quality

    Keyword arguments:
    df -- dataframe containing data
    fig_location -- plots will be saved to this file (default None)
    show_figure -- if True, function displays the plots on screen
    """
    regions = ["MSK", "JHM", "OLK", "ZLK"]
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 5))
    axs = [ax1, ax2, ax3, ax4]

    i = 0
    for region in regions:
        tmp_df = df.loc[df["region"] == region].copy()
        tmp_df = tmp_df[["p16", "p1", "date"]].copy()

        colnames = {
            1: "suchý nezněčištěný", 2: "suchý znečištěný", 3: "mokrý",
            4: "bláto", 5: "náledí, ujetý sníh - posypané", 6: "náledí, ujetý sníh - neposypané",
            7: "olej, nafta apod.", 8: "souvislý sníh", 9: "náhlá změna stavu", 0: "jiný stav",
        }
        ctab = pd.crosstab(index=tmp_df["date"], columns=tmp_df["p16"], values=tmp_df["p1"], aggfunc=len)
        ctab.rename(columns=colnames, inplace=True)
        resampled = ctab.resample('M').sum()
        resampled = resampled.stack().reset_index()

        sns.lineplot(data=resampled, ax=axs[i], x="date", hue="p16", y=resampled[0])
        axs[i].set(title=region)
        i += 1

    ax1.set(xlabel="", ylabel="Počet nehod")
    ax2.set(xlabel="", ylabel="")
    ax3.set(xlabel="Datum nehody", ylabel="Počet nehod")
    ax4.set(xlabel="Datum nehody", ylabel="")

    ax1.get_legend().remove()
    ax2.legend(fontsize=8, title="Stav vozovky", bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ax3.get_legend().remove()
    ax4.get_legend().remove()

    fig.tight_layout()

    if show_figure:
        plt.show()

    if fig_location:
        plt.savefig(fig_location)

    plt.close(fig)


if __name__ == "__main__":
    df = get_dataframe(verbose=True)
    plot_conseq(df, "conseq.pdf")
    plot_damage(df, "damage.pdf")
    plot_surface(df, "surface.pdf")
