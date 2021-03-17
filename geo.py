"""IZV project part 3.1

This module plots maps visualising car crashes.
"""

__author__ = "Martin Kostelník (xkoste12)"

import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import contextily as ctx
import sklearn.cluster
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


def make_geo(df: pd.DataFrame) -> geopandas.GeoDataFrame:
    """Create geodataframe from existing dataframe using correct encoding

    Keyword arguments:
    df -- existing dataframe containing car crashes data

    Returns:
    Newly created GeoDataFrame
    """
    df.dropna(subset=['d', 'e'], inplace=True)
    return geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df['d'], df['e']), crs="EPSG:5514")


def plot_geo(gdf: geopandas.GeoDataFrame, fig_location: str = None, show_figure: bool = False):
    """Plot two charts displaying car crashes in MSK. First chart displays crashes inside cities,
    while the second one displays crashes outsides cities.

    Keyword arguments:
    gdf -- existing geodataframe containing car crashes data
    fig_location -- plots will be saved to this file (default None)
    show_figure -- if True, function displays the plots on screen (default False)
    """
    gdf = gdf[gdf.region == "MSK"].copy().to_crs(crs="EPSG:3857")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

    gdf[gdf.p5a == 1].plot(ax=ax1, markersize=0.7, alpha=0.3)
    gdf[gdf.p5a == 2].plot(ax=ax2, markersize=0.7, alpha=0.3)
    
    ctx.add_basemap(ax1, crs=gdf.crs.to_string(), source=ctx.providers.Stamen.TonerLite, attribution_size=5)
    ax1.set(title="Nehody v MSK v obci", xlim=ax2.get_xlim(), ylim=ax2.get_ylim())
    ax1.axis("off")

    ctx.add_basemap(ax2, crs=gdf.crs.to_string(), source=ctx.providers.Stamen.TonerLite, attribution_size=5)
    ax2.set(title="Nehody v MSK mimo obec")
    ax2.axis("off")

    fig.tight_layout()

    if show_figure: 
        plt.show()

    if fig_location:
        plt.savefig(fig_location)


def plot_cluster(gdf: geopandas.GeoDataFrame, fig_location: str = None, show_figure: bool = False):
    """Plot car crashes in clusters
    
    Keyword arguments:
    gdf -- existing geodataframe containing car crashes data
    fig_location -- plots will be saved to this file (default None)
    show_figure -- if True, function displays the plots on screen (default False)
    """
    gdf = gdf[gdf.region == "MSK"].copy().to_crs(crs="EPSG:3857")

    N_CLUSTERS = 24
    coords = np.dstack([gdf.geometry.x, gdf.geometry.y]).reshape(-1, 2)
    model = sklearn.cluster.MiniBatchKMeans(n_clusters=N_CLUSTERS).fit(coords)

    gdf_c = gdf.copy()
    gdf_c["cluster"] = model.labels_
    gdf_c = gdf_c.dissolve(by="cluster", aggfunc={"p1": "count"}).rename(columns={"p1": "cnt"})
    gdf_c_coords = geopandas.GeoDataFrame(geometry=geopandas.points_from_xy(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1]))
    gdf_c = gdf_c.merge(gdf_c_coords, left_on="cluster", right_index=True).set_geometry("geometry_y")

    plt.figure(figsize=(12, 10))
    ax = plt.gca()

    # Manipulate the colorbar
    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.1)

    gdf.plot(ax=ax, markersize=0.5, alpha=0.3) # Plot all crashes
    gdf_c.plot(ax=ax, cax=cax, markersize=gdf_c["cnt"] * 1.5, column="cnt", legend=True, legend_kwds={"label": "Počet nehod"}, alpha=0.5) # Plot clusters

    ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.Stamen.TonerLite, attribution_size=10)
    ax.set(title="Nehody v MSK")
    ax.axis("off")

    plt.tight_layout()

    if show_figure:
        plt.show()

    if fig_location:
        plt.savefig(fig_location)


if __name__ == "__main__":
    gdf = make_geo(pd.read_pickle("accidents.pkl.gz"))
    plot_geo(gdf, "geo1.png", False)
    plot_cluster(gdf, "geo2.png", False)
