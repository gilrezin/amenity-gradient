# Gil Rezin
# 7/24/2026
# This file contains various functions pertaining to Points of Interest,

from pathlib import Path

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent

WGS84 = "EPSG:4326"
# NAD83 / New York Long Island (ftUS) - a projected CRS so distances come out in feet
# instead of meaningless degrees. Appropriate since both datasets are NYC-specific.
NY_STATE_PLANE_FT = "EPSG:2263"


def _to_geodataframe(df, columns, crs=WGS84):
    """
    Builds point/polygon geometry for a DataFrame from either a [latitude, longitude]
    column pair or a single well-known-text (WKT) geometry column.
    """
    if len(columns) == 2:
        lat_col, lon_col = columns
        geometry = gpd.points_from_xy(df[lon_col], df[lat_col])
    elif len(columns) == 1:
        geometry = gpd.GeoSeries.from_wkt(df[columns[0]])
    else:
        raise ValueError("columns must be a single WKT column or a [latitude, longitude] pair")
    return gpd.GeoDataFrame(df, geometry=geometry, crs=crs)


def merge_parcels_and_pois(parcels_csv, pois_csv, parcels_columns, pois_columns):
    """
    Inputs the csv file containing both property parcels and points of interest, as well as their defined columns, and finds the cross-distance between each parcel and its nearest point of interest.

    Returns:
        A dataframe containing the cross-distance between each parcel and its nearest point of interest.
    """
    parcels_df = pd.read_csv(parcels_csv, low_memory=False).dropna(subset=parcels_columns)
    pois_df = pd.read_csv(pois_csv).dropna(subset=pois_columns)

    parcels_gdf = _to_geodataframe(parcels_df, parcels_columns).to_crs(NY_STATE_PLANE_FT)
    pois_gdf = _to_geodataframe(pois_df, pois_columns).to_crs(NY_STATE_PLANE_FT)

    # sjoin_nearest uses a spatial index (STRtree) so this stays fast even with
    # hundreds of thousands of parcels matched against thousands of POI polygons.
    return gpd.sjoin_nearest(parcels_gdf, pois_gdf, how="left", distance_col="distance_ft")


def find_nearest_poi(merged_data):
    """
    Finds the nearest point of interest for each property parcel in the merged data.

    Args:
        merged_data (DataFrame): A DataFrame containing both property parcels and points of interest.

    Returns:
        DataFrame: A DataFrame with each parcel and its nearest point of interest.
    """
    # sjoin_nearest can return more than one row per parcel when multiple POIs are
    # exactly equidistant (ties), so reduce to a single nearest match per parcel.
    # The parcel id (original row index) is duplicated across tied rows, so it can't
    # be used directly for .loc lookups; reset it into a plain column first so idxmin
    # resolves to unique positional labels.
    working = merged_data.reset_index(drop=False)
    parcel_id_col = working.columns[0]
    nearest_idx = working.groupby(parcel_id_col)["distance_ft"].idxmin()
    return working.loc[nearest_idx].drop(columns=[parcel_id_col]).reset_index(drop=True)


# Example usage of the functions
parcels_csv = SCRIPT_DIR / "Primary_Land_Use_Tax_Lot_Output_(PLUTO)_20260723.csv"
pois_csv = SCRIPT_DIR / "Parks_Properties_20260723.csv"
parcels_columns = ["latitude", "longitude"]
pois_columns = ["multipolygon"]

merged_data = merge_parcels_and_pois(parcels_csv, pois_csv, parcels_columns, pois_columns)
nearest_pois = find_nearest_poi(merged_data) # tiebreaker function

print(nearest_pois)