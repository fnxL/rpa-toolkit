"""
Polars DataFrame utility functionsfor RPA Toolkit.
"""

import polars as pl


def rename_columns(
    df: pl.DataFrame | pl.LazyFrame, columns_map: dict, strict: bool = True
) -> pl.DataFrame | pl.LazyFrame:
    """
    Rename columns of a Polars DataFrame or LazyFrame.

    Args:
        df (pl.DataFrame | pl.LazyFrame): The input DataFrame or LazyFrame.
        columns_map (dict): A dictionary mapping old column names to new column names.

    Returns:
        pl.DataFrame | pl.LazyFrame: The DataFrame or LazyFrame with renamed columns.
    """
    return df.rename(columns_map, strict=strict)
