"""
Polars DataFrame utility functions for RPA Toolkit.
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


def reorder_columns(
    df: pl.DataFrame | pl.LazyFrame, columns_order: list[str]
) -> pl.DataFrame | pl.LazyFrame:
    """
    Reorder columns of a Polars DataFrame or LazyFrame.

    Args:
        df (pl.DataFrame | pl.LazyFrame): The input DataFrame or LazyFrame.
        columns_order (list[str]): A list specifying the desired order of columns or subset of columns that you want to be ordered.

    Returns:
        pl.DataFrame | pl.LazyFrame: The DataFrame or LazyFrame with reordered columns.

    Example:
        >>> import polars as pl
        >>> df = pl.DataFrame({
        ...     "A": [1, 2, 3],
        ...     "B": [4, 5, 6],
        ...     "C": [7, 8, 9]
        ... })
        >>> reordered_df = reorder_columns(df, ["C", "A"])
        >>> print(reordered_df)
        shape: (3, 3)
        ┌─────┬─────┬─────┐
        │ C   ┆ A   ┆ B   │
        │ --- ┆ --- ┆ --- │
        │ i64 ┆ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 7   ┆ 1   ┆ 4   │
        │ 8   ┆ 2   ┆ 5   │
        │ 9   ┆ 3   ┆ 6   │
        └─────┴─────┘

    """
    # Select the specified columns in the desired order, then append any remaining columns
    selected_cols = [pl.col(col) for col in columns_order if col in df.columns]
    remaining_cols = [pl.col(col) for col in df.columns if col not in columns_order]
    return df.select(selected_cols + remaining_cols)
