import polars as pl
from rpa_toolkit.df import reorder_columns


def test_reorder_subset_columns():
    df = pl.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
    reordered_df = reorder_columns(df, ["C", "A"])
    expected_df = pl.DataFrame({"C": [7, 8, 9], "A": [1, 2, 3], "B": [4, 5, 6]})
    assert reordered_df.equals(expected_df)


def test_reorder_all_columns():
    df = pl.DataFrame({"X": [10, 20], "Y": [30, 40], "Z": [50, 60]})
    reordered_df = reorder_columns(df, ["Z", "Y", "X"])
    expected_df = pl.DataFrame({"Z": [50, 60], "Y": [30, 40], "X": [10, 20]})
    assert reordered_df.equals(expected_df)
