import polars as pl
import logging
from polars._typing import FileSource
from typing import Any, Sequence

logger = logging.getLogger(__name__)


def read_excel(
    source: FileSource,
    *,
    sheet_id: int | None = None,
    sheet_name: str | None = None,
    header_row: int | None = 0,
    has_header: bool = True,
    columns: Sequence[int] | Sequence[str] | str | None = None,
    drop_empty_rows: bool = True,
    drop_empty_cols: bool = True,
    raise_if_empty: bool = True,
    cast: dict[str, pl.DataType] | None = None,
    **kwargs: Any,
) -> pl.LazyFrame:
    """
    Read an Excel file into a Polars LazyFrame with enhanced processing options.

    This function extends Polars' read_excel functionality by adding automatic
    column name cleaning and optional data type casting. It reads Excel data
    and returns a LazyFrame for efficient data processing.

    Parameters
    ----------
    source : FileSource
        Path to the Excel file or file-like object to read
    sheet_id : int, optional
        Sheet number to read (cannot be used with sheet_name)
    sheet_name : str, optional
        Sheet name to read (cannot be used with sheet_id)
    header_row : int, optional
        Row index of the header row (default is 0)
    has_header : bool, default=True
        Whether the sheet has a header row
    columns : Sequence[int] | Sequence[str] | str, optional
        Columns to select (by index or name)
    drop_empty_rows : bool, default=True
        Remove empty rows from the result
    drop_empty_cols : bool, default=True
        Remove empty columns from the result
    raise_if_empty : bool, default=True
        Raise an exception if the resulting DataFrame is empty
    cast : dict[str, pl.DataType], optional
        Dictionary mapping column names to desired data types for casting
    **kwargs : Any
        Additional keyword arguments passed to polars.read_excel

    Returns
    -------
    pl.LazyFrame
        A Polars LazyFrame containing the Excel data with cleaned column names
        and optional type casting applied

    Raises
    ------
    ValueError
        If both sheet_id and sheet_name are specified

    Examples
    --------
    >>> df = read_excel("data.xlsx")
    >>> df = read_excel("data.xlsx", sheet_name="Sheet1")
    >>> df = read_excel("data.xlsx", cast={"date": pl.Date, "value": pl.Float64})
    """
    if sheet_id is not None and sheet_name is not None:
        raise ValueError("sheet_id and sheet_name cannot be both specified.")

    read_options = {
        "header_row": header_row,
    }

    df = pl.read_excel(
        source,
        sheet_id=sheet_id,
        sheet_name=sheet_name,
        read_options=read_options,
        has_header=has_header,
        columns=columns,
        drop_empty_rows=drop_empty_rows,
        drop_empty_cols=drop_empty_cols,
        raise_if_empty=raise_if_empty,
        **kwargs,
    )

    df.columns = [col.strip().lower() for col in df.columns]
    logger.info(f"Intial df rows: {df.height}, columns: {df.width}")

    if df.height == 0:
        logger.warning("Dataframe is empty.")

    if cast is not None:
        for col, dtype in cast.items():
            if col not in df.columns:
                logger.warning(f"Column {col} not found in dataframe.")
                continue

            logger.info(f"Casting column {col} to {dtype}")
            df = df.with_columns(pl.col(col).cast(dtype, strict=False))

    return df.lazy()
