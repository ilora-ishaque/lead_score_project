import polars as pl


def describe_objects(df: pl.DataFrame) -> pl.DataFrame:
    """
    Generate a summary for string or categorical columns in a Polars DataFrame,
    similar to Pandas' describe() for objects.

    Parameters:
    df (pl.DataFrame): Polars DataFrame to summarize.

    Returns:
    pl.DataFrame: Summary with statistics as row headers.
    """
    summaries = []

    for col in df.columns:
        if df[col].dtype == pl.Utf8:
            column_data = df[col]
            count = column_data.drop_nulls().len()
            unique = column_data.n_unique()
            null_count = column_data.null_count()
            top = column_data.mode().to_list()[:1]  # Get the most frequent value (if available)
            top_value = top[0] if top else None

            if top_value is not None:
                freq = column_data.filter(column_data == top_value).len()
            else:
                freq = None

            summaries.append({
                "Statistic": col,
                "count": count,
                "unique": unique,
                "null_count": null_count,
                "top": top_value,
                "freq": freq
            })

    # Convert the summaries into a Polars DataFrame
    summary_df = pl.DataFrame(summaries)

    # Transpose the DataFrame so statistics are row headers
    summary_df = summary_df.transpose(include_header=True)
    summary_df = summary_df.rename({"column": "Statistic"})

    return summary_df


def count_missing(data: pl.DataFrame) -> pl.DataFrame:
    """Return a polars dataframe with missing counts per columns

    Args:
        data (pl.DataFrame): input dataframe to be analysed

    Returns:
        pl.DataFrame: dataframe with missing counts
    """
    missing = data.select(
        pl.col(c).is_null().sum().alias(f"{c}_missing") for c in data.columns
    )

    return missing

