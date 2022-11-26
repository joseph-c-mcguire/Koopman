import numpy as np
import pandas as pd
import typing


def add_lags(df: pd.DataFrame,
             columns: typing.Union[list or np.array] = [],
             lag: typing.Union[int or list] = 1,
             drop_na: bool = True
             ) -> pd.DataFrame:
    """
    Adds lags to columns
    ...
    Parameters
    __________
    df : pd.DataFrame
        The dataframe to add lags on to the columns.
    columns: list = []
        The columns to add lags to
    lag : int or list = 1
        The lags to add to the dataframe,
        if list given then use the lag[x] for column x
    drop_na : bool = True
        Whether to drop NA-rows after adding lags
    Returns
    _______
    temp : pd.DataFrame
        The new dataframe with lags added

    """
    # Copy the dataframe to not overwrite
    temp = df.copy()
    # If empty just do to all columns
    if not columns:
        columns = temp.columns

    # If given int, turn to a list
    if isinstance(lag, int):
        if lag == 0:
            return temp
        lag = [lag]*len(columns)
    else:
        if len(lag) != len(columns):
            raise"lag must be same size as columns"
    # For each element of columns, and lags add a new column to df with the lag l
    for i in range(len(columns)):
        temp["{}_lag_{}".format(columns[i], lag[i])] = temp[columns[i]].shift(lag[i])
    if drop_na:
        temp = temp.dropna()
    return temp
