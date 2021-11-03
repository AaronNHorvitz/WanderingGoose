import numpy as np

from WanderingGoose.stats.tests.stationarity_randomwalk.augmented_dickey_fuller_test import (
    augmented_dickey_fuller_test,
)


def make_stationary(
    y: pd.Series,
    look_back_window: int = 100,
    max_steps: int = 4,
    significance_level: str = "5%",
    print_test_results=True,
):

    """
    This take the lowess smoothed data and puts it through a series of transformations until it is 
    stationary. The first is a log transformation followed by single differencing. Once the series passes
    the Dickey Fuller test for stationarity, the process stops. Then the record of transformation along 
    with the transformed values is returned. 
    
    Parameters
    ----------
    data: (pd.DataFrame) - DataFrame with all the values
    look_back_window: (int) - (Default to 100) The distance back in rows to cut off the values
    significance_level (str) - (Default to 4) Value passed to the stationarity test which is the significance of the test. 
    max_steps: (int) - (Default to 4) maximum steps to take
    print_test_results: (bool)  True - Print ADF fuller tests as they are conducted through each iteration. 
                                False -Don't print the results of each stationarity test. 
    """
    step = 0
    transformation_record = {}
    stationarity_test = False

    y_trans = y[-look_back_window:].copy()  # take the lookback window size

    print(
        f"\nAttempting the Augmented Dickey-Fuller Test for Stationarity at the {significance_level} level.\nAfter each transformation."
    )

    # Log transformation - record the first/last values prior to the transformation
    transformation_record[step] = ["log", y_trans.iloc[0], y_trans.iloc[-1]]
    y_trans = np.log1p(y_trans).fillna(0)  # log transformation

    print(f":  Log transformed at step: {step}.")

    # Difference until stationary
    while stationarity_test == False:

        step += 1

        # check for max steps
        if step == (max_steps + 1):
            print(f"Failed to make stationary after {step} steps.")
            return y_trans, transformation_record

        # Augmented Dickey Fuller Test
        stationarity_test = augmented_dickey_fuller_test(
            y_trans,
            significance_level=significance_level,
            print_test_results=print_test_results,
        )

        if stationarity_test == True:
            print("Passed")

        else:
            # Difference - record the first/last values prior to differencing them.
            transformation_record[step] = ["diff", y_trans.iloc[0], y_trans.iloc[-1]]
            y_trans = y_trans.diff().fillna(0)
            print(f":  Differenced at step: {step}.")

    return y_trans, transformation_record
