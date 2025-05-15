import polars as pl

def load_all_data(train_path: str = 'data/train.csv', test_path: str = 'data/test.csv') -> pl.DataFrame:
    """
    Load the full train and test datasets for calorie expenditure prediction, and return a single dataframe containing both.

    Args:
        train_path (str, optional): path to the train dataset. Defaults to 'data/train.csv'.
        test_path (str, optional): path to the test dataset. Defaults to 'data/test.csv'.

    Returns:
        pl.DataFrame: A single dataframe containing both the train and test datasets. Includes a boolean-type column ('train') to indicate the dataset.
    """
    # load the train and test dataframes
    train = pl.read_csv(train_path)
    test = pl.read_csv(test_path)
    
    # add a column to indicate the dataset. Can easily use a filter to separate them
    train = train.with_columns(train = True)
    test = test.with_columns(Calories = None, train = False)

    # concatenate the two dataframes
    all_data = pl.concat([train, test], how='vertical')

    return all_data