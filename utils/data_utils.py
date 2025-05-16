import polars as pl

SEX_TO_BINARY = {'male': 0, 'female': 1}

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


def add_bmi(all_data: pl.DataFrame) -> pl.DataFrame:
    """Adds a new column to the full dataframe with the BMI of each participant. Returns a new dataframe (couldn't get in-place to work).

    all_data is expected to have a 'Height' column representing the height in cm and a 'Weight' column representing the weight in kg.

    Args:
        all_data (pl.DataFrame): dataframe containing train and test data

    Returns:
        pl.DataFrame: dataframe with a new column 'bmi' representing the BMI of each participant
    """
    # create new height metric in meters
    all_data = all_data.with_columns(height_meters = pl.col('Height') / 100)

    # compute bmi
    all_data = all_data.with_columns(bmi = pl.col('Weight') / (pl.col('height_meters') ** 2))

    # drop the height_meters column
    all_data = all_data.drop('height_meters')

    return all_data


def linearize_body_temp(all_data: pl.DataFrame) -> pl.DataFrame:
    """Linearizes body temperature with respect to calories by taking the exponential of the body temperature. Drops the original Body_Temp column.

    Args:
        all_data (pl.DataFrame): dataframe containing train and test data

    Returns:
        pl.DataFrame: dataframe with a new column 'exp_Body_Temp' representing the exp(body temperature) of each participant
    """
    all_data = all_data.with_columns(exp_Body_Temp=pl.col('Body_Temp').exp())
    
    all_data = all_data.drop('Body_Temp')

    return all_data


def load_and_process_data(train_path: str = 'data/train.csv', test_path: str = 'data/test.csv', linearize_body_temp: bool = False) -> pl.DataFrame:
    """Pipeline to load and perform all major data processing steps.

    Returns:
        pl.DataFrame: dataframe with all the data and features needed for training and testing
    """

    all_data = load_all_data(train_path, test_path)

    all_data = add_bmi(all_data)

    if linearize_body_temp:
        # only really necessary if you're doing linear regression. This causes problems with numerical stability due to very extreme values which breaks the model, though
        all_data = linearize_body_temp(all_data)

    # convert string categorical column 'Sex' to a binary one-hot encoded column
    all_data = all_data.with_columns(Sex_encoded = pl.col('Sex').replace_strict(SEX_TO_BINARY)) # polars says this is more efficient than map_elements
    all_data.drop_in_place('Sex')

    return all_data
