import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load message and category data from specified file paths (CSV files).

    Parameters
    ----------
    messages_filepath : str
        file path to a message dataset
    categories_filepath : str
        file path to a category dataset

    Returns
    ----------
    df : DataFrame
        data merging message and category dataset
    """

    messages = pd.read_csv(messages_filepath)      # load messages dataset
    categories = pd.read_csv(categories_filepath)  # load categories dataset
    df = messages.merge(categories, on='id')       # merge datasets

    return df


def clean_data(df):
    """
    Clean data, mainly dealing with category values.

    Parameters
    ----------
    df : DataFrame
        input data

    Returns
    ----------
    df : DataFrame
        cleaned data
    """

    # Split `categories` into separate category columns
    separated_categories = df.categories.str.split(';', expand=True) # create a dataframe of the 36 individual category columns
    row = separated_categories.iloc[0]                               # select the first row of the categories dataframe
    category_colnames = row.str.split('-').map(lambda x : x[0])      # use the row to extract a list of new column names for categories.
    separated_categories.columns = category_colnames                 # rename the columns of `categories`

    # Convert category values to just numbers 0 or 1
    for column in separated_categories:
        # set each value to be the last character of the string
        separated_categories[column] = separated_categories[column].str.split('-').map(lambda x: x[1])
        # convert column from string to numeric
        separated_categories[column] = separated_categories[column].astype(int)

    # Replace `categories` column in `df` with new category columns
    df = df.drop('categories', axis=1)                 # drop the original categories column from `df`
    df = pd.concat([df, separated_categories], axis=1) # concatenate the original dataframe with the new `categories` dataframe

    # drop duplicates
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    """
    Save cleaned data to a database file (sqlite).

    Parameters
    ----------
    df : DataFrame
        data to be saved as sqlite file
    database_filename : str
        database file name to save data
    """

    # Save the clean dataset into an sqlite database
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('disaster_response', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()