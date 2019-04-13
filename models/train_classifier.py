import sys
import re
import pickle
import pandas as pd
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def load_data(database_filepath):
    """
    Load data from file path to a database file (sqlite).

    Parameters
    ----------
    database_filepath : str
        database file path (sqlite)

    Returns
    -------
    X : DataFrame
        feature variables
    Y : DataFrame
        target variables
    category_names : list of str
        response category names (ex. 'related', 'request', 'offer')
    """

    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disaster_response', con=engine)

    # split data into X, Y, category_names
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = Y.columns.values

    return X, Y, category_names


def tokenize(text):
    """
    Tokenize text.

    Parameters
    ----------
    text : str
        input text

    Returns
    ----------
    tokens_lemmatized : list of str
        tokens lemmatized without English stop words
    """

    lower_text = text.lower()                                                                         # lower text
    cleaned_text = re.sub(r"[^a-z0-9]", " ", lower_text)                                              # clean text
    tokens = word_tokenize(cleaned_text)                                                              # tokenize text
    tokens_withou_stop_words = [token for token in tokens if token not in stopwords.words("english")] # remove stop words
    tokens_lemmatized = [WordNetLemmatizer().lemmatize(token) for token in tokens_withou_stop_words]  # lemmatize tokens

    return tokens_lemmatized


def build_model():
    """
    Build a model based on a pipeline with grid search parameters.

    Returns
    ----------
    cv : GridSearchCV
        model
    """

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # grid search parameters
    parameters = {
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__max_depth': [3, 5, 8]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate and print results of predictions.

    Parameters
    ----------
    model : GridSearchCV
        model to be evaluated
    X_test : Series
        feature variables for test dataset
    Y_test : DataFrame
        target variables for test dataset
    category_names : list of str
        response category names (ex. 'related', 'request', 'offer')
    """

    Y_pred = model.predict(X_test)

    for i, column in enumerate(Y_test.columns):
        print("Category: {}".format(category_names[i]))
        print("Accuracy Score: {:.2f}".format(accuracy_score(Y_test.values[i], Y_pred[i])))
        print("Classification Report: \n {}".format(classification_report(Y_test.values[i], Y_pred[i])))


def save_model(model, model_filepath):
    """
    Save trained model to a specified filepath (saved as pickle).

    Parameters
    ----------
    model : GridSearchCV
        trained model to be saved
    model_filepath : str
        file path to save a model
    """

    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()