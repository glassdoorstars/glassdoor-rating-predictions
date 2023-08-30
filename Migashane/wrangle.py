################# IMPORTS #####################
#  data manipulation
import pandas as pd
import numpy as np

# natural language processing
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
import unicodedata
import re

# split data
from sklearn.model_selection import train_test_split

# Quiet all warnings
import warnings
warnings.filterwarnings('ignore')

############## ACQUISITION FUNCTIONS ########################



############### PREPARATION FUNCTIONS #######################
def clean(string):
    """
    This function puts a string in lowercase, normalizes any unicode characters, removes anything that         
    isn't an alphanumeric symbol or single quote.
    """
    # Normalize unicode characters
    string = unicodedata.normalize('NFKD', string).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    
    # Remove unwanted characters and put string in lowercase
    string = re.sub(r"[^\w0-9'\s]", '', string).lower()
            
    return string

def lemmatize(string):
    """
    This function takes in a string, lemmatizes each word, and returns a lemmatized version of the orignal string
    """
    # Build the lemmatizer
    lemmatizer = nltk.stem.WordNetLemmatizer()
    
    # Run the lemmatizer on each word after splitting the input string, store results in the 'results' list
    results = []
    for word in string.split():
        results.append(lemmatizer.lemmatize(word))
    
    # Convert results back into a string
    string = ' '.join(results)
    
    # Return the resulting string
    return string

def remove_stopwords(string, extra_words=None, exclude_words=None):
    """
    Takes in a string, with optional arguments for words to add to stock stopwords and words to ignore in the 
    stock list removes the stopwords, and returns a stopword free version of the original string
    """
    # Get the list of stopwords from nltk
    stopword_list = stopwords.words('english')
    
    # Create a set of stopwords to exclude
    excluded_stopwords = set(exclude_words) if exclude_words else set()
    
    # Include any extra words in the stopwords to exclude
    stopwords_to_exclude = set(stopword_list) - excluded_stopwords
    
    # Add extra words to the stopwords set
    stopwords_to_exclude |= set(extra_words) if extra_words else set()
    
    # Tokenize the input string
    words = string.split()
    
    # Filter out stopwords from the tokenized words
    filtered_words = [word for word in words if word not in stopwords_to_exclude]
    
    # Convert back to string
    string = ' '.join(filtered_words)
    
    # Return the resulting string
    return string

def split_readmes(df):
    """
    Takes in a dataframe and performs a 70/15/15 split. Outputs a train, validate, and test dataframe
    """
    # Perfrom a 70/15/15 split
    train_val, test = train_test_split(df, test_size=.2, random_state=95)
    train, validate = train_test_split(train_val, test_size=.25, random_state=95)
    
    # Return the dataframe slices
    return train, validate, test


def prep_readmes(df, col:str="content_column"):
    """
    Takes in the dataframe and the column name that contains the corpus data, creates a column of cleaned data, then uses that 
    to create a column without stopwords that is lemmatized, performs a train-validate-test split, and returns train, validate,
    and test.
    """
    # Initialize a list to collect cleaned elements in the for-loop below
    cleaned_row = []
    
    # Iterate through the readme_content values...
    for i in df[col].values:
        
        # Clean each value in the column and append to the 'cleaned_row' list
        cleaned_row.append(clean(i))
        
    # Assign the clean row content to a new column in the dataframe named 'cleaned_content
    df = df.assign(cleaned_content=cleaned_row)
    
    # Using a lambda, lemmatize all values in the 'cleaned_content' column and assign to a new column called 'lemmatized'
    df['lemmatized'] = df['cleaned_content'].apply(lambda x: lemmatize(remove_stopwords(x)))
    
    # Split the dataframe (70/15/15)
    train, validate, test = split_readmes(df)
    
    # Return train, validate, and test dataframes
    return train, validate, test

################ MAIN FUNCTION #####################
if __name__ == "__main__":
    train, validate, test = prep_readmes()