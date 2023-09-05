################# IMPORTS #####################
#  data manipulation
import pandas as pd
import numpy as np

from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

# web scraping
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from webdriver_manager.chrome import ChromeDriverManager

# natural language processing
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
import unicodedata
import re
import os
# split data
from sklearn.model_selection import train_test_split


################ Acquistion (Scraping Functions) #####################
def acquire_ratings_and_review_urls():
    # Install Webdriver
    service = Service(ChromeDriverManager().install())
    
    review_urls = []
    
    max_retries = 3
    retry_count = 0
    while retry_count < max_retries:
        try:
            for page in range(1, 101):
                url = f"https://www.glassdoor.com/Reviews/index.htm?overall_rating_low=1&page={page}&locId=1&locType=N&locName=United%20States&filterType=RATING_OVERALL"
                # access company page
                driver = webdriver.Chrome(service=service)
                driver.get(url)
                
                # Find employer name tags
                employer_names = driver.find_elements(By.XPATH, "//h2[@data-test='employer-short-name']")
                # Find employer rating tags
                employer_ratings = driver.find_elements(By.XPATH, "//span[@data-test='rating']")
                # Find employer review link tags
                review_links = driver.find_elements(By.XPATH, '//a[@data-test="cell-Reviews-url"]')

                # Extract names and ratings and add them to the lists
                for name, rating, link in zip(employer_names, employer_ratings, review_links):
                    names.append(name.text)
                    ratings.append(rating.text)
                    review_urls.append(link.get_attribute("href"))

                driver.quit()
            break

        except Exception as e:
            print(f"An error occurred: {e}")
            retry_count += 1
            print(f"Retrying... Attempt {retry_count}")
            time.sleep(5)  # Wait for a few seconds before retrying
    
    # Create a DataFrame
    data = {'name': names,
            'rating': ratings,
            'url': review_urls}
    
    ratings = pd.DataFrame(data) 
    
    return review_urls, ratings

def acquire_reviews(review_urls):
    # Initialize an empty list to store data
    data = []

    # Loop through the review URLs
    for company_url in review_urls:
        counter += 1
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                pros = ''
                cons = ''

                for i in range(10):
                    if i == 0:
                        next_url = company_url

                    driver = webdriver.Chrome(service=service)
                    driver.get(next_url)

                    for pro in driver.find_elements(By.XPATH, "//span[@data-test='pros']"):
                        pros += pro.text + '\n'

                    for con in driver.find_elements(By.XPATH, "//span[@data-test='cons']"):
                        cons += con.text + '\n'

                    driver.find_element(By.CSS_SELECTOR, 'button[data-test="pagination-next"]').click()
                    next_url = driver.current_url

                    driver.quit()

                data.append({'url': company_url, 'pros': pros, 'cons': cons})
                break  # Break the while loop if successful
            except Exception as e:
                print(f"An error occurred: {e}")
                retry_count += 1
                print(f"Retrying... Attempt {retry_count}")
                time.sleep(5)  # Wait for a few seconds before retrying

    # Create a DataFrame from the collected data
    reviews = pd.DataFrame(data)
    
    return reviews

def join_ratings_and_reviews(reviews, ratings):
    return ratings.join(reviews.set_index('url'), on='url', how='inner')

################ Acquistion (Scraping Functions) #####################

################ Preparation Functions #####################

def clean_strings(string):
    """
    This function puts a string in lowercase, normalizes any unicode characters, removes anything that         
    isn't an alphanumeric symbol or single quote.
    """
    # Normalize unicode characters
    string = unicodedata.normalize('NFKD', string).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    
    # Remove unwanted characters and put string in lowercase
    string = re.sub(r"[^\w0-9'\s]", '', string).lower()
    
    # Tokenize the string
    string = ToktokTokenizer().tokenize(string, return_str = True)
            
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

def split_data(df, test_size=.20, validate_size=.20, random_state=95):
    """
    Takes in a dataframe and performs a 60/20/20 split. Outputs a train, validate, and test dataframe
    """
    # Perfrom a 70/15/15 split
    train_val, test = train_test_split(df, test_size=test_size, random_state=random_state)
    train, validate = train_test_split(train_val, test_size=validate_size/(1-test_size),
                                       random_state=random_state )
    
    # Return the dataframe slices
    return train, validate, test


def prepare_data(df, text_cols):
    """
    Takes in the dataframe and the text columns that contain the reviews,
    creates a column of cleaned data, then uses that to
    create a column without stopwords that is lemmatized,
    performs a train-validate-test split, and returns train, validate, and test.
    """
    # Remove any nulls found in the pros and cons section of the data
    df.dropna(subset=['pros', 'cons'], inplace=True)
    
    # Iterate through text columns
    for col in text_cols:
        # Clean text
        df[f'{col}_cleaned'] = df[col].apply(lambda x: clean_strings(x))
        # Lemmatize text
        df[f'{col}_lemmatized'] = df[f'{col}_cleaned'].apply(lambda x: lemmatize(remove_stopwords(x)))

    # Return train, validate, and test dataframes
    return df

################ Preparation Functions #####################
def uni_count_vect(glassdr):
    vectorizer = CountVectorizer()
    X_pros = vectorizer.fit_transform(glassdr["pros_lemmatized"])
    pros_word_df = pd.DataFrame(X_pros.toarray(), columns=["pros_" + word for word in vectorizer.get_feature_names_out()])
    X_cons = vectorizer.fit_transform(glassdr["cons_lemmatized"])
    cons_word_df = pd.DataFrame(X_cons.toarray(), columns=["cons_" + word for word in vectorizer.get_feature_names_out()])
    df = pd.concat([pros_word_df, cons_word_df], axis=1)
    df["binned_rating_int"] = glassdr.binned_rating_int.values
    return df

def bi_count_vect(glassdr):
    vectorizer = CountVectorizer(ngram_range=(2, 2))  # Set ngram_range to (3, 3) for trigrams
    X_pros = vectorizer.fit_transform(glassdr["pros_lemmatized"])
    pros_word_df = pd.DataFrame(X_pros.toarray(), columns=["pros_" + word for word in vectorizer.get_feature_names_out()])
    X_cons = vectorizer.fit_transform(glassdr["cons_lemmatized"])
    cons_word_df = pd.DataFrame(X_cons.toarray(), columns=["cons_" + word for word in vectorizer.get_feature_names_out()])
    df = pd.concat([pros_word_df, cons_word_df], axis=1)
    df["binned_rating_int"] = glassdr.binned_rating_int.values
    return df

def tri_count_vect(glassdr):
    vectorizer = CountVectorizer(ngram_range=(3, 3))  # Set ngram_range to (3, 3) for trigrams
    X_pros = vectorizer.fit_transform(glassdr["pros_lemmatized"])
    pros_word_df = pd.DataFrame(X_pros.toarray(), columns=["pros_" + word for word in vectorizer.get_feature_names_out()])
    X_cons = vectorizer.fit_transform(glassdr["cons_lemmatized"])
    cons_word_df = pd.DataFrame(X_cons.toarray(), columns=["cons_" + word for word in vectorizer.get_feature_names_out()])
    df = pd.concat([pros_word_df, cons_word_df], axis=1)
    df["binned_rating_int"] = glassdr.binned_rating_int.values
    return df

################ Main Function #####################
def wrangle_glassdoor(filepath = "./data/glassdoor_reviews.csv"):
    """
    Acquires and prepares glassdoor reviews.
    Returns train, validate, and test dataframes
    """
    # Acquire reviews
    glassdoor = pd.read_csv(filepath, index_col=0)

    # Prep and split the data
    df = prepare_data(glassdoor, ["pros", "cons"])
    
    bin_edges = [2.0, 2.9, 3.9, 4.9]

    # Define bin labels
    bin_labels = ['Two', 'Three', 'Four']
    bin_label_int = [2, 3, 4]
    
    # Bin the 'Values' column
    df['binned_rating'] = pd.cut(df['rating'], bins=bin_edges, labels=bin_labels)
    df['binned_rating_int'] = pd.cut(df['rating'], bins=bin_edges, labels=bin_label_int)
    
    df = df[df.binned_rating_int != 2]    
    
    # Split the data
    train, validate, test = split_data(df)
    count_vect_train, count_vect_validate, count_vect_test = split_data(uni_count_vect(df))
    bi_count_vect_train, bi_count_vect_validate, bi_count_vect_test = split_data(bi_count_vect(df))
    tri_count_vect_train, tri_count_vect_validate, tri_count_vect_test = split_data(tri_count_vect(df))
    
    # Return train, validate and test
    return (train, validate, test), (count_vect_train, count_vect_validate, count_vect_test), (bi_count_vect_train, bi_count_vect_validate, bi_count_vect_test),(tri_count_vect_train, tri_count_vect_validate, tri_count_vect_test)

################ Main Function #####################
