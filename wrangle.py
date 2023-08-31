################# IMPORTS #####################
#  data manipulation
import pandas as pd
import numpy as np

# Web scraping
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import time

# natural language processing
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
import unicodedata
import re
import os
# split data
from sklearn.model_selection import train_test_split

# Quiet all warnings
import warnings
warnings.filterwarnings('ignore')

############## ACQUISITION FUNCTIONS ########################
def selenium_scrap():
    for step in range(100, 103):
        data = []

        for page_number in range(1):

            url = f"https://www.glassdoor.com/Reviews/index.htm?overall_rating_low=1&page={step}&locId=1&locType=N&locName=United%20States&filterType=RATING_OVERALL"
            # access company page
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service = service)
            driver.get(url)
            review_links = driver.find_elements(By.XPATH, '//span[@class="css-u9lko5 euttuq60"]')
            # Find employer name tags
            # Find employer rating tags


            review_urls = []
            for ele in driver.find_elements(By.XPATH, '//a[@data-test="cell-Reviews-url"]'):
                review_urls.append(ele.get_attribute("href"))
            # Create a list to store data

            # Initialize webdriver service
            # Loop through company URLs
            for i, review_url in enumerate(review_urls):
                company_data = {'url': review_url,
                                'pros': '',
                                'cons': ''}
                for i in range(10):
                    driver = webdriver.Chrome(service=service)
                    driver.get(review_url)
                    # Extract pros and cons
                    pros = [pro.text for pro in driver.find_elements(By.XPATH, "//span[@data-test='pros']")]
                    cons = [con.text for con in driver.find_elements(By.XPATH, "//span[@data-test='cons']")]
                    # Add to company_data
                    company_data['pros'] += ' '.join(pros)
                    company_data['cons'] += ' '.join(cons)
                    try:
                        # Try to click the pagination next button
                        pagination_next = driver.find_element(By.XPATH, '//button[@data-test="pagination-next"]')
                        pagination_next.click()
                        next_url = driver.current_url
                    except:
                        # If there's no next page, break the loop
                        print('error')
                        break
                    driver.quit()
                data.append(company_data)
        # Create a DataFrame from the collected data
        df = pd.DataFrame(data)
        csv_filename = f'./data_files/part3_{step}.csv'  # Change this to your desired filename
        df.to_csv(csv_filename, index=False)
        print(step)
        time.sleep(300)
        
def combine_all_files():
    data_frames = []
    data_folder = os.listdir("./data_folder")

    for folder in data_folder:
        folder_path = os.path.join('./data_folder', folder)
        parts = [".DS_Store", ".ipynb_checkpoints"]
        if folder not in parts:
            files = pd.Series(os.listdir(folder_path))
            for ele in files:
                if ele.endswith(".csv"):
                    df = pd.read_csv(f"./data_folder/{folder}/{ele}")
                    data_frames.append(df)

    combined_df = pd.concat(data_frames, ignore_index=True)
    combined_df.to_csv("glassdoor_full.csv", mode= "w")
    return combined_df


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


def prep_readmes(df, cols:str=[]):
    """
    Takes in the dataframe and the column name that contains the corpus data, creates a column of cleaned data, then uses that 
    to create a column without stopwords that is lemmatized, performs a train-validate-test split, and returns train, validate,
    and test.
    """
    for idx, col in enumerate(cols):
        # Initialize a list to collect cleaned elements in the for-loop below
        cleaned_row = []

        # Iterate through the readme_content values...
        for i in df[col].values:

            # Clean each value in the column and append to the 'cleaned_row' list
            cleaned_row.append(clean(i))
        
        if idx == 0:
            # Assign the clean row content to a new column in the dataframe named 'cleaned_content
            df = df.assign(pros_cleaned_content=cleaned_row)
            
            # Using a lambda, lemmatize all values in the 'cleaned_content' column and assign to a new column called 'lemmatized'
            df[f'{col}_lemmatized'] = df['pros_cleaned_content'].apply(lambda x: lemmatize(remove_stopwords(x)))
        if idx == 1:
            # Assign the clean row content to a new column in the dataframe named 'cleaned_content
            df = df.assign(cons_cleaned_content=cleaned_row)
            # Using a lambda, lemmatize all values in the 'cleaned_content' column and assign to a new column called 'lemmatized'
            df[f'{col}_lemmatized'] = df['cons_cleaned_content'].apply(lambda x: lemmatize(remove_stopwords(x)))

    # Split the dataframe (70/15/15)
    train, validate, test = split_readmes(df)
    
    # Return train, validate, and test dataframes
    return train, validate, test

################ MAIN FUNCTION #####################
def wrangle_readmes():
    """
    Acquires the Glass door data then preps it. Returns train, validate, and test dataframes
    """
    glassdoor = combine_all_files()
    
    # remove any nuls found in the pros and cons section of the data
    glassdoor = glassdoor.dropna()
    # get company names
    glassdoor['name'] = glassdoor['url'].apply(lambda url: url[34: url.find('-Reviews')].replace('-', ' '))

    # Perform acquire and then prep the data, store in train, validate, and test dataframes
    train, validate, test = prep_readmes(glassdoor, ["pros", "cons"])
    
    # Return train, validate and test
    return train, validate, test
