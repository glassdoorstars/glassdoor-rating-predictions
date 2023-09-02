<a name="top"></a>

# Project: Finding Drivers That Predict Glasswo

by Jonathan Ware, Martin Reyes, and Victoire Migashane



**Description:**

In this project, we aim to collect Glassdoor ratings and pros and cons for various companies and develop classification and regression models to predict Glassdoor star scores. The primary goal is to identify quality improvements in workplaces and enhance the representation of companies in the public eye.

**Goals:**

- collect atleast 1000 companies and 100 reviews for each company.
- predict, using classification and regression, glassdoor star scores for companies.
- find key words that are deciding factors in a company's score.

## Acquire Data

1. Scraped glassdoor.com for companies and company reviews
2. After the companies were scraped, each company had 10 pages of reviews scraped as well.
3. The data is then saved into a [csv file]('glassdoor_reviews.csv').

## Prepare Data
1. DF cleaned:
    1. words were lemmatized
    2. removed punctuation, numbers, extra whitespaces, long words, accented/special characters, and stopwords
    3. created unigrams, bigrams, trigrams.
    
2. Reviews divided into pros and cons turned into features.


## Data Exploration (EDA)

Data is split into training and test data. Analysis is performed on training data to avoid bias and data leakage during modeling. 

Test data is separated to test ML regression models later in the project.

**Does word count affect sentiment?**



**Is there a correlation between word frequency (tf) and sentiment?**



**Does the inverse document frequency (idf) of a word impact its sentiment?**



**Do words with higher tf-idf scores tend to have a specific sentiment?**



**Is there a significant difference in sentiment between different documents (doc) or groups of documents?**



**Do specific words have significantly different sentiment scores compared to the overall sentiment of the documents they appear in?**








## Machine Learning Models: NLP and Classification

**Baseline Model**

- Accuracy for the Baseline model was --

**Best Regression Model: Logistic Regression**

- Model predictions are off by about 6 wins, on average, and explain 77% of the variance.

| Model              | Accuracy   |
| :----------------- | ---------- |
| Baseline           |  %       |
| Logistic Regression  |  %      |
|add more as you go..| %      |


## Conclusion

### Summary

Analysis:
- Statistical tests _____ showed 

Modeling:
- Baseline accuracy was --% on the training set and --% on validation set.
- The only model(s) to beat baseline were ---------, which scores were --% on the training set and --% on the validation set.


### Next Steps
- 


[Back to top](#top)

---


<a name="data-dictionary"></a>

## Data Dictionary

| Column         | Description                                 |
|-----------------|---------------------------------------------|
| url            | where to locate the webpage on internet                     |
| pros       | the pros to the reviews for the company     |
| cons | the cons to the reviews for the company                |
| name     | name of the company    |
| rating            | the overall glassdoor star rating        |



[Back to top](#top)