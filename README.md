<a name="top"></a>

# Predicting Company Success: Analyzing Glassdoor Reviews for Insights

by Victoire Migashane, Jonathan Ware, and Martin Reyes

<p align="left">
  <a href="https://github.com/MigashaneVictoire" target="_blank">
    <img alt="Victoire" src="https://img.shields.io/github/followers/MigashaneVictoire?label=Follow Victoire&style=social" />
  </a>
  <a href="https://github.com/JonathanTWare" target="_blank">
    <img alt="Fonathan" src="https://img.shields.io/github/followers/JonathanTWare?label=Follow Jonathan&style=social" />
  </a>
  <a href="https://github.com/martin-reyes" target="_blank">
    <img alt="Martin" src="https://img.shields.io/github/followers/martin-reyes?label=Follow Martin&style=social" />
  </a>
</p>

## Description

This project aims to predict Glassdoor company star ratings by analyzing employee reviews, incorporating both the positive and negative aspects. It employs natural language processing and machine learning classification techniques to provide insights, enhance the work environment, increase employee satisfaction, and improve the overall reputation of the company.

We achieved this by:

- Scraping glassdoor.com for companies and company reviews using Selenium.
- Collecting at least 1000 companies and 100 reviews for each company.
- Identifying keywords that are deciding factors in a company's score.
- Creating unigrams, bigrams, and trigrams.

## Goals

Our project's primary objectives were to:

- Predict Glassdoor company star ratings.
- Analyze the impact of word count on sentiment.
- Explore correlations between word frequency (tf) and sentiment.
- Investigate the influence of inverse document frequency (idf) on word sentiment.
- Determine whether words with higher tf-idf scores tend to have specific sentiments.
- Assess significant differences in sentiment between different documents or groups of documents.
- Identify specific words with significantly different sentiment scores compared to the overall sentiment of the documents they appear in.

## Acquire Data

We acquired our data through the following steps:

1. Scraped glassdoor.com for companies and company reviews.
2. Scraped 10 pages of reviews for each company.
3. Saved the data into a [csv file](glassdoor_reviews.csv).

## Prepare Data

Data preparation involved the following steps:

1. Removed punctuation, numbers, extra whitespaces, long words, accented/special characters, and stopwords.
2. Tokenized the data.
3. Lemmatized words.
4. Created unigrams, bigrams, and trigrams.
5. Created count vectorization data frames for modeling.

## Data Exploration (EDA)

Data was split into training, validation, and test datasets to avoid bias and data leakage during modeling. Additional exploration questions included:

- Analyzing word distribution differences between different binned star rating categories.
- Examining sentiment in the pros and cons sections.
- Assessing the impact of review length on star ratings.
- Identifying instances of reviews with contradictory sentiments.
- Discovering words that uniquely identify pros and cons.

## Modeling

We used machine learning classification with accuracy as the evaluation metric. Our baseline model achieved an accuracy of 70%. We utilized various classification models and achieved the following results:

| Model              | Train Accuracy   | Validation Accuracy |
| :----------------- | --------------- | ------------------- |
| Decision Tree      | 91%             | 65%                 |
| Random Forest      | 100%            | 64%                 |
| KNN                | 74%             | 67%                 |
| Logistic Regression| 72%             | 67%                 |
| Naive Bayes        | 69%             | 64%                 |
| XG Boost           | 80%             | 67%                 |

Our best model, Logistic Regression, achieved a test accuracy of 70%, matching our baseline predictions.

## Conclusion

Our analysis of Glassdoor employee reviews identified work-life balance as the primary key factor for employee satisfaction. The Logistic Regression model performed well on the test dataset, achieving a 70% accuracy score, consistent with our baseline predictions.

### Summary

Our project successfully predicted Glassdoor company star ratings based on employee reviews and highlighted the importance of work-life balance in employee satisfaction.

### Next Steps

In future iterations, we plan to:

- Identify what doesnt drive company ratings.
- Explore additional features and data sources to enhance model performance.
- Investigate the impact of company size, and industry on ratings.

## Data Dictionary

| Column         | Description                                 | Data Type       |
|-----------------|---------------------------------------------|-----------------|
| url            | The webpage location on the internet.      | Text or String  |
| pros           | Pros mentioned in the reviews for the company. | Text or String  |
| cons           | Cons mentioned in the reviews for the company. | Text or String  |
| name           | Name of the company.                        | Text or String  |
| rating         | The overall Glassdoor star rating.         | Numeric (e.g., Decimal or Float) |
| ceo_approval   | The percentage of reviewers that approve of the CEO. | Numeric (e.g., Integer or Decimal) |
| recommended    | The percentage of reviewers that recommend working for the company. | Numeric (e.g., Integer or Decimal) |

[Back to top](#top)
