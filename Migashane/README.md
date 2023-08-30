<p align="center">
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

# Analyzing Employee Sentiments and Predicting Binned Company Ratings from Glassdoor Reviews
# <bu> NATURAL LANGUAGE PROCESSING PROJECT</bu>
by Jon Ware, Martin Rayes, Victoire Migashane

<!-- ![image](https://github.com/Science-of-the-Fork/nlp-project/assets/131205837/3c286317-2826-45ad-bfb6-8c9ec3a8679f) -->
___

<a id='navigation'></a>
[[Project Description](#project-description)]
[[Statistical Testing Hypothesis and NLP Techniques](#statistical-hypotheses-and-nlp-techniques)]
[[Data Dictionary](#data-dictionary)]
[[Planning_Process](#planning-process)]
[[Project Reproduction](#project-reproduction)]
[[Key Findings](#key-findings)]
[[Statistical and NLP Techniques Findings](#statistical-and-nlp-techniques-findings)]
[[Next Steps](#next-steps)]
[[Conclusion](#conclusion)]
[[Recommendations](#recommendations)]


## <u>Project Description</u>

This project involves a comprehensive analysis of employee reviews collected from Glassdoor, with a specific focus on the pros and cons views provided by employees. The primary goal is to uncover the key drivers that influence a company's Glassdoor star rating score based on the sentiments expressed in these reviews. By utilizing natural language processing (NLP) techniques and machine learning, the project aims to provide companies with actionable insights to enhance their work environment, address concerns, and improve overall company reputation.

## <u>Project Goals</u>

- **Sentiment Analysis of Pros and Cons**: Analyze the sentiment of employee reviews' pros and cons sections to understand the prevailing positive and negative aspects mentioned by employees.

- **Identify Key Drivers**: Determine the main factors and themes driving company Glassdoor star rating scores by analyzing the sentiment trends and topics in the pros and cons views.

- **Predictive Modeling**: Develop a predictive model to associate sentiment-rich features derived from pros and cons views with the corresponding Glassdoor star rating scores.

- **Insight Generation**: Generate actionable insights for companies to understand what aspects contribute to higher or lower star ratings, enabling them to make data-driven decisions for improvement.

   
[Jump to Navigation](#navigation)

## <u>Initial Questions</u>

1. Are there differences in word/words distribution between different binned star rating categories?
- Are there specific keywords that frequently appear in reviews associated with higher or lower star ratings?
   
2. Are reviews generally more positive in the pros section and more negative in the cons section, or is there a mix of sentiments?
- What are the most common positive themes mentioned in the pros reviews?
- What are the most common negative themes mentioned in the cons reviews?

3. Do longer reviews have more impanct to the overall star rating compared to shorter reviews?
   
4. Are there instances where reviews express positive sentiments despite lower star ratings or vice versa?

5. Are there any topics that seem to have a stronger association with higher or lower star ratings between all the companies?

[Jump to Navigation](#navigation)

## Data Dictionary

The initial dataset comprised scraped columns.

| Column Name      | Description                                              | Data Type|
|------------------|----------------------------------------------------------|----------|
| Organization Name| The name of the company or organization being reviewed.  | string   |
| Overall Stars    | The overall star rating given by employees on Glassdoor. | string   |
| Pros             | The positive aspects or highlights mentioned in reviews. | string   |
| Cons             | The negative aspects or criticisms mentioned in reviews. | string   |


The columns attained trough preparation.

| Column Name      | Description                                              | Data Type|
|------------------|----------------------------------------------------------|----------|
| Organization Name| The name of the company or organization being reviewed.  | string   |
| Overall Stars    | The overall star rating given by employees on Glassdoor. | integer   |
| Pros             | The positive aspects or highlights mentioned in reviews. | string   |
| Cons             | The negative aspects or criticisms mentioned in reviews. | string   |

[Jump to Navigation](#navigation)

## <u>Statistical Testing Hypothesis and NLP Techniques </u>

Hypothesis 1 - Chi-squared test of independence to determine if the distribution of programming languages (JavaScript, Python) significantly differs within the READMEs.

alpha = .05
* H0: Programming languages (JavaScript, Python) are not independent of ReadMe
* Ha: Programming languages (JavaScript, Python) are independent of ReadMe
* Outcome: We accept or reject the null hypothesis.


Hypothesis 2 - Term Frequency-Inverse Document Frequency (TF-IDF) analysis to use scores for words across the repository texts, in order to identify the most significant and frequent words. Selecting the top five words based on their TF-IDF scores.

Hypothesis 3 - T-Test will be performed on the top 80 most frequent words in the curated dataset to determine which are the most 5 significant words and their relationship to Programing languages (Python and JavaScript).  Use words for future modeling.

* H0: Word did not show significant relationship to programming language (Python and JavaScript) 
* Ha: Word did show signficant relationship to programming language(Python and JavaScript) 
* Outcome: We accept or reject the null hypothesis 

[Jump to Navigation](#navigation)

## <u>Planning Process</u>

#### Planning
1. Clearly define the problem statement related to Natural Language Processing, determining site to scrape repositories and scope of data to scrape. Formulate intial questions. Keep in mind that GitHub's API and repository content may change over time, so **time stamp intial scrape** and ensure your scraping and data processing methods are adaptable to potential modifications in the API or repository structures.

2. As a preliminary step, identify the scripting language used in each repository by inspecting its primary programming language. This can be extracted from GitHub's repository metadata.

3. Create a detailed README.md file documenting the project's context, dataset characteristics, and analysis procedure for easy reproducibility.

#### Acquisition and Preparation
1. **Acquiring Data from GitHub Readme Files by Scraping the GitHub API** Must secure a GitHub token https://github.com/settings/tokens. Utilize the GitHub API to access the README files of the selected repositories. Extract the README content using API calls for each repository. Ensure you adhere to rate limits and fetch the necessary data efficiently.
2. **Cleaning and Preparing Data Using RegEx and Beautiful Soup Libraries** Process the raw README content to remove HTML tags, code snippets, and other irrelevant elements using Beautiful Soup. Employ regular expressions (RegEx) to clean the text further by eliminating special characters, punctuation, and numbers, while retaining meaningful text.

3. **Cleaning and Preparing Data Using RegEx and Beautiful Soup:** Process the raw README content to remove HTML tags, code snippets, and other irrelevant elements using Beautiful Soup. Employ regular expressions (RegEx) to clean the text further by eliminating special characters, punctuation, and numbers, while retaining meaningful text.

4. Preprocess the data, handling missing values and outliers effectively during data loading and cleansing.

5. Perform feature selection meticulously, identifying influential features impacting the prevalence of the chronic disease through correlation analysis, feature importance estimation, or domain expertise-based selection criteria.

6. Develop specialized scripts (e.g., acquire.py and wrangle.py) for efficient and consistent data acquisition, preparation, and data splitting.

7. Safeguard proprietary aspects of the project by implementing confidentiality and data security measures, using .gitignore to exclude sensitive information.

#### Exploratory Analysis
1. **Exploring Data for Relevant Keyword Grouping Using Bi-grams and Unigrams:** Implement a mechanism to tokenize the cleaned text into words. Create bi-grams (pairs of adjacent words) and unigram (single word or ) from the tokenized text. Calculate the frequency of these word sequences within the repository data.

   This step involves:
   - Tokenization: Split the cleaned text into individual words.
   - Bi-gram Generation: Form pairs of adjacent words to create bi-grams.
   - Unigram Generation: Generate sequences of single word .
   - Frequency Calculation: Count the occurrences of each bi-gram and unigram.

   By analyzing the most frequent bi-grams and unigrams, you can identify keyword groupings that occur frequently in the READMEs. These groupings could represent significant terms, programming concepts, or patterns prevalent across the repositories.

2. Utilize exploratory data analysis techniques, employing compelling visualizations and relevant statistical tests to extract meaningful patterns and relationships within the dataset.

#### Modeling
1. Carefully choose a suitable machine learning algorithm based on feature selection and features engineered, evaluating options like K- Nearest Neighbor, Logistic Regression, Decision Trees, or Random Forests, tailored for the classification regression task.

2. Implement the selected machine learning models using robust libraries (e.g., scikit-learn), splitting the data, systematically evaluating multiple models with a fixed Random State value = 123 for reproducibility.

3. Train the models rigorously to ensure optimal learning and model performance.

4. Conduct rigorous model validation techniques to assess model generalization capability and reliability.

5. Select the most effective model(e.g Logistic Regression), based on accuracy and a thorough evaluation of metrics before selecting best model to test.

#### Product Delivery
1. Assemble a final notebook, combining superior visualizations, well-trained models, and pertinent data to present comprehensive insights and conclusions with scientific rigor.

2. Generate a Prediction.csv file containing predictions from the chosen model on test data for further evaluation and utilization.

3. Maintain meticulous project documentation, adhering to scientific and professional standards, to ensure successful presentation or seamless deployment.

[Jump to Navigation](#navigation)

## <u> How to Reproduce the Final Project Notebook</u> 
To successfully run/reproduce the final project notebook, please follow these steps:

1. Read this README.md document to familiarize yourself with the project details and key findings.
2. Before proceeding, ensure that you have the necessary database GitHub token credentials. Get data set from https://github.com/search?o=desc&q=stars:%3E1&s=forks&type=Repositories and .gitignore for privacy if necessary
3. Clone the nlp_project repository from our The Science of The Fork organization or download the following files: aquire.py, wrange.py or prepare.py, and final_report.ipynb. You can find these files in the organization's project repository.
4. Open the final_report.ipynb notebook in your preferred Jupyter Notebook environment or any compatible Python environment.
5. Ensure that all necessary libraries or dependent programs are installed (e.g. nltk, Beautiful Soup). You may need to install additional packages if they are not already present in your environment.
6. Run the final_report.ipynb notebook to execute the project code and generate the results.
By following these instructions, you will be able to reproduce the analysis and review the project's final report. Feel free to explore the code, visualizations, and conclusions presented in the notebook.


## <u>Key Findings</u>

* <span style ='color:#151E3D'> 1. Although almost equally distributed, between JavaScript and Python, Python language exhibits greater prevalence within the content of the READMEs at Python 54% to JavaScript at 46%. 
* <span style ='color:#151E3D'> 2. The word "model" scored the highest across the repositiory ReadMe texts and was the most frequently used, especially in the Python language.The word 'function' was the second highest across all the forked repository ReadMe curated. Of note it was the most frequently used word  JavaScript language       
* <span style ='color:#151E3D'> 3. Bigram frequency analysis. Identify and count the most common bigrams (pairs of adjacent words) in JavaScript code segments.
    

| JavaScript Bigram     |Count |
|-----------------------|------|
|1. make sure           | 30   |
|2. npm install         | 21   |
|3. pull request        | 18   |
|4. task implement      | 15   |
|5. implement function  | 14   |

* <span style ='color:#151E3D'> 4. The most common bigrams (pairs of adjacent words) in JavaScript code Top 100 Forked ReadMe's were:

4. Bigram frequency analysis. Identify and count the most common bigrams (pairs of adjacent words) in Python code segments.

|Python Bigram         |Count |
|----------------------|------|
|1.released paper      | 579  |
|2.et al               | 121  |
|3.ai released         | 105  |
|4.research released   | 102  |
|5.language model      |  85  |

5. **JavaScript code segments, what are the most frequent unigrams encountered?**

| JavaScript Unigram    |Count |
|-----------------------|------|
|1. npm                 | 68   |
|2. indexjs             | 59   |
|3. alarm               | 50   |
|4. dom                 | 47   |
|5. we're               | 39   |

6. **Python code segments which unigrams have highest frequency?**

|Python Unigram        |Count |
|----------------------|------|
|1.paper               | 844  |
|2.transformer         | 359  |
|3.pertaining          | 209  |
|4.face                | 191  |
|5.research            | 184  |


[Jump to Navigation](#navigation)    

## <u>Statistical and NLP Techniques Findings: </u>

Hypothesis 1 - Chi-squared test determined that Programing languages were not of READMEs.We accept the null (H0) hypothesis 

Hypothesis 2 - The top five words based on their TF-IDF scores.    
    
|Word       |TF-IDF Score |
|-----------|-------------|
|1. model   |  578        |
|2. function|  298        |
|3. test    |  242        |
|4. use     |  232        |
|5. code    |  255        |    
    
Hypothesis 3 - T-Test of the top 5 most significant words:
1. learning
2. test
3. library
4. create
5. line
* Outcome - Ha: They all rejected the null hypothesis and showed relationship to the program language.
* 
[Jump to Navigation](#navigation)

## <u>Conclusion</u>
In the realm of Natural Language Processing (NLP), our analysis delved into the linguistic patterns and language prevalence within the READMEs of the top 100 most forked repositories on GitHub. Our findings uncovered several intriguing insights. Firstly, we observed a slightly higher prevalence of the Python language within README contents, constituting 54% of the distribution compared to JavaScript's 46%. Delving into the most frequently used words, "model" surfaced as a dominant term across the repository ReadMe texts, particularly pronounced within the Python language. Additionally, the word "function" held significance across all repositories, notably emerging as the most frequent term in JavaScript. Notably, we engaged in bigram frequency analysis, revealing notable pairs of adjacent words in JavaScript code segments, such as "function expression" and "npm test." 

Our investigation extends beyond linguistics, embracing statistical and machine learning methodologies. The Chi-squared test confirms the intertwined relationship between programming language distribution and README content. Furthermore, t-tests on the top five most significant words unveil substantial frequency differences, deepening our understanding of language nuances. Incorporating classification models (Decision Tree, Random Forest, K-Nearest Neighbor, Logistic Regression, we will not use any of the classification models as none beat baseline and the model that ran the best was RandomForest which produced a Test score of 38%, 16% below baseline accuracy.  In essence, our analysis encapsulates the multifaceted landscape and challenges of predicting programming languages of GitHub README.md, while offering insights on words that resonate with Github users, developers and the evolving world of open-source coding practices.
    
[Jump to Navigation](#navigation)

## <u>Next Steps</u>

1. **Enhance Classification Model Performance:**
Although our classification models didn't perform well, there's an opportunity to enhance their performance by considering the following:

* Feature Engineering: Experiment with more advanced text preprocessing techniques like word embeddings (Word2Vec, GloVe) or pre-trained language models (BERT, GPT) to capture semantic relationships.
Hyperparameter Tuning: Optimize hyperparameters for your classification models to improve their accuracy and robustness.
Ensemble Learning: Combine the predictions of multiple models using ensemble methods like stacking or boosting, which can often lead to better results.

2. **Trigram Exploration:** Further explore trigrams or three consecutive word use to find insights. 

[Jump to Navigation](#navigation)

## <u>Recommendations</u>
1. **Language-Specific Documentation Enhancement:** Recognizing Python's higher prevalence in READMEs, capitalize on this trend by enhancing language-specific documentation. Develop comprehensive examples, tutorials, and best practices that cater to Python's prevalent usage. This approach will aid developers, especially newcomers to Python, in quickly grasping essential concepts and utilizing the language's features effectively.
3. **Further Explore  CountVectorization and TF-IDF:** 
3. **Code Reusability and Patterns:** Responding to the significance of "model" and "function" as highly frequent terms, prioritize the promotion of code reusability and design patterns associated with these concepts. Craft libraries, modules, or templates that encapsulate common functionalities or algorithms related to models and functions. This strategic approach fosters efficient development, encourages uniform coding practices, and contributes to cohesive project architectures.

[Jump to Navigation](#navigation)    

    





 