# Project 3: Classification Modeling and Natural Language Processing of Reddit posts

### Problem Statement

In this project, we were tasked with creating classification models to predict the subreddit of origin based on documents of text scraped from posts on various subreddits. While my initial goal was to use natural language processing techniques to determine the differentiators between the subreddits for the two most popular sports leagues in the United States, the NFL and the NBA. However, after my initial assessment, I wanted to dive deeper, and compare these leagues to their collegiate equivalents to see how they differentiate in their language.

My goal in analyzing the language in all four of these subreddits was to determine which topics or trends separate the Reddit discussions of these leagues the most. I will look at the most important word features for each model and asses the performance of each model to draw my conclusions.

### Executive Summary

The objective of this report is to predict a Reddit post's Subreddit of origin based on the classification of words in a document of its text. Using Reddit's built-in .json API functionality, I was able to acquire thousands of posts from four different subreddits: [r/nba](https://www.reddit.com/r/nba/), [r/nfl](https://www.reddit.com/r/nfl/), [r/cfb](https://www.reddit.com/r/CFB/), and [r/CollegeBasketball](https://www.reddit.com/r/CollegeBasketball/). I then cleaned each document of text using various functions in Python and Regular Expressions, deleted duplicate posts, and combined the text from each posts title and "self post" element.

I divided the posts from the four subreddits into three separate dataframes, each with similar features. The first dataframe contains the text from posts from r/nba and r/nfl, the professional sports leagues. The second dataframe contains the the text from posts from r/nba and r/CollegeBasketball, basketball subreddits. The third contains posts from r/nfl and r/cfb, football subreddits. While I was able to obtain a sizeable number of posts from all four subreddits, r/nba subset of posts is much larger due to the nature of the subreddit's index. I do not believe the slight unbalance between these posts and other subreddits negatively affected any of my models. After the dataframes were collected I found the base percentage of each category to establish a baseline for my model.

One of my main interests during my modeling process was to determine which words were more weighted with greater importance when predicting the subreddit of origin, rather the simply count of certain predictive words. So while I did utilize a Count Vectorization model, I found that the weighted Vectorization produced with ScikitLearn's term frequency–inverse document frequency functionality, TfidfVectorizer, produced more informative results when evaluating each model.

I used a Multinomial Naive Bayes (NB) model with both a Count Vectorizer and a TF-IDF Vectorizer. I used ScikitLearn's GridSearch Cross Validation to find the optimal hyperparamters for each model, and ran the models again with the optimal hyperparameters.

The TF-IDF model posted an accuracy score of 93.39% on the testing set with a weighted precision of 94%. This model correctly predicted whether a post was in r/nfl 98% of the time, and r/nba 91%. For this model, the most important words for the model in classifying a post from r/nba referenced free agency moves like "westbrook", "kawhi", "lebron", while the most important words in classfying a post from r/nfl were terms found in discussions about the actual football game, like "highlight", "yard", and "td".

The Count Vectorized model posted an accuracy score of 95.37% on the testing set with a weighted precision of 95%. This model correctly predicted whether a post was in r/nfl 97% of the time, and r/nba 93%. The most important words for the model in classifying a post from r/nba, again, referenced free agency moves like "westbrook", "kawhi", "lebron", but weighted more related terms like "trade" and "pick" as important. The most important words in classfying a post from r/nfl were slightly different from the TF-IDF model, with "day" having the highest count, and more generalized terms like "round" and "pass" being important in predicting r/nfl posts.

To suppplement the NB models, I also ran a TF-IDF Vectorizor and the Support Vector Machine Classifier on the data collected from r/nba and r/nfl. This model did not score as highly as the Multinomial NB models, but still performed very well, posting an accuracy score of 93.14% and a weighted precision of 93%. The important features selected by this model were similar to the NB classifiers, but tended to weight proper nouns like player names, team names, and even journalists, higher. However, more general terms related to the subject matter, like "basketball" and "football", were weighted heavily as well.

Given the subject matter differences between r/nba and r/nfl, I wanted to run the same models on subreddits that had similar subjects. So, I wanted to compare the two subreddits to their collegiate equivalents, r/CollegeBasketball and r/cfb (NCAA College Footall).

First, I compared r/nba and r/CollegeBasketball with a SVM Classifier with TF-IDF Vectorization. This model performed well, posting an accuracy score of 91.39% and a weighted precision of 91%. Like the SVM classifier for r/nba and r/nfl, this model tended to weight proper nouns like player names and team names high. Terms like "tournament", "transfer", and "commit" were given importance in r/collegebasktball as they used more widely those discussions than for r/nba, but instead of players, we find the names of coaches like "calipari" to be weighted heavily. Some interesting differentiators were "basketball" and "gif", which are likely common in either subreddit, but were weighted heavily toward r/CollegeBasketball.

Finally, I ran the same TF-IDF Vectorizor and SVM Classifier on the data collected from r/nfl and r/cfb. This model also performed well, posting an accuracy score of 92.43% and a weighted precision of 93%. Like both SVM classifiers before, this model weighted proper nouns high, but in the case of r/cfb, there were very little in terms of players or coaches names. Related terms like "commits", "star", and "conference" were given importance in r/cfb.  Like r/CollegeBasketball, common words like "basketball" and "tweet"  were weighted heavily toward the collegiate subreddit, r/cfb.

### Data Dictionary

You may find decription of the data found in the Reddit API .json in the API's [documentation](https://github.com/reddit-archive/reddit/wiki/JSON).

I used three datasets for the four subreddits. All having the same features:

|Feature|Type|Dataset|Description|
|---|---|---|---|
|text|object|r/nfl, r/nba, r/CollegeBasketball, r/cfb| The combined strings of 'title' and 'selftext' taken from the subreddit .json|
|name|object|r/nfl, r/nba, r/CollegeBasketball, r/cfb| The unique identifier for each post|
|subreddit|object|r/nfl, r/nba, r/CollegeBasketball, r/cfb| Name of the post's subreddit of origin|
|is_nba|int|r/nfl, r/nba, r/CollegeBasketball| Classification column for indicating whether a post originated from r/nba, used alongside posts from r/nfl and r/CollegeBasketball|
|is_nfl|int|r/nfl, r/cfb| Classification column for indicating whether a post originated from r/nfl, used alongside posts from r/cfb|


### Conclusions and Recommendations

All of the models that classified all iterations of the subreddit data score very highly, but the Count Vectorized Multinomial Naive Bayes was the best performing and most consistent across training and testing data. The Support Vector Machine models did not perform as well as the NB models, but gave me a better understanding of which words/terms were critical in classifying the posts.

Across all models analyzed in this project, they were able to give me an understanding of what type of content is important in classifying the posts between the subreddits. For r/nba, it is very clear that player names and transactions are a very frequent topic of posts, especially given the timing of the data taken. For r/nfl, the topics tended to be more team oriented, but game specific terms like "td" and "yard" were important across all models. For the collegiate variants, its very clear school pride is important for post topics, as the names of schools are important for both, but conferences are more important for r/cfb, while coaches are mentioned more for r/CollegeBasketball.

### Sources Used
- https://github.com/reddit-archive/reddit/wiki/JSON
- https://www.reddit.com/r/nba/
- https://www.reddit.com/r/nfl/
- https://www.reddit.com/r/cfb/
- https://www.reddit.com/r/collegebasktball/
- https://medium.com/@aneesha/visualising-top-features-in-linear-svm-with-scikit-learn-and-matplotlib-3454ab18a14d
