# Fake-News-Detection-with-Sentiment-analysis-ANN

# Overview
This project aimed to develop a system for classifying news articles as real or fake using Artificial Neural Networks (ANN) and sentiment analysis. The process involved collecting data, preprocessing it, and applying feature engineering to prepare the text data for analysis. An ANN model was then trained on this preprocessed data to identify complex patterns, with sentiment analysis providing insight into the emotional context of the articles. The model's performance was evaluated, and a user-friendly web interface was created using Streamlit for real-time predictions. This integration aims to effectively combat the spread of misinformation.

# About Dataset
The dataset was sourced from The Online Academic Community, University of Victoria, comprising the ISOT Fake News dataset. This dataset includes thousands of fake and truthful articles from various legitimate and unreliable news sites flagged by Politifact.com. The majority of the articles focus on political and World news topics. It consists of two CSV files: "True.csv" with over 12,600 articles from reuter.com, and "Fake.csv" with over 12,600 articles from various fake news outlets. Each article includes the title, text, type, and publication date, primarily collected from 2016 to 2017.
For the dataset [check here](https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/).

# Methodologies

## Data Preprocessing:
1. Data Preprocessing:
      * Merged true and fake datasets with a target flag indicating the text as true (0) or fake (1).
