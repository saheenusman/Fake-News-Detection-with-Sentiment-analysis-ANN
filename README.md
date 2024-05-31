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
      * Dropped unnecessary columns such as title and date.
      * Confirmed no null values were present.
      * Shuffled the data to avoid bias in the model's learning process.
2. Text Preprocessing:
      * Tokenized using TweetTokenizer.
      * Removed special characters with Regular Expression.
      * Lemmatized using WordNetLemmatizer.
      * Converted text to lowercase.
      * Removed stopwords.
      * Transformed the text into numerical representation using TF-IDF Vectorizer.
3. Splitting Data:
      * Split data into training and testing sets (70-30 split).
4. Scaling Training Data:
      * Used MaxAbsScaler to scale the training data, transforming features to a similar scale. MaxAbsScaler scales each feature by its maximum absolute value, preserving data sparsity and ensuring features fall 
        within the range [-1, 1]. This method is particularly useful for text data in fake news detection, maintaining interpretability while preventing outliers from dominating the learning process.
5. Setting Up ANN Model:
      * The model architecture is Sequential with layers added sequentially.
      * Two dense layers:
           * First dense layer with 64 neurons.
           * Second dense layer with a single neuron.
      * First layer activation function: Rectified Linear Unit (ReLU) for introducing non-linearity.
      * Second layer activation function: Sigmoid for binary classification.
      * Input shape for the first layer is determined by the shape of the scaled training data.
      * Compiled using binary cross-entropy loss function.
      * Used Adam optimizer for optimizing model parameters.
      * Accuracy chosen as the evaluation metric.

# Conclusion

Despite achieving a high accuracy of 99%, there is a significant risk of overfitting. Various preprocessing techniques and both simpler and more complex models were experimented with, but results consistently indicated potential overfitting. This suggests that while the model performs well on the training data, it may not generalize effectively to unseen data.

The training data, collected from 2016 to 2017, includes 10,145 articles on World News, 11,272 on Politics, and 23,481 labeled as Fake News. Additionally, it encompasses 1,570 Government News articles, 778 on the Middle East, 783 on US News, 4,459 on Left-wing News, and 6,841 on general politics. Given the specific nature and categories of the dataset, the model may not be generalized for all purposes. The varied topics reflect a diverse range of subjects, yet the model's performance is tailored to this particular distribution, limiting its applicability to other contexts.

Finally, a user-friendly interface was developed using Streamlit to facilitate real-time predictions.
