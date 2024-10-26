# Tweet Sentiment Analysis: Positive or Negative Classifier

## Overview
This project is a sentiment analysis model that predicts whether tweets are positive or negative. The model was built using machine learning techniques, and the entire project has been deployed on a web application for easy access and interaction.

---

## Project Contents
- **`data/`**: Contains the dataset used for training and testing the model.
- **`notebooks/`**: Jupyter notebooks used for exploratory data analysis (EDA) and model building.
- **`model/`**: Saved model files and other serialized objects.
- **`app/`**: The Flask web application for model deployment.


---

## Project Steps

### 1. Data Collection
   - **Description**: Collected a dataset of tweets labeled with sentiment (positive or negative).
   - **Data Source**: Mention any public datasets, or describe the data scraping if used (I used Kaggle ).

### 2. Data Preprocessing
   - **Text Cleaning**: Removed unwanted characters, stopwords, and performed stemming or lemmatization.
   - **Tokenization**: Converted sentences into word tokens.
   - **Vectorization**: Transformed text data into numerical format using techniques like TF-IDF or Word2Vec.

### 3. Exploratory Data Analysis (EDA)
   - **Visualization**: Analyzed the data distribution, common words in positive and negative tweets, etc.
   - **Insights**: Summarized any patterns observed in the data.

### 4. Model Building
   - **Algorithm**: Chose a suitable algorithm (I used Logistic Regression).
   - **Training**: Trained the model using the preprocessed data.
   - **Evaluation**: Achieved an accuracy of 0.77, tuning hyperparameters to improve performance.

### 5. Model Saving
   - **Serialization**: Saved the trained model using `pickle` or `joblib` for future use.

### 6. Building the Web Application
   - **Framework**: Used Flask to create a web app interface.
   - **Integration**: Integrated the model into the Flask app to accept user inputs and display predictions.
   - **User Interface**: Designed a simple form for users to input their tweet and receive a sentiment prediction.

### 7. Deployment
   - **Platform**: Mentioned where the app is hosted (e.g., Heroku, AWS, or locally).
   - **Deployment Steps**: Set up the environment, deployed the app, and ensured the model runs seamlessly online.

