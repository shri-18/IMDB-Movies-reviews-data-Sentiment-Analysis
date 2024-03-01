# IMDB Reviews Sentiment Analysis using Machine Learning

This project utilizes machine learning techniques to perform sentiment analysis on IMDB movie reviews. The goal is to classify whether a review is positive or negative based on its text content.

## Dataset
The dataset used for this project is the "IMDB Dataset of 50k Movie Reviews" available on Kaggle. You can download the dataset from the following link:
[Kaggle IMDB Dataset](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

## Dependencies
This project relies on the following dependencies:
- Python 3.x
- scikit-learn (sklearn)
- Pandas
- NumPy

You can install the dependencies using pip:


## Usage
1. Clone the repository:
git clone https://github.com/your_username/IMDB-Sentiment-Analysis.git
2. Navigate to the project directory
3. Download the IMDB dataset from Kaggle and place it in the following directory:
[https://kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

4. Run the Python script to train and evaluate the sentiment analysis model:
python basic.py




## Support Vector Machine (SVM) Classifier
This project employs the Support Vector Machine (SVM) algorithm for sentiment analysis. SVM is a powerful supervised learning algorithm capable of performing classification tasks. It works by finding the hyperplane that best separates the classes in a high-dimensional space. In the context of sentiment analysis, SVM learns to classify movie reviews as either positive or negative based on the features extracted from the text.

## How it Works
1. **Data Preprocessing**: The raw text data is preprocessed, which involves steps like tokenization, removing stopwords, and vectorization.
2. **Feature Extraction**: Text data is transformed into numerical feature vectors using techniques like TF-IDF (Term Frequency-Inverse Document Frequency).
3. **Model Training**: The SVM classifier is trained on the feature vectors along with their corresponding labels (positive or negative sentiment).
4. **Model Evaluation**: The trained model is evaluated using a separate test dataset to assess its performance in predicting sentiment.



Feel free to contribute to this project by opening issues or pull requests.
