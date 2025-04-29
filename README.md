# SMS Spam Detection

## Overview
This project is a machine learning-based SMS spam detection system developed entirely in Google Colab. It leverages natural language processing (NLP) techniques to classify SMS messages as either **spam** or **ham (not spam)** using multiple classification models.

## Dataset
- **Source:** [Kaggle - SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Total Messages:** 5,574
- **Labels:** `ham` (legitimate) and `spam` (unwanted)

The dataset is provided as a single text file where each line contains a label and the corresponding message.

## Data Preprocessing
The preprocessing pipeline includes:
- Lowercasing text
- Removing punctuation, numbers, and stopwords
- Tokenization using NLTK
- Transforming text into numerical form using:
  - **CountVectorizer** (Bag of Words model)
  - **TF-IDF Transformer** (to scale the importance of words)

## Model Training and Evaluation
Four machine learning models are trained and evaluated:
- **Logistic Regression**
- **Multinomial Naive Bayes**
- **Support Vector Machine (SVM)**
- **Random Forest Classifier**

### Evaluation Metrics:
- **Accuracy**
- **Precision, Recall, F1-Score** (via classification report)
- **Confusion Matrix** (for detailed error analysis)

Each model is trained using an 80-20 train-test split, and the best-performing model is identified based on evaluation results.

## Visualizations
- **Class Distribution:** Displays imbalance between spam and ham.
- **Word Clouds:** Shows most frequent words in spam and ham messages.
  - Common spam terms include: `free`, `win`, `cash`, etc.
  - Ham messages tend to contain more casual conversation.

## Interactive Demo (Colab Only)
The notebook includes a custom input section where users can:
1. Enter a new SMS message.
2. Instantly see whether it’s classified as spam or ham.

This section demonstrates real-time inference using the trained model.

## How to Run (on Google Colab)
1. Open the notebook: `sms_spam_detection.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Upload the dataset (`SMSSpamCollection.txt`) to the Colab environment.
3. Run all cells in order:
   - Preprocessing
   - Model Training
   - Evaluation
   - Visualization
   - Interactive Testing

Dependencies like `nltk`, `sklearn`, `matplotlib`, and `wordcloud` are installed within the notebook.

## Future Improvements
- Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
- Use of deep learning models like LSTMs or Transformers
- Handling class imbalance using SMOTE or class weighting
- Deployment as a web application using Streamlit or Flask

