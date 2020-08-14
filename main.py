import re
import time
import math
import warnings
from collections import Counter, defaultdict

import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, normalized_mutual_info_score
from sklearn.metrics.classification import accuracy_score, log_loss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
from mlxtend.classifier import StackingClassifier


def preprocessing_text(text, text_df, index, col):
    """
    Pre-process the text and updates the dataframe

    Args:
    text : str
        text to pre-process
    text_df : df
        Dataframe containinf the texts to pre-process
    col : int
        Column of the dataframe
    index : int
        Index of the dataframe
    
    """
    if type(text) is not int:
        preprocessed_text = ""

        text = re.sub(
            "[^a-zA-Z0-9\n]", " ", text
        )  # Replace every special character with a space
        text = re.sub("\s+", " ", text)  # Replace multiple space with single space
        text = text.lower()

        for word in text.split():
            if word not in stop_words:
                preprocessed_text += word + " "

        text_df[col][index] = preprocessed_text


def exploratory_data_analysis():
    """
    Perform exploratory data analysis and return the processed dataset

    Returns:
    final_result : Dataframe
          Result obtained after preprocessing steps
    """
    # Reading Gene and Vairation data
    data_variants = pd.read_csv("/Users/lf/cancer_detection/datasets/training_variants")

    # print("Number of data points : ", data_variants.shape[0])
    # print("Number of features : ", data_variants.shape[1])
    # print("Features : ", data_variants.columns.values)
    # print(data_variants.head())

    # Reading test data
    data_texts = pd.read_csv(
        "/Users/lf/cancer_detection/datasets/training_text",
        sep="\|\|",
        engine="python",
        names=["ID", "TEXT"],
        skiprows=1,
    )
    # print("Number of data points : ", data_texts.shape[0])
    # print("Number of features : ", data_texts.shape[1])
    # print("Features : ", data_texts.columns.values)
    # print(data_texts.head())

    # Text pre-processing stage
    start_time = time.clock()
    for index, row in data_texts.iterrows():
        if type(row["TEXT"]) is str:
            preprocessing_text(row["TEXT"], data_texts, index, "TEXT")
        else:
            print("There is not text description for id:", index)
    # print(
    #     "Time taken for pre-processing the text : ",
    #     time.clock() - start_time,
    #     "seconds",
    # )

    # Merge both dataframes
    final_result = pd.merge(data_variants, data_texts, on="ID", how="left")
    # print(final_result.head())

    # Check for null value
    # print(final_result[final_result.isnull().any(axis=1)])

    # Remove null values by substituting null values by comnination of Gene and Variation value
    final_result.loc[final_result["TEXT"].isnull(), "TEXT"] = (
        final_result["Gene"] + " " + final_result["Variation"]
    )

    # Check if null value still exists
    # print(final_result[final_result.isnull().any(axis=1)])

    return final_result


def plot_distribution(class_distribution, title, xlabel, ylabel):
    class_distribution.plot(kind="bar")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.show()


def show_distribution(y_train, y_test, y_cv):
    # it returns a dict, keys as class labels and values as the number of data points in that class
    train_class_distribution = y_train["Class"].value_counts().sort_index()
    test_class_distribution = y_test["Class"].value_counts().sort_index()
    cv_class_distribution = y_cv["Class"].value_counts().sort_index()

    plot_distribution(
        train_class_distribution,
        "Distribution of y_i in train data",
        "Class",
        "Data points per Class",
    )

    # ref: argsort https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html
    # -(train_class_distribution.values): the minus sign will give us in decreasing order
    sorted_yi = np.argsort(-train_class_distribution.values)
    for i in sorted_yi:
        print(
            "Number of data points in class",
            i + 1,
            ":",
            train_class_distribution.values[i],
            "(",
            np.round((train_class_distribution.values[i] / x_train.shape[0] * 100), 3),
            "%)",
        )

    print("-" * 80)

    plot_distribution(
        test_class_distribution,
        "Distribution of y_i in test data",
        "Class",
        "Data points per Class",
    )

    # ref: argsort https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html
    # -(test_class_distribution.values): the minus sign will give us in decreasing order
    sorted_yi = np.argsort(-test_class_distribution.values)
    for i in sorted_yi:
        print(
            "Number of data points in class",
            i + 1,
            ":",
            test_class_distribution.values[i],
            "(",
            np.round((test_class_distribution.values[i] / x_test.shape[0] * 100), 3),
            "%)",
        )

    print("-" * 80)

    plot_distribution(
        cv_class_distribution,
        "Distribution of y_i in cross validation data",
        "Class",
        "Data points per Class",
    )

    # ref: argsort https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html
    # -(cv_class_distribution.values): the minus sign will give us in decreasing order
    sorted_yi = np.argsort(-cv_class_distribution.values)
    for i in sorted_yi:
        print(
            "Number of data points in class",
            i + 1,
            ":",
            cv_class_distribution.values[i],
            "(",
            np.round((cv_class_distribution.values[i] / x_cv.shape[0] * 100), 3),
            "%)",
        )


if __name__ == "__main__":
    # Load stopwords from nltk library
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))

    preprocessed_data = exploratory_data_analysis()

    # Splitting data into train, test and cross validation
    y_true = preprocessed_data[["Class"]]
    x_true = preprocessed_data.drop(["Class"], axis=1)

    preprocessed_data.Gene = preprocessed_data.Gene.str.replace("\s+", "_")
    preprocessed_data.Variation = preprocessed_data.Variation.str.replace("\s+", "_")

    print("Feature columns in dataset: ")
    print(x_true.head())
    print()
    print("Target columns in dataset: ")
    print(y_true.head())

    # split the data into test and train by maintaining same distribution of output variable 'y_true' [stratify=y_true]
    x_train, x_test, y_train, y_test = train_test_split(
        x_true, y_true, stratify=y_true, test_size=0.2
    )
    # split the train data into train and cross validation by maintaining same distribution of output variable 'y_train' [stratify=y_train]
    x_train, x_cv, y_train, y_cv = train_test_split(
        x_train, y_train, stratify=y_train, test_size=0.2
    )

    print("Number of data points in train data:", x_train.shape[0])

    # View distribution of train, test and cross-validate
    show_distribution(y_train, y_test, y_cv)
