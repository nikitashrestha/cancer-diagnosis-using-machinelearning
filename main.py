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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB

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


# This function plots the confusion matrices given y_i, y_i_hat.
def plot_confusion_matrix(test_y, predict_y):
    C = confusion_matrix(test_y, predict_y)
    # C = 9,9 matrix, each cell (i,j) represents number of points of class i are predicted class j

    A = ((C.T) / (C.sum(axis=1))).T
    # divid each element of the confusion matrix with the sum of elements in that column

    # C = [[1, 2],
    #     [3, 4]]
    # C.T = [[1, 3],
    #        [2, 4]]
    # C.sum(axis = 1)  axis=0 corresonds to columns and axis=1 corresponds to rows in two diamensional array
    # C.sum(axix =1) = [[3, 7]]
    # ((C.T)/(C.sum(axis=1))) = [[1/3, 3/7]
    #                           [2/3, 4/7]]

    # ((C.T)/(C.sum(axis=1))).T = [[1/3, 2/3]
    #                           [3/7, 4/7]]
    # sum of row elements = 1

    B = C / C.sum(axis=0)
    # divid each element of the confusion matrix with the sum of elements in that row
    # C = [[1, 2],
    #     [3, 4]]
    # C.sum(axis = 0)  axis=0 corresonds to columns and axis=1 corresponds to rows in two diamensional array
    # C.sum(axix =0) = [[4, 6]]
    # (C/C.sum(axis=0)) = [[1/4, 2/6],
    #                      [3/4, 4/6]]

    labels = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # representing A in heatmap format
    print("-" * 20, "Confusion matrix", "-" * 20)
    plt.figure(figsize=(20, 7))
    sns.heatmap(
        C, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("Original Class")
    plt.show()

    print("-" * 20, "Precision matrix (Columm Sum=1)", "-" * 20)
    plt.figure(figsize=(20, 7))
    sns.heatmap(
        B, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels
    )
    plt.title("Precision matrix (Columm Sum=1)")

    plt.xlabel("Predicted Class")
    plt.ylabel("Original Class")
    plt.show()

    # representing B in heatmap format
    print("-" * 20, "Recall matrix (Row sum=1)", "-" * 20)
    plt.figure(figsize=(20, 7))
    sns.heatmap(
        A, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels
    )
    plt.title("Recall matrix (Row sum=1)")
    plt.xlabel("Predicted Class")
    plt.ylabel("Original Class")
    plt.show()


def predict_using_random_model(x_test, y_test, x_cv, y_cv):
    # We need to generate 9 numbers and the sum of numbers should be 1
    # one solution is to genarate 9 numbers and divide each of the numbers by their sum
    # ref: https://stackoverflow.com/a/18662466/4084039
    test_data_len = x_test.shape[0]
    cv_data_len = x_cv.shape[0]

    # we create a output array that has exactly same size as the CV data
    cv_predicted_y = np.zeros((cv_data_len, 9))
    for i in range(cv_data_len):
        rand_probs = np.random.rand(1, 9)
        cv_predicted_y[i] = (rand_probs / sum(sum(rand_probs)))[0]
    print(
        "Log loss on Cross Validation Data using Random Model",
        log_loss(y_cv, cv_predicted_y, eps=1e-15),
    )

    # Test-Set error.
    # We create a output array that has exactly same as the test data
    test_predicted_y = np.zeros((test_data_len, 9))
    for i in range(test_data_len):
        rand_probs = np.random.rand(1, 9)
        test_predicted_y[i] = (rand_probs / sum(sum(rand_probs)))[0]
    print(
        "Log loss on Test Data using Random Model",
        log_loss(y_test, test_predicted_y, eps=1e-15),
    )

    predicted_y = np.argmax(test_predicted_y, axis=1)

    plot_confusion_matrix(y_test, predicted_y + 1)

    return predicted_y, y_test


def get_gene_variation_feature_dict(alpha, feature, df):
    """
    Response coding with Laplace smoothing.

    Args:
    alpha : used for laplace smoothing
    feature: ['gene', 'variation']
    df: ['x_train', 'x_test', 'x_cv']
    """
    value_count = x_train[feature].value_counts()

    # gv_dict : Gene Variation Dict, which contains the probability array for each gene/variation
    gv_dict = dict()

    # denominator will contain the number of time that particular feature occured in whole data
    for i, denominator in value_count.items():
        # vec will contain (p(yi==1/Gi) probability of gene/variation belongs to perticular class
        # vec is 9 diamensional vector
        vec = []
        for k in range(1, 10):
            cls_cnt = x_train.loc[(y_train["Class"] == k) & (x_train[feature] == i)]

            # cls_cnt.shape[0](numerator) will contain the number of time that particular feature occured in whole data
            vec.append((cls_cnt.shape[0] + alpha * 10) / (denominator + 90 * alpha))

        # we are adding the gene/variation to the dict as key and vec as value
        gv_dict[i] = vec

    return gv_dict


def get_gv_feature(alpha, feature, df):
    """
    Get Gene variation feature
    """
    gv_dict = get_gene_variation_feature_dict(alpha, feature, df)

    # value_count is similar in get_gene_variation_feature_dict
    value_count = x_train[feature].value_counts()

    # gv_feature: Gene_variation feature, it will contain the feature for each feature value in the data
    gv_feature = []
    # for every feature values in the given data frame we will check if it is there in the train data then we will add the feature to gv_fea
    # if not we will add [1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9] to gv_fea
    for index, row in df.iterrows():
        if row[feature] in dict(value_count).keys():
            gv_feature.append(gv_dict[row[feature]])
        else:
            gv_feature.append(
                [1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9]
            )

    return gv_feature


def univariate_analysis(feature, x_train):
    unique_variations = x_train[feature].value_counts()
    print("Number of Unique {0} :".format(feature), unique_variations.shape[0])
    # the top 10 variations that occured most
    print(unique_variations.head(10))

    print(
        "Ans: There are", unique_variations.shape[0], "x",
    )

    s = sum(unique_variations.values)
    h = unique_variations.values / s
    plt.plot(h, label="Histrogram of {0} :".format(feature))
    plt.xlabel("Index of a {0}".format(feature))
    plt.ylabel("Number of Occurances")
    plt.legend()
    plt.grid()
    plt.show()

    c = np.cumsum(h)
    print(c)
    plt.plot(c, label="Cumulative distribution of {0} :".format(feature))
    plt.grid()
    plt.legend()
    plt.show()


def response_coding(alpha, feature, x_train, x_test, x_cv):
    """
    Response-coding of the feature

    Args:
    alpha: Used for laplace smoothing
    feature: Feature to encode
    x_train: Train dataset X
    x_test: Test dataset X
    x_cv: Cross-validation X
    """
    # train  feature
    train_feature_responseCoding = np.array(get_gv_feature(alpha, feature, x_train))

    # test feature
    test_feature_responseCoding = np.array(get_gv_feature(alpha, feature, x_test))

    # cross validation gene feature
    cv_feature_responseCoding = np.array(get_gv_feature(alpha, feature, x_cv))

    return (
        train_feature_responseCoding,
        test_feature_responseCoding,
        cv_feature_responseCoding,
    )


def one_hot_encoding(feature, x_train, x_test, x_cv):
    """
    one-hot encoding of Gene feature.

    Args:
    feature: Feature to encode
    x_train: Train dataset X
    x_test: Test dataset X
    x_cv: Cross-validation X
    """
    vectorizer = TfidfVectorizer()

    train_feature_onehotCoding = vectorizer.fit_transform(x_train[feature])
    test_feature_onehotCoding = vectorizer.transform(x_test[feature])
    cv_feature_onehotCoding = vectorizer.transform(x_cv[feature])

    return (
        train_feature_onehotCoding,
        test_feature_onehotCoding,
        cv_feature_onehotCoding,
    )


def logistic_regression_model(feature, x_test, x_train, y_train, x_cv, y_cv):
    """
    to estimate how good a feature is, in predicting y_i
    """
    alpha = [10 ** x for x in range(-5, 1)]  # hyperparam for SGD classifier.

    (
        train_feature_onehotCoding,
        test_feature_onehotCoding,
        cv_feature_onehotCoding,
    ) = one_hot_encoding(feature, x_train, x_test, x_cv)

    cv_log_error_array = []
    for i in alpha:
        clf = SGDClassifier(alpha=i, penalty="l2", loss="log", random_state=42)
        clf.fit(train_feature_onehotCoding, y_train)
        sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
        sig_clf.fit(train_feature_onehotCoding, y_train)
        predict_y = sig_clf.predict_proba(cv_feature_onehotCoding)
        cv_log_error_array.append(
            log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15)
        )
        print(
            "For values of alpha = ",
            i,
            "The log loss is:",
            log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15),
        )

    fig, ax = plt.subplots()
    ax.plot(alpha, cv_log_error_array, c="g")
    for i, txt in enumerate(np.round(cv_log_error_array, 3)):
        ax.annotate((alpha[i], np.round(txt, 3)), (alpha[i], cv_log_error_array[i]))
    plt.grid()
    plt.title("Cross Validation Error for each alpha")
    plt.xlabel("Alpha i's")
    plt.ylabel("Error measure")
    plt.show()

    best_alpha = np.argmin(cv_log_error_array)
    clf = SGDClassifier(
        alpha=alpha[best_alpha], penalty="l2", loss="log", random_state=42
    )
    clf.fit(train_feature_onehotCoding, y_train)
    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
    sig_clf.fit(train_feature_onehotCoding, y_train)

    predict_y = sig_clf.predict_proba(train_feature_onehotCoding)
    print(
        "For values of best alpha = ",
        alpha[best_alpha],
        "The train log loss is:",
        log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15),
    )
    predict_y = sig_clf.predict_proba(cv_feature_onehotCoding)
    print(
        "For values of best alpha = ",
        alpha[best_alpha],
        "The cross validation log loss is:",
        log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15),
    )
    predict_y = sig_clf.predict_proba(test_feature_onehotCoding)
    print(
        "For values of best alpha = ",
        alpha[best_alpha],
        "The test log loss is:",
        log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15),
    )


def extract_dictionary_paddle(cls_text):
    """
    for every row in data fram consider the 'TEXT'
    split the words by space
    make a dict with those words
    Increment its count whenever we see that word
    Args:
    cls_text : Dataframe

    """
    dictionary = defaultdict(int)
    for index, row in cls_text.iterrows():
        for word in row["TEXT"].split():
            dictionary[word] += 1
    return dictionary


def get_text_responsecoding(df):
    text_feature_responseCoding = np.zeros((df.shape[0], 9))
    for i in range(0, 9):
        row_index = 0
        for index, row in df.iterrows():
            sum_prob = 0
            for word in row["TEXT"].split():
                sum_prob += math.log(
                    ((dict_list[i].get(word, 0) + 10) / (total_dict.get(word, 0) + 90))
                )
            text_feature_responseCoding[row_index][i] = math.exp(
                sum_prob / len(row["TEXT"].split())
            )
            row_index += 1
    return text_feature_responseCoding


def top_tfidf_feats(row, features, top_n=25):
    """ Get top n tfidf values in row and return them with their corresponding feature names."""
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ["feature", "tfidf"]
    return df


def top_mean_feats(Xtr, features, min_tfidf=0.1, grp_ids=None, top_n=25):
    """ Return the top n features that on average are most important amongst documents in rows
        indentified by indices in grp_ids. """
    if grp_ids:
        D = Xtr[grp_ids].toarray()
    else:
        D = Xtr.toarray()

    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)


def predict_and_plot_confusion_matrix(train_x, train_y, test_x, test_y, clf):
    clf.fit(train_x, train_y)
    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
    sig_clf.fit(train_x, train_y)
    pred_y = sig_clf.predict(test_x)

    # for calculating log_loss we willl provide the array of probabilities belongs to each class
    print("Log loss :", log_loss(test_y, sig_clf.predict_proba(test_x)))
    # calculating the number of data points that are misclassified
    print(
        "Number of mis-classified points :",
        np.count_nonzero((pred_y - test_y)) / test_y.shape[0],
    )
    plot_confusion_matrix(test_y, pred_y)


def report_log_loss(train_x, train_y, test_x, test_y, clf):
    clf.fit(train_x, train_y)
    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
    sig_clf.fit(train_x, train_y)
    sig_clf_probs = sig_clf.predict_proba(test_x)
    return log_loss(test_y, sig_clf_probs, eps=1e-15)


# this function will be used just for naive bayes
# for the given indices, we will print the name of the features
# and we will check whether the feature present in the test point text or not
def get_impfeature_names(indices, text, gene, var, no_features):
    gene_count_vec = TfidfVectorizer()
    var_count_vec = TfidfVectorizer()
    text_count_vec = TfidfVectorizer(min_df=3)

    gene_vec = gene_count_vec.fit(x_train["Gene"])
    var_vec = var_count_vec.fit(x_train["Variation"])
    text_vec = text_count_vec.fit(x_train["TEXT"])

    fea1_len = len(gene_vec.get_feature_names())
    fea2_len = len(var_count_vec.get_feature_names())

    word_present = 0
    for i, v in enumerate(indices):
        if v < fea1_len:
            word = gene_vec.get_feature_names()[v]
            yes_no = True if word == gene else False
            if yes_no:
                word_present += 1
                print(
                    i,
                    "Gene feature [{}] present in test data point [{}]".format(
                        word, yes_no
                    ),
                )
        elif v < fea1_len + fea2_len:
            word = var_vec.get_feature_names()[v - (fea1_len)]
            yes_no = True if word == var else False
            if yes_no:
                word_present += 1
                print(
                    i,
                    "variation feature [{}] present in test data point [{}]".format(
                        word, yes_no
                    ),
                )
        else:
            word = text_vec.get_feature_names()[v - (fea1_len + fea2_len)]
            yes_no = True if word in text.split() else False
            if yes_no:
                word_present += 1
                print(
                    i,
                    "Text feature [{}] present in test data point [{}]".format(
                        word, yes_no
                    ),
                )

    print(
        "Out of the top ",
        no_features,
        " features ",
        word_present,
        "are present in query point",
    )


def stack_features():
    # merging gene, variance and text features

    # building train, test and cross validation data sets
    # a = [[1, 2],
    #      [3, 4]]
    # b = [[4, 5],
    #      [6, 7]]
    # hstack(a, b) = [[1, 2, 4, 5],
    #                [ 3, 4, 6, 7]]

    train_gene_var_onehotCoding = hstack(
        (train_gene_feature_onehotCoding, train_variation_feature_onehotCoding)
    )
    test_gene_var_onehotCoding = hstack(
        (test_gene_feature_onehotCoding, test_variation_feature_onehotCoding)
    )
    cv_gene_var_onehotCoding = hstack(
        (cv_gene_feature_onehotCoding, cv_variation_feature_onehotCoding)
    )

    train_x_onehotCoding = hstack(
        (train_gene_var_onehotCoding, train_text_feature_onehotCoding)
    ).tocsr()
    train_y = np.array(list(y_train["Class"]))

    test_x_onehotCoding = hstack(
        (test_gene_var_onehotCoding, test_text_feature_onehotCoding)
    ).tocsr()
    test_y = np.array(list(y_test["Class"]))

    cv_x_onehotCoding = hstack(
        (cv_gene_var_onehotCoding, cv_text_feature_onehotCoding)
    ).tocsr()
    cv_y = np.array(list(y_cv["Class"]))

    train_gene_var_responseCoding = np.hstack(
        (train_gene_feature_responseCoding, train_variation_feature_responseCoding)
    )
    test_gene_var_responseCoding = np.hstack(
        (test_gene_feature_responseCoding, test_variation_feature_responseCoding)
    )
    cv_gene_var_responseCoding = np.hstack(
        (cv_gene_feature_responseCoding, cv_variation_feature_responseCoding)
    )

    train_x_responseCoding = np.hstack(
        (train_gene_var_responseCoding, train_text_feature_responseCoding)
    )
    test_x_responseCoding = np.hstack(
        (test_gene_var_responseCoding, test_text_feature_responseCoding)
    )
    cv_x_responseCoding = np.hstack(
        (cv_gene_var_responseCoding, cv_text_feature_responseCoding)
    )

    train = {
        "train_x_onehotCoding": train_x_onehotCoding,
        "train_y": train_y,
        "train_x_responseCoding": train_x_responseCoding,
    }
    test = {
        "test_x_onehotCoding": test_x_onehotCoding,
        "test_y": test_y,
        "test_x_responseCoding": test_x_responseCoding,
    }
    cv = {
        "cv_x_onehotCoding": cv_x_onehotCoding,
        "cv_y": cv_y,
        "cv_x_responseCoding": cv_x_responseCoding,
    }

    return train, test, cv


def naive_bayes_model(train, test, cv):
    """
    Hyper parameter Tuning
    """
    alpha = [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100, 1000]

    cv_log_error_array = []

    for i in alpha:
        print("for alpha =", i)
        clf = MultinomialNB(alpha=i)
        clf.fit(train["train_x_onehotCoding"], train["train_y"])
        sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
        sig_clf.fit(train["train_x_onehotCoding"], train["train_y"])
        sig_clf_probs = sig_clf.predict_proba(cv["cv_x_onehotCoding"])
        cv_log_error_array.append(
            log_loss(cv["cv_y"], sig_clf_probs, labels=clf.classes_, eps=1e-15)
        )
        # to avoid rounding error while multiplying probabilites we use log-probability estimates
        print("Log Loss :", log_loss(cv["cv_y"], sig_clf_probs))

    fig, ax = plt.subplots()
    ax.plot(np.log10(alpha), cv_log_error_array, c="g")
    for i, txt in enumerate(np.round(cv_log_error_array, 3)):
        ax.annotate((alpha[i], str(txt)), (np.log10(alpha[i]), cv_log_error_array[i]))
    plt.grid()
    plt.xticks(np.log10(alpha))
    plt.title("Cross Validation Error for each alpha")
    plt.xlabel("Alpha i's")
    plt.ylabel("Error measure")
    plt.show()

    best_alpha = np.argmin(cv_log_error_array)
    clf = MultinomialNB(alpha=alpha[best_alpha])
    clf.fit(train["train_x_onehotCoding"], train["train_y"])
    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
    sig_clf.fit(train["train_x_onehotCoding"], train["train_y"])

    predict_y = sig_clf.predict_proba(train["train_x_onehotCoding"])
    print(
        "For values of best alpha = ",
        alpha[best_alpha],
        "The train log loss is:",
        log_loss(train["train_y"], predict_y, labels=clf.classes_, eps=1e-15),
    )

    predict_y = sig_clf.predict_proba(cv["cv_x_onehotCoding"])
    print(
        "For values of best alpha = ",
        alpha[best_alpha],
        "The cross validation log loss is:",
        log_loss(cv["cv_y"], predict_y, labels=clf.classes_, eps=1e-15),
    )

    predict_y = sig_clf.predict_proba(test["test_x_onehotCoding"])
    print(
        "For values of best alpha = ",
        alpha[best_alpha],
        "The test log loss is:",
        log_loss(test["test_y"], predict_y, labels=clf.classes_, eps=1e-15),
    )

    return alpha, best_alpha, predict_y


def test_model(train, test, cv):
    clf = MultinomialNB(alpha=alpha[best_alpha])
    clf.fit(train["train_x_onehotCoding"], train["train_y"])
    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
    sig_clf.fit(train["train_x_onehotCoding"], train["train_y"])
    sig_clf_probs = sig_clf.predict_proba(cv["cv_x_onehotCoding"])

    # to avoid rounding error while multiplying probabilites we use log-probability estimates
    print("Log Loss :", log_loss(cv["cv_y"], sig_clf_probs))
    print(
        "Number of missclassified point :",
        np.count_nonzero((sig_clf.predict(cv["cv_x_onehotCoding"]) - cv["cv_y"]))
        / cv["cv_y"].shape[0],
    )
    plot_confusion_matrix(
        cv["cv_y"], sig_clf.predict(cv["cv_x_onehotCoding"].toarray())
    )

    # Feature Importance and Correctly Classified points
    test_point_index = 2
    no_feature = 100
    predicted_cls = sig_clf.predict(test["test_x_onehotCoding"][test_point_index])
    print("Predicted Class :", predicted_cls[0])
    print(
        "Predicted Class Probabilities:",
        np.round(
            sig_clf.predict_proba(test["test_x_onehotCoding"][test_point_index]), 4
        ),
    )
    print("Actual Class :", test["test_y"][test_point_index])
    indices = np.argsort(-clf.coef_)[predicted_cls - 1][:, :no_feature]
    print("-" * 50)
    get_impfeature_names(
        indices[0],
        x_test["TEXT"].iloc[test_point_index],
        x_test["Gene"].iloc[test_point_index],
        x_test["Variation"].iloc[test_point_index],
        no_feature,
    )

    # Feature Importance and Incorrectly Classified points
    test_point_index = 100
    no_feature = 100
    predicted_cls = sig_clf.predict(test["test_x_onehotCoding"][test_point_index])
    print("Predicted Class :", predicted_cls[0])
    print(
        "Predicted Class Probabilities:",
        np.round(
            sig_clf.predict_proba(test["test_x_onehotCoding"][test_point_index]), 4
        ),
    )
    print("Actual Class :", test["test_y"][test_point_index])
    indices = np.argsort(-clf.coef_)[predicted_cls - 1][:, :no_feature]
    print("-" * 50)
    get_impfeature_names(
        indices[0],
        x_test["TEXT"].iloc[test_point_index],
        x_test["Gene"].iloc[test_point_index],
        x_test["Variation"].iloc[test_point_index],
        no_feature,
    )


def train_text_feature():
    """
    to estimate how good a text feature is, in predicting y_i
    Train a Logistic regression+Calibration model using text features which are on-hot encoded
    """
    alpha = [10 ** x for x in range(-5, 1)]

    # read more about SGDClassifier() at http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
    # ------------------------------
    # default parameters
    # SGDClassifier(loss=’hinge’, penalty=’l2’, alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=None, tol=None,
    # shuffle=True, verbose=0, epsilon=0.1, n_jobs=1, random_state=None, learning_rate=’optimal’, eta0=0.0, power_t=0.5,
    # class_weight=None, warm_start=False, average=False, n_iter=None)

    # some of methods
    # fit(X, y[, coef_init, intercept_init, …])	Fit linear model with Stochastic Gradient Descent.
    # predict(X)	Predict class labels for samples in X.

    # -------------------------------
    # video link:
    # ------------------------------

    cv_log_error_array = []
    for i in alpha:
        clf = SGDClassifier(alpha=i, penalty="l2", loss="log", random_state=42)
        clf.fit(train_text_feature_onehotCoding, y_train)

        sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
        sig_clf.fit(train_text_feature_onehotCoding, y_train)
        predict_y = sig_clf.predict_proba(cv_text_feature_onehotCoding)
        cv_log_error_array.append(
            log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15)
        )
        print(
            "For values of alpha = ",
            i,
            "The log loss is:",
            log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15),
        )

    fig, ax = plt.subplots()
    ax.plot(alpha, cv_log_error_array, c="g")
    for i, txt in enumerate(np.round(cv_log_error_array, 3)):
        ax.annotate((alpha[i], np.round(txt, 3)), (alpha[i], cv_log_error_array[i]))
    plt.grid()
    plt.title("Cross Validation Error for each alpha")
    plt.xlabel("Alpha i's")
    plt.ylabel("Error measure")
    plt.show()

    best_alpha = np.argmin(cv_log_error_array)
    clf = SGDClassifier(
        alpha=alpha[best_alpha], penalty="l2", loss="log", random_state=42
    )
    clf.fit(train_text_feature_onehotCoding, y_train)
    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
    sig_clf.fit(train_text_feature_onehotCoding, y_train)

    predict_y = sig_clf.predict_proba(train_text_feature_onehotCoding)
    print(
        "For values of best alpha = ",
        alpha[best_alpha],
        "The train log loss is:",
        log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15),
    )
    predict_y = sig_clf.predict_proba(cv_text_feature_onehotCoding)
    print(
        "For values of best alpha = ",
        alpha[best_alpha],
        "The cross validation log loss is:",
        log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15),
    )
    predict_y = sig_clf.predict_proba(test_text_feature_onehotCoding)
    print(
        "For values of best alpha = ",
        alpha[best_alpha],
        "The test log loss is:",
        log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15),
    )


def get_intersec_text(df, train_text_features):
    df_text_vec = TfidfVectorizer(min_df=3)
    df_text_fea = df_text_vec.fit_transform(df["TEXT"])

    df_text_features = top_mean_feats(
        df_text_fea, df_text_vec.get_feature_names(), top_n=1000
    )["feature"].tolist()

    df_text_fea_counts = df_text_fea.sum(axis=0).A1
    df_text_fea_dict = dict(zip(list(df_text_features), df_text_fea_counts))
    len1 = len(set(df_text_features))
    len2 = len(set(train_text_features) & set(df_text_features))
    return len1, len2


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

    # split the data into test and train by maintaining same distribution of output variable 'y_true' [stratify=y_true]
    x_train, x_test, y_train, y_test = train_test_split(
        x_true, y_true, stratify=y_true, test_size=0.2
    )
    # split the train data into train and cross validation by maintaining same distribution of output variable 'y_train' [stratify=y_train]
    x_train, x_cv, y_train, y_cv = train_test_split(
        x_train, y_train, stratify=y_train, test_size=0.2
    )

    (
        train_gene_feature_onehotCoding,
        test_gene_feature_onehotCoding,
        cv_gene_feature_onehotCoding,
    ) = one_hot_encoding("Gene", x_train, x_test, x_cv)

    (
        train_gene_feature_responseCoding,
        test_gene_feature_responseCoding,
        cv_gene_feature_responseCoding,
    ) = response_coding(1, "Gene", x_train, x_test, x_cv)

    (
        train_variation_feature_onehotCoding,
        test_variation_feature_onehotCoding,
        cv_variation_feature_onehotCoding,
    ) = one_hot_encoding("Variation", x_train, x_test, x_cv)

    (
        train_variation_feature_responseCoding,
        test_variation_feature_responseCoding,
        cv_variation_feature_responseCoding,
    ) = response_coding(1, "Variation", x_train, x_test, x_cv)

    dict_list = []
    # dict_list =[] contains 9 dictoinaries each corresponds to a class
    for i in range(1, 10):
        cls_text = x_train[y_train["Class"] == i]
        # build a word dict based on the words in that class
        dict_list.append(extract_dictionary_paddle(cls_text))

    # dict_list[i] is build on i'th  class text data
    # total_dict is buid on whole training text data
    total_dict = extract_dictionary_paddle(x_train)

    # building a TfidfVectorizer with all the words that occured minimum 3 times in train data
    text_vectorizer = TfidfVectorizer(min_df=3)

    train_text_feature_onehotCoding = text_vectorizer.fit_transform(x_train["TEXT"])
    train_text_feature_onehotCoding = normalize(train_text_feature_onehotCoding, axis=0)

    test_text_feature_onehotCoding = text_vectorizer.transform(x_test["TEXT"])
    test_text_feature_onehotCoding = normalize(test_text_feature_onehotCoding, axis=0)

    cv_text_feature_onehotCoding = text_vectorizer.transform(x_cv["TEXT"])
    cv_text_feature_onehotCoding = normalize(cv_text_feature_onehotCoding, axis=0)

    # response coding of text features
    train_text_feature_responseCoding = get_text_responsecoding(x_train)
    test_text_feature_responseCoding = get_text_responsecoding(x_test)
    cv_text_feature_responseCoding = get_text_responsecoding(x_cv)

    # we convert each row values such that they sum to 1
    train_text_feature_responseCoding = (
        train_text_feature_responseCoding.T
        / train_text_feature_responseCoding.sum(axis=1)
    ).T
    test_text_feature_responseCoding = (
        test_text_feature_responseCoding.T
        / test_text_feature_responseCoding.sum(axis=1)
    ).T
    cv_text_feature_responseCoding = (
        cv_text_feature_responseCoding.T / cv_text_feature_responseCoding.sum(axis=1)
    ).T

    train, test, cv = stack_features()

    alpha, best_alpha, predict_y = naive_bayes_model(train, test, cv)

    test_model(train, test, cv)
