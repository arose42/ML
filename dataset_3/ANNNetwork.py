import os
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import train_test_split as sp, train_test_split
from sklearn.base import BaseEstimator
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import f1_score, make_scorer, accuracy_score
import numpy as np
from sklearn.model_selection import learning_curve


# CODING CITATION: https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough
from sklearn.utils import compute_sample_weight
from tensorflow_estimator.python.estimator.estimator import Estimator

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
digits = load_digits


def ANN():
    digits = load_digits()
    data_features = digits.data[:, 0:-1]
    label = digits.data[:, -1]
    ylim = None

    digits_trainingX, digits_testingX, digits_trainingY, digits_testingY = train_test_split\
        (data_features, label, test_size=0.3, random_state=0,
                     stratify=label)

    feature_columns = pd.DataFrame(data=digits_trainingX).columns

    #clf = MLPClassifier(alpha=1e-05, hidden_layer_sizes=(63,), random_state=1,
                        #solver='adam')
    #clf.fit(digits_trainingX, digits_trainingY)
    #y_pred = clf.predict(digits_testingX)

    kb = SelectKBest(score_func=f_regression, k=45)
    kb.fit(digits_trainingX, digits_trainingY)
    mask = kb.get_support()
    chosen_features = []

    for bool, feature in zip(mask, feature_columns):
        if bool:
            chosen_features.append(feature)

    #indices = np.argsort(kb.scores_)[::-1]
    #selected_features = []
    #for i in range(63):
        #selected_features.append(pd.DataFrame(data=digits_trainingX).columns[indices[i]])

    df = pd.DataFrame(data=digits_trainingX)
    df = df[chosen_features]
    digits_trainingX = df.to_numpy()

    df2 = pd.DataFrame(data=digits_testingX)
    df2 = df2[chosen_features]
    digits_testingX = df2.to_numpy()

    #digits_trainingX = digits_trainingX[chosen_features]
    clf = MLPClassifier(alpha=1e-05, hidden_layer_sizes=(45,), random_state=1,
                        solver='lbfgs')
    clf.fit(digits_trainingX, digits_trainingY)
    y_pred = clf.predict(digits_testingX)



    train_sizes = np.linspace(.1, 1.0, 5)

    # ======================== CITATION BELOW ==============================================#
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html    cv = None
    n_jobs = None
    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(clf, digits_trainingX, digits_trainingY, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title('Control Curve')
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    # ======================== CITATION ABOVE ==============================================#

    optimizers = ['lbfgs', 'sgd', 'adam']
    max_iters = [100, 200, 500]
    batch_size = [5, 10, 100]
    seed = 52

    #for i in range(63):
        #selected_features.append(pd.DataFrame(data=digits_trainingX).columns[indices[i]])

    #plt.figure()
    #plt.bar(selected_features, kb.scores_[indices[range(63)]], color='r', align='center')
    #plt.xticks(rotation=45)
    #plt.xlabel('features')
    #plt.ylabel('score')

    param_grid = dict(solver=optimizers, max_iter=max_iters, batch_size=batch_size)

    grid = GridSearchCV(estimator=clf, param_grid=param_grid, cv=KFold(random_state=seed), verbose=10,
                        scoring='accuracy')
    grid_results = grid.fit(digits_trainingX, digits_trainingY)


def main():
    ANN()


if __name__ == "__main__":
    main()
