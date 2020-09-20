from sklearn import tree, ensemble
import pandas as pd
from sklearn import preprocessing
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
import numpy as np


def decisionTree():
    digits = load_digits()
    data_features = digits.data[:, 0:-1]
    label = digits.data[:, -1]
    ylim = None

    digits_trainingX, digits_testingX, digits_trainingY, digits_testingY = train_test_split \
        (data_features, label, test_size=0.3, random_state=0,
         stratify=label)

    feature_columns = pd.DataFrame(data=digits_trainingX).columns

    kb = SelectKBest(score_func=f_regression, k=45)
    kb.fit(digits_trainingX, digits_trainingY)
    mask = kb.get_support()
    chosen_features = []

    for bool, feature in zip(mask, feature_columns):
        if bool:
            chosen_features.append(feature)

    df = pd.DataFrame(data=digits_trainingX)
    df = df[chosen_features]
    digits_trainingX = df.to_numpy()

    df2 = pd.DataFrame(data=digits_testingX)
    df2 = df2[chosen_features]
    digits_testingX = df2.to_numpy()

    model = tree.DecisionTreeClassifier(random_state=0)
    path = model.cost_complexity_pruning_path(digits_trainingX, digits_trainingY)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    #code source: https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html
    params = {'n_estimators': 500,
              'max_depth': 4,
              'min_samples_split': 5,
              'learning_rate': 0.01,
              'loss': 'ls'}

    # ======================== CITATION BELOW ==============================================#
    # https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html

    fig, ax = plt.subplots()
    ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
    ax.set_xlabel("effective alpha")
    ax.set_ylabel("total impurity of leaves")
    ax.set_title("Total Impurity vs effective alpha for training set")

    clfs = []
    for ccp_alpha in ccp_alphas:
        clf = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
        clf.fit(digits_trainingX, digits_trainingY)
        clfs.append(clf)
    print("Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
        clfs[-1].tree_.node_count, ccp_alphas[-1]))

    clfs = clfs[:-1]
    ccp_alphas = ccp_alphas[:-1]

    # ======================== CITATION ABOVE ==============================================#

    # ======================== CITATION BELOW ==============================================#
    # https: // scikit - learn.org / stable / auto_examples / ensemble / plot_gradient_boosting_regression.html

    reg = ensemble.GradientBoostingRegressor(**params)
    reg.fit(digits_trainingX, digits_trainingY)

    mse = mean_squared_error(digits_trainingY, reg.predict(digits_trainingX))
    print("The mean squared error (MSE) on test set: {:.4f}".format(mse))

    test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
    for i, y_pred in enumerate(reg.staged_predict(digits_trainingX)):
        test_score[i] = reg.loss_(digits_trainingY, y_pred)

    fig = plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1)
    plt.title('Deviance')
    plt.plot(np.arange(params['n_estimators']) + 1, reg.train_score_, 'b-',
             label='Training Set Deviance')
    plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
             label='Test Set Deviance')
    plt.legend(loc='upper right')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Deviance')
    fig.tight_layout()
    plt.show()

    # ======================== CITATION ABOVE ==============================================#

    # ======================== CITATION BELOW ==============================================#
    # https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html
    node_counts = [clf.tree_.node_count for clf in clfs]
    depth = [clf.tree_.max_depth for clf in clfs]
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(ccp_alphas, node_counts, marker='o', drawstyle="steps-post")
    ax[0].set_xlabel("alpha")
    ax[0].set_ylabel("number of nodes")
    ax[0].set_title("Number of nodes vs alpha")
    ax[1].plot(ccp_alphas, depth, marker='o', drawstyle="steps-post")
    ax[1].set_xlabel("alpha")
    ax[1].set_ylabel("depth of tree")
    ax[1].set_title("Depth vs alpha")
    fig.tight_layout()

    train_scores = [clf.score(digits_trainingX, digits_trainingY) for clf in clfs]
    test_scores = [clf.score(digits_testingX, digits_testingY) for clf in clfs]

    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs alpha for training and testing sets")
    ax.plot(ccp_alphas, train_scores, marker='o', label="train",
            drawstyle="steps-post")
    ax.plot(ccp_alphas, test_scores, marker='o', label="test",
            drawstyle="steps-post")
    ax.legend()
    plt.show()
    # ======================== CITATION ABOVE ==============================================#


    #print('Accuracy pre-boosting:', accuracy_score(twitter_testingY, prediction) * 100, '%')

    #tree.export_graphviz(model, out_file='tree.dot', feature_names=twitter_trainingX.columns)

    AdaBoost = AdaBoostClassifier(n_estimators=400, learning_rate=1, algorithm='SAMME')
    #AdaBoost.fit(twitter_trainingX, twitter_trainingY)
    #prediction = AdaBoost.score(twitter_trainingX, twitter_trainingY)
    #print('Accuracy post-boosting: ', prediction * 100, '%')


def main():
    decisionTree()


if __name__ == "__main__":
    main()
