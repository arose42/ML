from sklearn import tree, ensemble
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
import numpy as np


def decisionTree():
    twitter_dataset = pd.read_csv('Sentiment.csv')

    column_names = ['candidate', 'candidate_confidence', 'relevant_yn', 'relevant_yn_confidence',
                    'sentiment_confidence', 'subject_matter', 'subject_matter_confidence',
                    'retweet_count', 'sentiment']

    column_names_without_label = ['candidate', 'candidate_confidence', 'relevant_yn', 'relevant_yn_confidence',
                                  'sentiment_confidence', 'subject_matter', 'subject_matter_confidence',
                                  'retweet_count']

    label = ['sentiment']
    twitter_dataset = twitter_dataset[column_names]
    twitter_dataset = twitter_dataset.dropna()
    label_encoder = preprocessing.LabelEncoder()

    twitter_dataset['candidate'] = label_encoder.fit_transform(twitter_dataset['candidate'])
    twitter_dataset['candidate'].unique()

    twitter_dataset['subject_matter'] = label_encoder.fit_transform(twitter_dataset['subject_matter'])
    twitter_dataset['subject_matter'].unique()

    twitter_dataset['sentiment'] = label_encoder.fit_transform(twitter_dataset['sentiment'])
    twitter_dataset['sentiment'].unique()

    twitter_dataset['relevant_yn'] = label_encoder.fit_transform(twitter_dataset['relevant_yn'])
    twitter_dataset['relevant_yn'].unique()

    twitter_dataset['candidate_confidence'] = twitter_dataset.candidate_confidence.astype(int)
    twitter_dataset['relevant_yn_confidence'] = twitter_dataset.relevant_yn_confidence.astype(int)
    twitter_dataset['sentiment_confidence'] = twitter_dataset.sentiment_confidence.astype(int)
    twitter_dataset['subject_matter_confidence'] = twitter_dataset.subject_matter_confidence.astype(int)

    twitterX = twitter_dataset[column_names_without_label].copy()
    twitterY = twitter_dataset[label].copy()

    twitter_trainingX, twitter_testingX, twitter_trainingY, twitter_testingY = train_test_split(twitterX, twitterY,
                                                                                                test_size=0.3,
                                                                                                random_state=0,
                                                                                                stratify=twitterY)
    #model = tree.DecisionTreeClassifier(max_depth=3, criterion='entropy')
    #https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html
    model = tree.DecisionTreeClassifier(random_state=0)
    path = model.cost_complexity_pruning_path(twitter_trainingX, twitter_trainingY)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    fig, ax = plt.subplots()
    ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
    ax.set_xlabel("effective alpha")
    ax.set_ylabel("total impurity of leaves")
    ax.set_title("Total Impurity vs effective alpha for training set")

    clfs = []

    clf = tree.DecisionTreeClassifier(random_state=0)
    clf.fit(twitter_trainingX, twitter_trainingY)
    y_pred_train = clf.predict(twitter_trainingX)
    train_scores = accuracy_score(y_pred_train, twitter_trainingY)
    y_pred = clf.predict(twitter_testingX)
    test_scores = accuracy_score(y_pred, twitter_testingY)

    train_sizes = np.linspace(.1, 1.0, 5)

    #======================== CITATION BELOW ==============================================#
    #https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html
    cv = None
    n_jobs = None
    ylim = None

    train_sizes = np.linspace(.1, 1.0, 5)

    for ccp_alpha in ccp_alphas:
        clf = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
        clf.fit(twitter_trainingX, twitter_trainingY)
        clfs.append(clf)

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(clf, twitter_trainingX, twitter_trainingY, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    _, axes = plt.subplots(1, 3, figsize=(20, 5))

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    ylim = None
    axes[0].set_title('Control Curve')
    if ylim is not None:
        axes[0].set_ylim(ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

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

    params = {'n_estimators': 500,
              'max_depth': 4,
              'min_samples_split': 5,
              'learning_rate': 0.0001,
              'loss': 'ls'}

    #======================== CITATION ABOVE ==============================================#

    #======================== CITATION BELOW ==============================================#
    #https: // scikit - learn.org / stable / auto_examples / ensemble / plot_gradient_boosting_regression.html
    reg = ensemble.GradientBoostingRegressor(**params)
    reg.fit(twitter_trainingX, twitter_trainingY)

    mse = mean_squared_error(twitter_trainingY, reg.predict(twitter_trainingX))
    print("The mean squared error (MSE) on test set: {:.4f}".format(mse))

    test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
    for i, y_pred in enumerate(reg.staged_predict(twitter_trainingX)):
        test_score[i] = reg.loss_(twitter_trainingY.to_numpy(), y_pred)

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
    #======================== CITATION ABOVE ==============================================#

    # ======================== CITATION BELOW ==============================================#
    # https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html
    for ccp_alpha in ccp_alphas:
        clf = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
        clf.fit(twitter_trainingX, twitter_trainingY)
        clfs.append(clf)
    print("Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
        clfs[-1].tree_.node_count, ccp_alphas[-1]))

    clfs = clfs[:-1]
    ccp_alphas = ccp_alphas[:-1]

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

    train_scores = [clf.score(twitter_trainingX, twitter_trainingY) for clf in clfs]
    test_scores = [clf.score(twitter_testingX, twitter_testingY) for clf in clfs]

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
    #======================== CITATION ABOVE ==============================================#

    #model.fit(twitter_trainingX, twitter_trainingY)
    prediction = model.predict(twitter_testingX)

    # ======================== CITATION BELOW ==============================================#
    #https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
    print('Accuracy pre-boosting:', accuracy_score(twitter_testingY, prediction) * 100, '%')

    tree.export_graphviz(model, out_file='tree.dot', feature_names=twitter_trainingX.columns)

    AdaBoost = AdaBoostClassifier(n_estimators=400, learning_rate=1, algorithm='SAMME')
    AdaBoost.fit(twitter_trainingX, twitter_trainingY)
    prediction = AdaBoost.score(twitter_trainingX, twitter_trainingY)
    print('Accuracy post-boosting: ', prediction * 100, '%')
    #======================== CITATION ABOVE ==============================================#


def main():
    decisionTree()


if __name__ == "__main__":
    main()
