import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV

#CODE CITATION: https://stackabuse.com/implementing-svm-and-kernel-svm-with-pythons-scikit-learn/
def SVM():
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
                                                                                                test_size=0.8,
                                                                                                random_state=0,
                                                                                                stratify=twitterY)
    preprocessing.normalize(twitter_trainingX)

    #linear kernel
    svclassifier = SVC(kernel='linear', probability=False)
    svclassifier.fit(twitter_trainingX, twitter_trainingY.values.ravel())

    prediction = svclassifier.predict(twitter_testingX)
    accuracy_score(prediction, twitter_testingY)
    print(confusion_matrix(twitter_testingY, prediction))
    print(classification_report(twitter_testingY, prediction))

    #gaussian kernel
    svclassifier = SVC(kernel='rbf', probability=False)
    svclassifier.fit(twitter_trainingX, twitter_trainingY.values.ravel())
    prediction = svclassifier.predict(twitter_testingX)

    print(confusion_matrix(twitter_testingY, prediction))
    print(classification_report(twitter_testingY, prediction))

    # ======================== CITATION BELOW ==============================================#
    #https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    scores = ['precision', 'recall']

    clf = GridSearchCV(
        SVC(), tuned_parameters, scoring='%s_macro' % scores[0]
    )
    grid_results = clf.fit(twitter_trainingX, twitter_trainingY)


    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(
            SVC(), tuned_parameters, scoring='%s_macro' % score
        )
        grid_results = clf.fit(twitter_trainingX, twitter_trainingY)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = twitter_testingY, clf.predict(twitter_testingX)
        print(classification_report(y_true, y_pred))
        print()
        # ======================== CITATION ABOVE ==============================================#


def main():
    SVM()


if __name__ == "__main__":
    main()
