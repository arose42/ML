from sklearn import preprocessing, ensemble
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, KFold


#CITATION: https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/

def get_integer_mapping(le):
    '''
    Return a dict mapping labels to their integer values
    from an SKlearn LabelEncoder
    le = a fitted SKlearn LabelEncoder
    '''
    res = {}
    for cl in le.classes_:
        res.update({cl:le.transform([cl])[0]})

    return res

def KNN():
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
    integerMapping = get_integer_mapping(label_encoder)

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
                                                                                                test_size=0.3, random_state=0,
                                                                                                stratify=twitterY)

    #======================== CITATION BELOW ==============================================#
    #https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html
    params = {'n_estimators': 500,
              'max_depth': 4,
              'min_samples_split': 5,
              'learning_rate': 0.01,
              'loss': 'ls'}

    reg = ensemble.GradientBoostingRegressor(**params)
    reg.fit(twitter_trainingX, twitter_trainingY)

    mse = mean_squared_error(twitter_trainingY, reg.predict(twitter_trainingX))
    print("The mean squared error (MSE) on test set: {:.4f}".format(mse))

    test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
    twitter_trainingX = twitter_trainingX.to_numpy()
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

    algorithm = ['ball_tree', 'kd_tree']
    weights = ['uniform', 'distance']
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(twitter_trainingX, twitter_trainingY)
    y_pred = classifier.predict(twitter_testingX)
    seed=52
    
    accuracy_score(y_pred, twitter_testingY)

    param_grid = dict(algorithm=algorithm, weights=weights)

    grid = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=KFold(random_state=seed), verbose=10,
                        scoring='accuracy')
    grid_results = grid.fit(twitter_trainingX, twitter_trainingY)


    classifier.fit(twitter_trainingX, twitter_trainingY)
    prediction = classifier.predict(twitter_testingX)



    #evaluating algorithm
    #negative = 0
    #neutral = 1
    #positive = 2

    print(confusion_matrix(twitter_testingY, prediction))
    print(classification_report(twitter_testingY, prediction))

    #======================== CITATION BELOW ==============================================#
    #https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/
    error = []

    for i in range(1, 40):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(twitter_trainingX, twitter_trainingY)
        pred_i = knn.predict(twitter_testingX)
        error.append(np.mean(pred_i != twitter_testingY.T))

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)
    plt.title('Error Rate K Value')
    plt.xlabel('K Value')
    plt.ylabel('Mean Error')

    # ======================== CITATION ABOVE ==============================================#

    AdaBoost = AdaBoostClassifier(n_estimators=400, learning_rate=1, algorithm='SAMME')
    AdaBoost.fit(twitter_trainingX, twitter_trainingY)
    prediction = AdaBoost.score(twitter_trainingX, twitter_trainingY)
    print('Accuracy post-boosting: ', prediction * 100, '%')

def main():
    KNN()


if __name__ == "__main__":
    main()
