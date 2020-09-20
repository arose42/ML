from sklearn import preprocessing
from sklearn.datasets import load_digits
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    digits = load_digits()
    data_features = digits.data[:, 0:-1]
    label = digits.data[:, -1]
    ylim = None

    digits_trainingX, digits_testingX, digits_trainingY, digits_testingY = train_test_split \
        (data_features, label, test_size=0.3, random_state=0,
         stratify=label)

    feature_columns = pd.DataFrame(data=digits_trainingX).columns

    # ======================== CITATION BELOW ==============================================#
    #https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html
    kb = SelectKBest(score_func=f_regression, k=45)
    kb.fit(digits_trainingX, digits_trainingY)
    mask = kb.get_support()
    chosen_features = []

    for bool, feature in zip(mask, feature_columns):
        if bool:
            chosen_features.append(feature)
    # ======================== CITATION ABOVE ==============================================#


    df = pd.DataFrame(data=digits_trainingX)
    df = df[chosen_features]
    digits_trainingX = df.to_numpy()

    df2 = pd.DataFrame(data=digits_testingX)
    df2 = df2[chosen_features]
    digits_testingX = df2.to_numpy()

    classifier = KNeighborsClassifier(n_neighbors=5, weights='distance')
    classifier.fit(digits_trainingX, digits_trainingY)
    prediction = classifier.predict(digits_testingX)

    accuracy_score(prediction, digits_testingY)

    algorithm = ['ball_tree', 'kd_tree']
    weights = ['uniform', 'distance']
    seed = 52

    param_grid = dict(algorithm=algorithm, weights=weights)

    grid = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=KFold(random_state=seed), verbose=10,
                        scoring='accuracy')
    grid_results = grid.fit(digits_trainingX, digits_trainingY)


    #evaluating algorithm
    #negative = 0
    #neutral = 1
    #positive = 2

    #print(confusion_matrix(twitter_testingY, prediction))
    #print(classification_report(twitter_testingY, prediction))

    # ======================== CITATION BELOW ==============================================#
    # https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/
    error = []

    for i in range(1, 40):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(digits_trainingX, digits_trainingY)
        pred_i = knn.predict(digits_testingX)
        error.append(np.mean(pred_i != digits_testingY.T))

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)
    plt.title('Error Rate K Value')
    plt.xlabel('K Value')
    plt.ylabel('Mean Error')
    # ======================== CITATION ABOVE ==============================================#


    AdaBoost = AdaBoostClassifier(n_estimators=400, learning_rate=1, algorithm='SAMME')
    AdaBoost.fit(digits_trainingX, digits_trainingY)
    prediction = AdaBoost.score(digits_trainingX, digits_trainingY)
    print('Accuracy post-boosting: ', prediction * 100, '%')

def main():
    KNN()


if __name__ == "__main__":
    main()
