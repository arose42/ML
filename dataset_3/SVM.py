import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn import preprocessing
from sklearn.datasets import load_digits
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV

#CODE CITATION: https://stackabuse.com/implementing-svm-and-kernel-svm-with-pythons-scikit-learn/
def SVM():
    digits = load_digits()
    data_features = digits.data[:, 0:-1]
    label = digits.data[:, -1]
    ylim = None

    digits_trainingX, digits_testingX, digits_trainingY, digits_testingY = train_test_split \
        (data_features, label, test_size=0.3, random_state=0,
         stratify=label)

    feature_columns = pd.DataFrame(data=digits_trainingX).columns

    # fit the model
    digits_trainingX_2 = digits_trainingX[:, [5,6]]
    h = .02

    # ======================== CITATION BELOW ==============================================#
    #https://scikit-learn.org/0.18/auto_examples/svm/plot_iris.html

    # we create an instance of SVM and fit out data. We do not scale our
    # data since we want to plot the support vectors
    C = 1.0  # SVM regularization parameter
    svc = SVC(kernel='linear', C=C).fit(digits_trainingX_2, digits_trainingY)
    rbf_svc = SVC(kernel='rbf', gamma=0.7, C=C).fit(digits_trainingX_2, digits_trainingY)
    poly_svc = SVC(kernel='poly', degree=3, C=C).fit(digits_trainingX_2, digits_trainingY)
    lin_svc = LinearSVC(C=C).fit(digits_trainingX_2, digits_trainingY)

    poly_model =  SVC(kernel='poly', degree=3, C=C)
    poly_model.fit(digits_trainingX, digits_trainingY)
    y_pred = poly_model.predict(digits_testingX)

    oly_model = SVC(kernel='rbf', degree=3, C=C)
    poly_model.fit(digits_trainingX, digits_trainingY)
    y_pred2 = poly_model.predict(digits_testingX)

    accuracy_score(y_pred, digits_testingY)

    # create a mesh to plot in
    x_min, x_max = digits_trainingX_2[:, 0].min() - 1, digits_trainingX_2[:, 0].max() + 1
    y_min, y_max = digits_trainingX_2[:, 1].min() - 1, digits_trainingX_2[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # title for the plots
    titles = ['SVC with linear kernel',
              'LinearSVC (linear kernel)',
              'SVC with RBF kernel',
              'SVC with polynomial (degree 3) kernel']

    for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        plt.subplot(2, 2, i + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=cm.get_cmap("Spectral"), alpha=0.8)

        # Plot also the training points
        plt.scatter(digits_trainingX_2[:, 0], digits_trainingX_2[:, 1], c=digits_trainingY, cmap=cm.get_cmap("Spectral"))
        plt.xlabel('Sepal length')
        plt.ylabel('Sepal width')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title(titles[i])

    plt.show()
    # ======================== CITATION ABOVE==============================================#

    #accuracy_score(prediction, digits_testingY)

    #print(confusion_matrix(digits_testingY, prediction))
    #print(classification_report(digits_testingY, prediction))

    #gaussian kernel
    svclassifier = SVC(kernel='rbf', probability=False)
    svclassifier.fit(digits_trainingX, digits_trainingY.values.ravel())
    prediction = svclassifier.predict(digits_testingX)

    print(confusion_matrix(digits_testingY, prediction))
    print(classification_report(digits_testingY, prediction))

    #polynomial kernel
    #svclassifier = SVC(kernel='poly')
    #svclassifier.fit(twitter_trainingX, twitter_trainingY)
    #prediction = svclassifier.predict(twitter_testingX)

    #print(confusion_matrix(twitter_testingY, prediction))
    #print(classification_report(twitter_testingY, prediction))

    # sigmoid kernel
    #svclassifier = SVC(kernel='sigmoid')
    #svclassifier.fit(twitter_trainingX, twitter_trainingY)
    #prediction = svclassifier.predict(twitter_testingX)

    #print(confusion_matrix(twitter_testingY, prediction))
    #print(classification_report(twitter_testingY, prediction))

    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    scores = ['precision', 'recall']

    # ======================== CITATION BELOW ==============================================#
    #https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(
            SVC(), tuned_parameters, scoring='%s_macro' % score
        )
        clf.fit(digits_trainingX, digits_trainingY)

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
        y_true, y_pred = digits_testingY, clf.predict(digits_testingX)
        print(classification_report(y_true, y_pred))
        print()
    # ======================== CITATION ABOVE ==============================================#



def main():
    SVM()


if __name__ == "__main__":
    main()
