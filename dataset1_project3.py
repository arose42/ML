import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score, accuracy_score, mean_squared_error
from sklearn.decomposition import PCA, IncrementalPCA, FastICA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
from sklearn.random_projection import GaussianRandomProjection
from yellowbrick.cluster import InterclusterDistance
from yellowbrick.cluster import SilhouetteVisualizer
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics.cluster import v_measure_score
import pandas as pd
import seaborn as sns
from kneed import KneeLocator

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

def preprocess():
    twitter_dataset = pd.read_csv('twitter_dataset.csv')

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

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(twitterX)


    return twitterX, twitterY, twitter_dataset, scaled_features


def kMeans():

    twitterX, twitterY, twitter_dataset, scaled_features = preprocess()

    gm = GaussianMixture(covariance_type='tied', n_components=18, n_init=10)
    gm.fit(scaled_features)
    print("GM Converged", gm.converged_)
    print("GM Convergence Iterations", gm.n_iter_)
    print("GM weights", gm.weights_)

    gm.predict(scaled_features)
    gm.predict_proba(scaled_features)
    gm.score_samples(scaled_features)

    aic = []
    bic = []

    for i in range(10):
        gm = GaussianMixture(covariance_type='spherical', n_components=9, n_init=10)
        gm.fit(scaled_features)
        aic.append(gm.aic(scaled_features))
        bic.append(gm.bic(scaled_features))

    plt.plot(aic, label="AIC")
    plt.plot(bic, label="BIC")
    # plt.xticks(np.arange(min(x), max(x) + 1, 1.0))
    # plt.xticks(range(1,18))
    plt.xlabel("Number of Clusters")
    plt.ylabel("Information Criterion")
    plt.legend()
    plt.show()

    twitter_trainingX, twitter_testingX, twitter_trainingY, twitter_testingY = train_test_split(twitterX, twitterY)

    error = []

    #citation: https://stackoverflow.com/questions/36566844/pca-projection-and-reconstruction-in-scikit-learn


    for i in range(1, 8):
        pca = FastICA(n_components=i)
        pca.fit(twitter_trainingX)
        U, S, VT = np.linalg.svd(twitter_trainingX - twitter_trainingX.mean(0))
        x_train_pca = pca.transform(twitter_trainingX)
        x_train_pca2 = (twitter_trainingX - pca.mean_).dot(pca.components_.T)
        x_projected = pca.inverse_transform(x_train_pca)
        x_projected2 = x_train_pca.dot(pca.components_) + pca.mean_
        loss = ((twitter_trainingX - x_projected) ** 2).mean()
        error.append(loss)

    plt.clf()
    plt.figure(figsize=(15, 15))
    plt.title("reconstruction error")
    plt.plot(error, 'r')
    plt.xticks(range(len(error)), range(1, 8), rotation='vertical')
    plt.xlim([-1, len(error)])
    plt.show()


    clf = MLPClassifier(alpha=0.001, hidden_layer_sizes=(8,), random_state=1,
                        solver='lbfgs')
    clf.fit(twitter_trainingX, twitter_trainingY)
    y_pred = clf.predict(twitter_testingX)

    print("Accuracy Score Normal", accuracy_score(twitter_testingY, y_pred))

    kmeans = KMeans(
        init="random",
        n_clusters=3,
        n_init=10,
        max_iter=300,
        random_state=42
    )
    kmeans.fit(scaled_features)
    labels = kmeans.fit_predict(twitter_testingX)

    print("Accuracy Score K-Means", accuracy_score(twitter_testingY, labels))

    for i in range(9):
        pca = PCA(n_components=i)
        pca.fit(scaled_features)
        cumsum = np.cumsum(pca.explained_variance_ratio_)

    plt.plot(cumsum, label="Explained Variance Ratio")
    # plt.xticks(np.arange(min(x), max(x) + 1, 1.0))
    # plt.xticks(range(1,18))
    plt.xlabel("Number of Dimensions")
    plt.ylabel("Explained Variance Ratio")
    plt.legend()
    plt.show()

    # ica
    num_batches = 100
    inc_pca = IncrementalPCA(n_components=5)
    for X_batch in np.array_split(scaled_features, num_batches):
        inc_pca.partial_fit(X_batch)
    X_reduced_inc = inc_pca.transform(scaled_features)

    # randomized projections
    rnd_pca = PCA(n_components=5, svd_solver="randomized")
    X_reduced_rand = rnd_pca.fit_transform(scaled_features)

    # citation: https://scikit-learn.org/stable/auto_examples/feature_selection/plot_feature_selection.html#sphx-glr-auto-examples-feature-selection-plot-feature-selection-py
    # k best
    scaler = MinMaxScaler()
    digits_indices = np.arange(twitterX.shape[-1])
    scaled_features_norm = scaler.fit_transform(scaled_features)
    k_selected = SelectKBest(f_classif, k=8)
    k_selected.fit(scaled_features_norm, twitterY)
    scores = -np.log10(k_selected.pvalues_)
    plt.bar(digits_indices - .45, scores, width=.2, label=r'Univariate score ($-Log(p_{value})$)')
    plt.xlabel("Features")
    plt.ylabel("F-Score")
    plt.show()

    digits

    kmeans = KMeans(
        init="random",
        n_clusters=5,
        n_init=10,
        max_iter=300,
        random_state=42
    )
    kmeans.fit(scaled_features)
    labels = kmeans.fit_predict(twitter_dataset)

    #the lowest SSE value
    print("KMeans Inertia", kmeans.inertia_)

    #final locations of the centroid
    print("KMeans Cluster Centers", kmeans.cluster_centers_)

    #num of iterations required to converge
    print("KMeans Iterations Required To Converge", kmeans.n_iter_)

    #labels
    print("KMeans Labels", kmeans.labels_[:5])

    kmeans_kwargs = {
        "init":"random",
        "n_init":10,
        "max_iter":300,
        "random_state":42,
    }

    sse = []
    for k in range(1, 18):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(scaled_features)
        sse.append(kmeans.inertia_)

    model = KMeans(n_clusters=9)
    elbow_visualizer = KElbowVisualizer(model, k=(2, 18))
    elbow_visualizer.fit(twitterX)
    elbow_visualizer.show()

    silhouette_visualizer = SilhouetteVisualizer(model, colors='yellowbrick')
    silhouette_visualizer.fit(twitterX)
    silhouette_visualizer.show()

    ic_visualizer = InterclusterDistance(model)
    ic_visualizer.fit(twitterX)
    ic_visualizer.show()

    X = twitter_dataset[:, []]
    plt.scatter()


def main():
    kMeans()

if __name__ == "__main__":
    main()