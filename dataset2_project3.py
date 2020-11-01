import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.decomposition import PCA, IncrementalPCA, FastICA
from sklearn.pipeline import Pipeline
import time
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.random_projection import GaussianRandomProjection
from yellowbrick.cluster import InterclusterDistance
from yellowbrick.cluster import SilhouetteVisualizer
from yellowbrick.cluster import KElbowVisualizer
from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd
import seaborn as sns
from kneed import KneeLocator
from sklearn.feature_selection import chi2


def kMeans():
    # citation: https://realpython.com/k-means-clustering-python/
    digits = load_digits()

    # features
    digits_features = digits.data[:, 0:-1]
    # label
    label = digits.data[:, -1]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(digits_features)

    # citation: hands on machine learning
    gm = GaussianMixture(covariance_type='spherical', n_components=8, n_init=10)
    gm.fit(scaled_features)
    print("GM Converged", gm.converged_)
    print("GM Convergence Iterations", gm.n_iter_)
    print("GM weights", gm.weights_)

    gm.predict(scaled_features)
    gm.predict_proba(scaled_features)
    gm.score_samples(scaled_features)

    aic = []
    bic = []

    for i in range(21):
        gm = GaussianMixture(covariance_type='spherical', n_components=20, n_init=10)
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

    # x_centered = digits_features - digits_features.mean(axis=0)
    # U, s, Vt = np.linalg.svd(x_centered)
    # c1 = Vt.T[:, 0]
    # c2 = Vt.T[:, 1]

    # W2 = Vt.T[:, :2]
    # X2D = x_centered.dot(W2)

    # pca = PCA()
    # pca.fit(scaled_features)
    # cumsum = np.cumsum(pca.explained_variance_ratio_)
    # d = np.argmax(cumsum >= 0.95) + 1

    # pca = PCA(n_components=0.95)
    # X_reduced = pca.fit_transform(scaled_features)

    explained_variance = []
    for i in range(63):
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

    digits_trainingX, digits_testingX, digits_trainingY, digits_testingY = train_test_split(digits_features, label)

    # ica
    # citation: https://stackoverflow.com/questions/36566844/pca-projection-and-reconstruction-in-scikit-learn

    error = []

    for i in range(1, 50):
        pca = PCA(n_components=i)
        pca.fit(digits_trainingX)
        U, S, VT = np.linalg.svd(digits_trainingX - digits_trainingX.mean(0))
        x_train_pca = pca.transform(digits_trainingX)
        x_train_pca2 = (digits_trainingX - pca.mean_).dot(pca.components_.T)
        x_projected = pca.inverse_transform(x_train_pca)
        x_projected2 = x_train_pca.dot(pca.components_) + pca.mean_
        loss = ((digits_trainingX - x_projected) ** 2).mean()
        error.append(loss)

    plt.clf()
    plt.figure(figsize=(15, 15))
    plt.title("reconstruction error")
    plt.plot(error, 'r')
    plt.xticks(range(len(error)), range(1, 50), rotation='vertical')
    plt.xlim([-1, len(error)])
    plt.show()

    clf = MLPClassifier(alpha=0.001, hidden_layer_sizes=(8,), random_state=1,
                        solver='lbfgs')
    clf.fit(digits_trainingX, digits_trainingY)
    y_pred = clf.predict(digits_testingX)
    print("Accuracy Score Normal", accuracy_score(digits_testingY, y_pred))

    k_acc = []
    k_gm = []
    time_arr = []
    for k in range(1, 15):
        kmeans = KMeans(n_clusters=k)
        X_train = kmeans.fit_transform(digits_trainingX)
        X_test = kmeans.transform(digits_testingX)
        start_time = time.time()
        clf = MLPClassifier(alpha=0.001, hidden_layer_sizes=(8,), random_state=1,
                            solver='lbfgs')
        clf.fit(X_train, digits_trainingY)
        total_time = time.time() - start_time
        y_pred = clf.predict(X_test)
        score = accuracy_score(digits_testingY, y_pred)
        k_acc.append(score)
        time_arr.append(total_time)

    plt.plot(k_acc, label="K-Means")
    plt.plot(time_arr, label="Computation Time")
    # plt.xticks(np.arange(min(x), max(x) + 1, 1.0))
    # plt.xticks(range(1,18))
    plt.xlabel("k # of clusters")
    plt.ylabel("NN Accuracy")
    plt.legend()
    plt.show()

    acc = []
    acc_ica = []
    acc_rca = []
    for i in range(1, 40):
        pca = PCA(n_components=i)
        X_train = pca.fit_transform(digits_trainingX)
        X_test = pca.transform(digits_testingX)
        clf = MLPClassifier(alpha=0.001, hidden_layer_sizes=(8,), random_state=1,
                            solver='lbfgs')
        clf.fit(X_train, digits_trainingY)
        y_pred = clf.predict(X_test)
        score = accuracy_score(digits_testingY, y_pred)
        acc.append(score)

        ica = FastICA(n_components=i)
        x_train_i = ica.fit_transform(digits_trainingX)
        x_test_i = ica.transform(digits_testingX)
        clf.fit(x_train_i, digits_trainingY)
        y_pred_i = clf.predict(x_test_i)
        score_i = accuracy_score(digits_testingY, y_pred_i)
        acc_ica.append(score_i)

        rca = GaussianRandomProjection(n_components=i)
        x_train_r = rca.fit_transform(digits_trainingX)
        x_test_r = rca.transform(digits_testingX)
        clf.fit(x_train_r, digits_trainingY)
        y_pred_r = clf.predict(x_test_r)
        score_r = accuracy_score(digits_testingY, y_pred_r)
        acc_rca.append(score_r)

    plt.plot(acc, label="PCA")
    plt.plot(acc_ica, label="ICA")
    plt.plot(acc_rca, label="RCA")
    # plt.xticks(np.arange(min(x), max(x) + 1, 1.0))
    # plt.xticks(range(1,18))
    plt.xlabel("Components")
    plt.ylabel("NN Accuracy")
    plt.legend()
    plt.show()
    # cumsum = np.cumsum(pca.explained_variance_ratio_)
    # d = np.argmax(cumsum >= 0.95) + 1

    # randomized projections
    rnd_pca = PCA(n_components=50, svd_solver="randomized")
    X_reduced_rand = rnd_pca.fit_transform(scaled_features)

    # citation: https://scikit-learn.org/stable/auto_examples/feature_selection/plot_feature_selection.html#sphx-glr-auto-examples-feature-selection-plot-feature-selection-py
    # k best
    scaler = MinMaxScaler()
    digits_indices = np.arange(digits_features.shape[-1])
    scaled_features_norm = scaler.fit_transform(scaled_features)
    k_selected = SelectKBest(f_classif, k=50)
    k_selected.fit(scaled_features_norm, label)
    scores = -np.log10(k_selected.pvalues_)
    plt.bar(digits_indices - .45, scores, width=.2, label=r'Univariate score ($-Log(p_{value})$)')
    plt.xlabel("Features")
    plt.ylabel("F-Score")
    plt.show()

    gm = GaussianMixture(covariance_type='spherical', n_components=8, n_init=10)
    gm.fit(X_reduced_inc)
    print("GM Converged - PCA Inc", gm.converged_)
    print("GM Convergence Iterations", gm.n_iter_)
    print("GM weights", gm.weights_)

    gm.predict(X_reduced_inc)
    gm.predict_proba(X_reduced_inc)
    gm.score_samples(X_reduced_inc)

    kmeans = KMeans(
        init="random",
        n_clusters=63,
        n_init=10,
        max_iter=300,
        random_state=42
    )
    kmeans.fit(scaled_features)

    # the lowest SSE value
    print("KMeans Inertia", kmeans.inertia_)

    # final locations of the centroid
    print("KMeans Cluster Centers", kmeans.cluster_centers_)

    # num of iterations required to converge
    print("KMeans Iterations Required To Converge", kmeans.n_iter_)

    # labels
    print("KMeans Labels", kmeans.labels_[:5])

    kmeans_kwargs = {
        "init": "random",
        "n_init": 10,
        "max_iter": 300,
        "random_state": 42,
    }

    sse = []
    for k in range(1, 63):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(scaled_features)
        sse.append(kmeans.inertia_)

    kl = KneeLocator(
        range(1, 63), sse, curve="convex", direction="decreasing"
    )

    # optimal k (number of clusters) for this dataset
    print("Elbow", kl.elbow)

    clf = MLPClassifier(alpha=0.001, hidden_layer_sizes=(8,), random_state=1,
                        solver='lbfgs')
    clf.fit(digits_trainingX, digits_trainingY)
    y_pred = clf.predict(digits_testingX)

    model = KMeans(n_clusters=5)
    kmeans.fit(scaled_features)
    labels = kmeans.fit_predict(digits_testingX)

    print("Accuracy Score Normal", accuracy_score(digits_testingY, y_pred))
    print("Accuracy Score K-Means", accuracy_score(digits_testingY, labels))

    elbow_visualizer = KElbowVisualizer(model, k=(2, 63))
    elbow_visualizer.fit(digits_features)
    elbow_visualizer.show()

    silhouette_visualizer = SilhouetteVisualizer(model, colors='yellowbrick')
    silhouette_visualizer.fit(digits_features)
    silhouette_visualizer.show()

    ic_visualizer = InterclusterDistance(model)
    ic_visualizer.fit(digits_features)
    ic_visualizer.show()

    # gmm = GaussianMixture(n_components=7).fit(digits_features)
    # labels = gmm.predict(digits_features)
    # plt.scatter(digits_features[:, 0], digits_features[:, 1], c=labels, s=40, cmap='viridis')
    # plt.show()

    # digits_features_pd = pd.DataFrame(data=digits_features[1:, 1:],
    # index=digits_features[1:,0],
    # columns=digits_features[0,1:])

    # pd.plotting.scatter_matrix(digits_features_pd)

    # probs = GaussianMixture.predict_proba(digits_features)
    # print(probs[:5].round(3))

    kmeans = KMeans(
        init="random",
        n_clusters=18,
        n_init=10,
        max_iter=300,
        random_state=42
    )
    kmeans.fit(X_reduced_inc)

    # the lowest SSE value
    print("KMeans Inertia", kmeans.inertia_)

    # final locations of the centroid
    print("KMeans Cluster Centers", kmeans.cluster_centers_)

    # num of iterations required to converge
    print("KMeans Iterations Required To Converge", kmeans.n_iter_)

    # labels
    print("KMeans Labels", kmeans.labels_[:5])

    kmeans_kwargs = {
        "init": "random",
        "n_init": 10,
        "max_iter": 300,
        "random_state": 42,
    }

    sse = []
    for k in range(1, 18):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(scaled_features)
        sse.append(kmeans.inertia_)

    kl = KneeLocator(
        range(1, 18), sse, curve="convex", direction="decreasing"
    )

    # optimal k (number of clusters) for this dataset
    print("Elbow", kl.elbow)

    model = KMeans()
    elbow_visualizer = KElbowVisualizer(model, k=(2, 18))
    elbow_visualizer.fit(X_reduced_inc)
    elbow_visualizer.show()

    silhouette_visualizer = SilhouetteVisualizer(model, colors='yellowbrick')
    silhouette_visualizer.fit(X_reduced_inc)
    silhouette_visualizer.show()

    ic_visualizer = InterclusterDistance(model)
    ic_visualizer.fit(X_reduced_inc)
    ic_visualizer.show()


def main():
    kMeans()


if __name__ == "__main__":
    main()
