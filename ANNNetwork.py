import os
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, KFold, learning_curve
from sklearn.model_selection import train_test_split as sp, train_test_split
from sklearn.base import BaseEstimator
import numpy as np
from tensorflow_estimator.python.estimator.estimator import Estimator


# ======================== CITATION BELOW ==============================================#
 #https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def pack_features_vector(features, labels):
    """Pack the features into a single array."""
    features = tf.stack(list(features.values()), axis=1)
    return features, labels


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def loss(model, x, y, training):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    y_ = model(x, training=training)

    return loss_object(y_true=y, y_pred=y_)

# ======================== CITATION ABOVE ==============================================#

def ANN():

    class_names = ['Positive', 'Neutral', 'Negative']
    twitter_dataset = pd.read_csv('Sentiment.csv')
    column_names_without_label = ['candidate', 'candidate_confidence', 'relevant_yn', 'relevant_yn_confidence',
                    'sentiment_confidence', 'subject_matter', 'subject_matter_confidence',
                    'retweet_count']

    column_names = ['candidate', 'candidate_confidence', 'relevant_yn', 'relevant_yn_confidence',
                    'sentiment_confidence', 'subject_matter', 'subject_matter_confidence',
                    'retweet_count', 'sentiment']
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

    scaler = MinMaxScaler()
    scaler.fit(twitter_dataset)
    normalized = scaler.transform(twitter_dataset)
    inverse = scaler.inverse_transform(normalized)

    twitterX = twitter_dataset[column_names_without_label].copy()
    twitterY = twitter_dataset[label].copy()




    twitter_trainingX, twitter_testingX, twitter_trainingY, twitter_testingY = train_test_split(twitterX, twitterY, test_size=0.3, random_state=0,
                                                                         stratify=twitterY)

    training_concat = pd.concat([twitter_trainingX, twitter_trainingY], axis=1)

    training_concat.to_csv('twitter_trainig_processed.csv', encoding='utf-8', columns=column_names, index=False)

    testing_concat = pd.concat([twitter_testingX, twitter_testingY], axis=1)

    training_concat.to_csv('twitter_testing_processed.csv', encoding='utf-8', columns=column_names, index=False)


    clf = MLPClassifier(alpha=0.001, hidden_layer_sizes=(8,), random_state=1,
                  solver='lbfgs')
    clf.fit(twitter_trainingX, twitter_trainingY)
    clf.predict(twitter_testingX)

    optimizers = ['lbfgs', 'sgd', 'adam']
    max_iters = [100, 500, 1000]
    batch_size = [5, 10, 100]
    alpha = np.logspace(-5, 3, 5)
    seed = 52

    #param_grid = dict(solver=optimizers, max_iter=max_iters, batch_size=batch_size, alpha=alpha)

    #grid = GridSearchCV(estimator=clf, param_grid=param_grid, cv=KFold(random_state=seed), verbose=10,
                        #scoring='accuracy')
    #grid_results = grid.fit(twitter_trainingX, twitter_trainingY)

    train_sizes = np.linspace(.1, 1.0, 5)

    # ======================== CITATION BELOW ==============================================#
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

    cv = None
    n_jobs = None
    ylim = None

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(clf, twitter_trainingX, twitter_trainingY, cv=cv, n_jobs=n_jobs,
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

    batch_size = 32
    train_dataset = tf.data.experimental.make_csv_dataset(
        'twitter_processed.csv',
        batch_size,
        select_columns=column_names,
        label_name='sentiment',
        num_epochs=1,
        ignore_errors=True,
    )

    train_dataset = train_dataset.map(pack_features_vector)

    features, labels = next(iter(train_dataset))

    print(features[:5])


    # ======================== CITATION BELOW ==============================================#
    #https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(8,)),  # input shape required
        tf.keras.layers.Dense(10, activation=tf.nn.relu),
        tf.keras.layers.Dense(3)
    ])

    predictions = model(features)
    predictions[:5]

    tf.nn.softmax(predictions[:5])

    print("Prediction: {}".format(tf.argmax(predictions, axis=1)))
    print("    Labels: {}".format(labels))

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    l = loss(model, features, labels, training=False)
    print("Loss test: {}".format(l))

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    loss_value, grads = grad(model, features, labels)

    print("Step: {}, Initial Loss: {}".format(optimizer.iterations.numpy(),
                                              loss_value.numpy()))

    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    print("Step: {},         Loss: {}".format(optimizer.iterations.numpy(),
                                              loss(model, features, labels, training=True).numpy()
                                              ))

    ## Note: Rerunning this cell uses the same model variables

    # Keep results for plotting
    train_loss_results = []
    train_accuracy_results = []

    num_epochs = 201

    for epoch in range(num_epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        # Training loop - using batches of 32
        #for x, y in train_dataset:
            # Optimize the model
            #loss_value, grads = grad(model, x, y)
            #optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Track progress
            #epoch_loss_avg.update_state(loss_value)  # Add current batch loss
            # Compare predicted label to actual label
            # training=True is needed only if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            #epoch_accuracy.update_state(y, model(x, training=True))

        # End epoch
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

        #if epoch % 50 == 0:
            #print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                        #epoch_loss_avg.result(),
                                                                        #epoch_accuracy.result()))
        fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
        fig.suptitle('Training Metrics')

        axes[0].set_ylabel("Loss", fontsize=14)
        axes[0].plot(train_loss_results)

        axes[1].set_ylabel("Accuracy", fontsize=14)
        axes[1].set_xlabel("Epoch", fontsize=14)
        axes[1].plot(train_accuracy_results)
        #plt.show()

        test_dataset = tf.data.experimental.make_csv_dataset(
            'twitter_testing_processed.csv',
            batch_size,
            select_columns=column_names,
            label_name='sentiment',
            num_epochs=1,
            ignore_errors=True,
        )

        test_dataset = test_dataset.map(pack_features_vector)

        test_accuracy = tf.keras.metrics.Accuracy()

        # ======================== CITATION ABOVE ==============================================#

        #for (x, y) in test_dataset:
            # training=False is needed only if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            #logits = model(x, training=False)
            #prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
            #test_accuracy(prediction, y)

        #print("Test set accuracy: {:.3%}".format(test_accuracy.result()))

        #test unlabeled examples
        #predict_dataset = tf.convert_to_tensor(twitter_trainingX)

        #predictions = model(predict_dataset, training=False)

        #for i, logits in enumerate(predictions):
            #class_idx = tf.argmax(logits).numpy()
            #p = tf.nn.softmax(logits)[class_idx]
            #name = class_names[class_idx]
            #print("Example {} prediction: {} ({:4.1f}%)".format(i, name, 100 * p))

    #AdaBoost = AdaBoostClassifier(n_estimators=400, learning_rate=1, algorithm='SAMME')
    #AdaBoost.fit(twitter_trainingX, twitter_trainingY)
    #prediction = AdaBoost.score(twitter_trainingX, twitter_trainingY)

    optimizers = ['lbfgs', 'sgd', 'adam']
    max_iters = [100, 500, 1000]
    batch_size = [5, 10, 100]
    seed = 52

    param_grid = dict(solver=optimizers, max_iter = max_iters, batch_size=batch_size)

    grid = GridSearchCV(estimator=clf, param_grid = param_grid, cv=KFold(random_state=seed), verbose=10,
                        scoring='accuracy')
    grid_results = grid.fit(twitter_trainingX, twitter_trainingY)

    print("Best: {0}, using {1}".format(grid_results.best_score_, grid_results.best_params_))
    #print('Accuracy post-boosting: ', prediction * 100, '%')


def main():
    ANN()


if __name__ == "__main__":
    main()
