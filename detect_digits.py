import pandas as pd
import numpy as np
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib
from matplotlib import pyplot as plt


def main():
    # import dataset
    digits = datasets.load_digits()

    # clean data
    X_train, X_test, y_train, y_test = train_test_split(digits.data[:-1], digits.target[:-1], random_state=0)

    # hyper parameters
    parameters = {'kernel': ['linear', 'rbf'],
                  'C': [1e-05, 1e-04, 1e-03, 1e-02]}

    # model
    clf = svm.SVC()
    model = GridSearchCV(clf, parameters, cv=3)
    model.fit(X_train, y_train)
    print('Best_score_ : , ', model.best_score_)
    print('Best_params_ : , ', model.best_params_)

    # validating
    best_model = model.best_estimator_
    score_test = best_model.score(X_test, y_test)
    print("Test accuracy : ", score_test)

    # predict
    print("Predicting")
    digit_to_test = 1
    some_digit = digits.data[digit_to_test]
    digit_target = digits.target[digit_to_test]
    prediction = best_model.predict([some_digit])
    print("Prediction ", prediction)
    print("Real Value ", digit_target)

    # plot
    some_digit_image = some_digit.reshape(8, 8)
    plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()