from sklearn.neural_network import MLPClassifier
from sklearn.model_selection  import cross_validate

def sklearn_reference_XOR():
    X = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]


    y = [0., 1., 1., 0. ]
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(2, ), random_state=42,max_iter=200000)

    clf.fit(X, y)
    print("Training set score: %f" % clf.score(X, y))
    #print("Test set score: %f" % mlp.score(learnset, learnlabels))
    scores = cross_validate(clf, X=X, y=y, cv=2, scoring='neg_mean_squared_error')
    print(scores)
if __name__ == '__main__':
    sklearn_reference_XOR()