from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score


def main():
    iris = load_iris()

    # Testing different maximum-depth values
    for max_depth in range(1, 5):
        # Create a decision tree classifier
        clf = DecisionTreeClassifier(max_depth=max_depth)

        # Perform cross-validation with 5 folds
        scores = cross_val_score(clf, iris.data, iris.target, cv=5).mean()

        # Print the cross-validation scores
        print(f"Cross-validation scores for max_depth={max_depth}: {scores:.5f}")


main()
