from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, cross_validate
import mlflow


def main():
    cancer = load_breast_cancer()
    # Añadimos dónde se guardarán los datos
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    # Creo el experimento si no existe
    mlflow.set_experiment("Cancer - max_depth study auto")
    mlflow.autolog()

    # Testing different maximum-depth values
    for max_depth in range(1, 5):
        # Con with no es necesario iniciar y cerrar, es más cómodo
        with mlflow.start_run(run_name=f"max_depth-{max_depth}"):
            # Create a decision tree classifier
            clf = DecisionTreeClassifier(max_depth=max_depth)

            # Perform cross-validation with 5 folds
            scores = cross_validate(
                clf,
                cancer.data,
                cancer.target,
                cv=5,
                scoring=["accuracy", "f1", "recall", "precision", "roc_auc"],
            )

            for score in scores:
                print(score)
                print(scores[score].mean())


main()
