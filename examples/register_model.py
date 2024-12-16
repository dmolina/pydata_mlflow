from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import mlflow


def main():
    iris = load_iris()
    # Añadimos dónde se guardarán los datos
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    # Creo el experimento si no existe
    mlflow.set_experiment("Iris - max_depth study")

    # Testing different maximum-depth values
    for max_depth in range(1, 5):
        # Con with no es necesario iniciar y cerrar, es más cómodo
        with mlflow.start_run():
            # Registro el parámetro
            mlflow.log_param("max_depth", max_depth)

            # Create a decision tree classifier
            clf = DecisionTreeClassifier(max_depth=max_depth)

            # Perform cross-validation with 5 folds
            scores = cross_val_score(clf, iris.data, iris.target, cv=5).mean()
            # Print the cross-validation scores
            print(f"Cross-validation scores for max_depth={max_depth}: {scores:.5f}")
            # Registro la métrica
            mlflow.log_metric("accuracy", scores)


main()
