import os
import sys
from dataclasses import dataclass
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from scipy import sparse
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            if sparse.issparse(y_train):
                y_train = y_train.toarray()
            if sparse.issparse(y_test):
                y_test = y_test.toarray()

            if y_train.ndim > 1:
                y_train = y_train.ravel()
            if y_test.ndim > 1:
                y_test = y_test.ravel()

            params = {
                "Logistic Regression": {'C': 0.001, 'penalty': 'l2', 'solver': 'liblinear'},
                "SVM Classifier": {'kernel': 'linear', 'C': 0.0001},
                "Decision Tree": {'criterion': 'entropy', 'max_depth': 1, 'min_samples_leaf': 5},
                "Random Forest Classifier": {'n_estimators': 50, 'max_depth': 1, 'min_samples_leaf': 5},
                "Gradient Boosting Classifier": {'n_estimators': 15, 'max_depth': 1, 'min_samples_leaf': 3, 'learning_rate': 0.1},
                "AdaBoost Classifier": {'n_estimators': 2, 'learning_rate': 1.0, 'algorithm': 'SAMME'}
            }

            models = {
                "Logistic Regression": LogisticRegression(**params["Logistic Regression"]),
                "SVM Classifier": svm.SVC(**params["SVM Classifier"]),
                "Decision Tree": DecisionTreeClassifier(**params["Decision Tree"]),
                "Random Forest Classifier": RandomForestClassifier(**params["Random Forest Classifier"]),
                "Gradient Boosting Classifier": GradientBoostingClassifier(**params["Gradient Boosting Classifier"]),
                "AdaBoost Classifier": AdaBoostClassifier(**params["AdaBoost Classifier"])
            }


            model_report = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, param=params)
            
            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found", "All models performed below threshold")

            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")

            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)

            predicted = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, predicted)
            return accuracy

        except Exception as e:
            raise CustomException(e, sys)

# Additional code (like imports) might be necessary depending on your project structure




