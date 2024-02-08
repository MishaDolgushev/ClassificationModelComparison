import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


class LogisticRegressor(BaseEstimator, ClassifierMixin):
    def __init__(self) -> None:
        self._weights = None

    def sigmoid(self, number: int):
        clipped_number = np.clip(number, -100, 100)
        return 1 / (1 + np.e ** (-clipped_number))

    def fit(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        batch_size: int,
        n_epoch: int = 10,
        learning_rate: float = 1e-3,
    ):
        self._weights = np.random.standard_normal(X_train.shape[1])

        for _ in tqdm(range(n_epoch)):
            train_data = np.hstack([X_train, Y_train.reshape(-1, 1)])
            np.random.shuffle(train_data)
            for batch in range(0, len(train_data), batch_size):
                X_batch = train_data[batch : batch + batch_size, :-1]
                y_batch = train_data[batch : batch + batch_size, -1]
                predictions = self.predict_probas(X_batch)
                gradient = -np.dot(X_batch.T, y_batch - predictions) / len(X_batch)

                self._weights -= learning_rate * gradient

        return self

    def predict_probas(self, X_test: np.ndarray):
        return self.sigmoid(X_test @ self._weights)

    def predict_labels(self, X_test: np.ndarray):
        probas = self.predict_probas(X_test)
        labels = (probas > 0.5).astype("int")
        return labels


