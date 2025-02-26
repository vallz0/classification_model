import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score


class NeuralNetworkClassifier:
    def __init__(self, dataset_path: str):
        self.dataset = self._load_dataset(dataset_path)
        self.model = self._build_model()

    def _load_dataset(self, dataset_path: str):
        return pd.read_csv(dataset_path)

    def preprocess_data(self):
        X, y = self._extract_features_and_labels()
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def _extract_features_and_labels(self):
        X = self.dataset.iloc[:, 0:6].values
        y = (self.dataset.iloc[:, 6].values == 'Bart')
        return X, y

    def _build_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(units=4, activation='relu', input_shape=(6,)),
            tf.keras.layers.Dense(units=4, activation='relu'),
            tf.keras.layers.Dense(units=4, activation='relu'),
            tf.keras.layers.Dense(units=1, activation='sigmoid')
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self, X_train, y_train, epochs=50):
        history = self.model.fit(X_train, y_train, epochs=epochs, validation_split=0.1, verbose=0)
        self._plot_training_history(history)

    def _plot_training_history(self, history):
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.legend()
        plt.show()

    def evaluate(self, X_test, y_test):
        predictions = self._make_predictions(X_test)
        self._display_metrics(y_test, predictions)

    def _make_predictions(self, X_test):
        return (self.model.predict(X_test) > 0.5)

    def _display_metrics(self, y_test, predictions):
        accuracy = accuracy_score(y_test, predictions)
        print(f'Accuracy: {accuracy:.2f}')
        self._plot_confusion_matrix(y_test, predictions)

    def _plot_confusion_matrix(self, y_test, predictions):
        cm = confusion_matrix(y_test, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.show()

    def visualize_data(self):
        self._plot_class_distribution()
        self._plot_correlation_matrix()

    def _plot_class_distribution(self):
        sns.countplot(x='classe', data=self.dataset)
        plt.show()

    def _plot_correlation_matrix(self):
        sns.heatmap(self.dataset.select_dtypes(include=['number']).corr(), annot=True, cmap='coolwarm')
        plt.show()


if __name__ == '__main__':
    classifier = NeuralNetworkClassifier('personagens.csv')
    classifier.visualize_data()
    X_train, X_test, y_train, y_test = classifier.preprocess_data()
    classifier.train(X_train, y_train)
    classifier.evaluate(X_test, y_test)
