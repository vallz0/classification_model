import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

data = pd.read_csv('data/personagens.csv')

def main(dataset) -> None:

    sns.countplot(x='classe', data=dataset)
    plt.show()
    sns.heatmap(dataset.select_dtypes(include=['number']).corr(), annot=True, cmap="coolwarm")

    plt.show()

    X =  dataset.iloc[:, 0:6].values

    y = dataset.iloc[:, 6].values
    y = (y == 'Bart')


    X_training, X_test, y_training, y_test = train_test_split(X, y, test_size= 0.2)

    rede_neural = tf.keras.models.Sequential()
    rede_neural.add(tf.keras.layers.Dense(units=4, activation='relu', input_shape=(6, )))
    rede_neural.add(tf.keras.layers.Dense(units=4, activation='relu'))
    rede_neural.add(tf.keras.layers.Dense(units=4, activation='relu'))
    rede_neural.add(tf.keras.layers.Dense(units=4, activation='sigmoid'))

    rede_neural.summary()
    rede_neural.compile(optimizer='Adam', loss='binary_crossentropy',metrics = ['accuracy'])

    history = rede_neural.fit(X_training, y_training, epochs=50, validation_split=0.1)

    plt.plot(history.history['val_loss'])
    plt.plot(history.history['val_accuracy'])

    previsoes = rede_neural.predict(X_test)

    previsoes = (previsoes > 0.5)

    accuracy_score(previsoes, y_test)

    cm = confusion_matrix(y_test, previsoes)
    sns.heatmap(cm, annot=True)

if __name__ == '__main__':
    main(data)
