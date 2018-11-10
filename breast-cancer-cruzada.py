import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score


previsores = pd.read_csv('entradas-breast.csv')
classe = pd.read_csv('saidas-breast.csv')

def criarRede():
    classificador = Sequential()
    
    classificador.add(Dense(units = 16, activation = 'relu', 
                        kernel_initializer = 'random_uniform', input_dim = 30))
    classificador.add(Dropout(0.2))
    
    classificador.add(Dense(units = 8, activation = 'relu', 
                        kernel_initializer = 'random_uniform'))
    classificador.add(Dropout(0.2))
    
    classificador.add(Dense(units = 4, activation = 'relu', 
                        kernel_initializer = 'normal'))
    
    classificador.add(Dense(units = 1, activation = 'sigmoid'))
    otimizador = keras.optimizers.Adamax(lr = 0.002, decay = 0.0001, clipvalue = 1.25)
    classificador.compile(optimizer = otimizador, loss = 'binary_crossentropy',
                      metrics = ['binary_accuracy'])
    return classificador

classificador = KerasClassifier(build_fn = criarRede,
                                epochs = 150,
                                batch_size = 15)
resultados = cross_val_score(estimator = classificador,
                             X = previsores, y = classe,
                             cv = 10, scoring = 'accuracy')
media = resultados.mean()
desvio = resultados.std()
print(resultados)
print("media:%f desvio:%f"% (media, desvio))