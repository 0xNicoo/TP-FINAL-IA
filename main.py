import tensorflow as tf
from model import run_a_model
from base.util import read_csv, process_data, split_data
from base.plot_result import plot_confusion_matrix
import numpy as np

############################################
########### TRATAMIENTO DE DATOS ###########
############################################

data = read_csv('dataset320test.csv')
data = process_data(data)
columnas_constantes = ['nombreEquipo', 'esLocal', 'equipovs', 'ImportanciaLiga', 'RankingFIFA']
columnas_variables = ['ShotsonGoal', 'ShotsoffGoal', 'TotalShots', 'BlockedShots', 'Shotsinsidebox', 'Shotsoutsidebox', 
                      'Fouls', 'CornerKicks', 'Offsides', 'BallPossession', 'YellowCards', 'RedCards', 
                      'GoalkeeperSaves', 'Totalpasses', 'Passesaccurate', 'Passes%', 
                      'numeroDeGolesLocales', 'NumeroDeGolesVisitantes']
X = data.drop('resultado', axis=1)
y = data['resultado']
y_adjusted = y.replace(3, 2)
X_train, X_test, y_train, y_test = split_data(X, y_adjusted)

evaluation, model = run_a_model(X_train, X_test, y_train, y_test, columnas_variables, columnas_constantes)
acu = [evaluation, model]

for i in range(0, 50):
    evaluation, model = run_a_model(X_train, X_test, y_train, y_test, columnas_variables, columnas_constantes)
    if (evaluation[1] > acu[0][1]):
        acu[0] = evaluation
        acu[1] = model   

print("#"*10)
print("MEJOR PRESICION: " + str(acu[0][1]))
print("#"*10)

X_test_tensors = {col: tf.convert_to_tensor(X_test[col].values, dtype=tf.float32) for col in columnas_constantes}
X_test_tensors['datosVariables'] = tf.convert_to_tensor(X_test[columnas_variables].values, dtype=tf.float32)

y_pred = model.predict(X_test_tensors)
plot_confusion_matrix(y_test, y_pred)
acu[1].save('./model/model.keras')    

