import tensorflow as tf
from tensorflow import keras
from base.run_model import predict
import numpy as np

columnas_constantes = ['nombreEquipo', 'esLocal', 'equipovs', 'ImportanciaLiga', 'RankingFIFA']
columnas_variables = ['ShotsonGoal', 'ShotsoffGoal', 'TotalShots', 'BlockedShots', 'Shotsinsidebox', 'Shotsoutsidebox', 
                      'Fouls', 'CornerKicks', 'Offsides', 'BallPossession', 'YellowCards', 'RedCards', 
                      'GoalkeeperSaves', 'Totalpasses', 'Passesaccurate', 'Passes%', 
                      'numeroDeGolesLocales', 'NumeroDeGolesVisitantes']

X_pred_constantes = {'nombreEquipo': tf.convert_to_tensor([1]), 
                     'esLocal': tf.convert_to_tensor([0]),
                     'equipovs': tf.convert_to_tensor([12]), 
                     'ImportanciaLiga': tf.convert_to_tensor([40]),
                     'RankingFIFA': tf.convert_to_tensor([1])}


X_pred_variables = tf.convert_to_tensor(np.zeros((1, len(columnas_variables))), dtype=tf.float32)

loaded_model = keras.models.load_model('./model/model.keras')

predictions = predict(loaded_model, X_pred_constantes, X_pred_variables, columnas_constantes)
print(predictions)