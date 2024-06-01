import tensorflow as tf
from tensorflow import keras
import numpy as np
import random

# Definimos los equipos
equipos = {
    1: 'Argentina',
    2: 'Peru',
    3: 'Chile',
    4: 'Canada',
    5: 'Mexico',
    6: 'Ecuador',
    7: 'Venezuela',
    8: 'Jamaica',
    9: 'USA',
    10: 'Uruguay',
    11: 'Panama',
    12: 'Brazil',
    13: 'Colombia',
    14: 'Paraguay',
    15: 'Costa Rica',
    16: 'Bolivia'
}

# Fase de grupos
grupos = {
    'Grupo A': [1, 2, 3, 4],
    'Grupo B': [5, 6, 7, 8],
    'Grupo C': [9, 10, 11, 16],
    'Grupo D': [12, 13, 14, 15]
}

# Rankins FIFA
ranking_fifa = {
    1: 1,
    2: 22,
    3: 17,
    4: 67,
    5: 9,
    6: 46,
    7: 53,
    8: 47,
    9: 13,
    10: 16,
    11: 61,
    12: 3,
    13: 11,
    14: 35,
    15: 43,
    16: 82
}

# Función para simular partidos usando el modelo
def simular_partido(loaded_model, equipo1, esLocal, equipo2, ImportanciaLiga, RankingFIFA):
    # Crear tensores para cada entrada
    nombreEquipo = tf.convert_to_tensor([[equipo1]], dtype=tf.float32)
    esLocal = tf.convert_to_tensor([[esLocal]], dtype=tf.float32)
    equipovs = tf.convert_to_tensor([[equipo2]], dtype=tf.float32)
    ImportanciaLiga = tf.convert_to_tensor([[ImportanciaLiga]], dtype=tf.float32)
    ranking_fifa_tensor = tf.convert_to_tensor([[RankingFIFA]], dtype=tf.float32)
    X_pred_variables = tf.convert_to_tensor(np.zeros((1, 18)), dtype=tf.float32)  # Asegúrate que el tamaño aquí coincida con las variables esperadas

    # Pasar todas las entradas al modelo
    pred = loaded_model.predict([nombreEquipo, esLocal, equipovs, ImportanciaLiga, ranking_fifa_tensor, X_pred_variables])
    resultado = np.argmax(pred, axis=1)[0]  # Obtener el índice de la clase con mayor probabilidad (0, 1 o 2)
    return resultado  # 0: derrota, 1: empate, 2: victoria

# Cargar el modelo
loaded_model = keras.models.load_model('./model/model2.keras')

# Simulación de la fase de grupos
def simular_grupos():
    resultados_grupos = {grupo: {equipo: 0 for equipo in equipos_grupo} for grupo, equipos_grupo in grupos.items()}
    
    for grupo, equipos_grupo in grupos.items():
        for i in range(len(equipos_grupo)):
            for j in range(i + 1, len(equipos_grupo)):
                equipo1 = equipos_grupo[i]
                equipo2 = equipos_grupo[j]
                esLocal = random.choice([0, 1])
                resultado = simular_partido(loaded_model, equipo1, esLocal, equipo2, 45, ranking_fifa[equipo1])
                
                if resultado == 0:  # Victoria equipo1
                    resultados_grupos[grupo][equipo1] += 3
                elif resultado == 2:  # Derrota equipo1
                    resultados_grupos[grupo][equipo2] += 3
                elif resultado == 1:  # Empate
                    resultados_grupos[grupo][equipo1] += 1
                    resultados_grupos[grupo][equipo2] += 1
    
    return resultados_grupos

# Simular fase de grupos
resultados_grupos = simular_grupos()
print("Resultados Fase de Grupos:")
print(resultados_grupos)

# Determinar equipos que pasan a octavos de final (dos primeros de cada grupo)
def determinar_clasificados(resultados_grupos):
    clasificados = []
    for grupo, resultados in resultados_grupos.items():
        clasificados.extend(sorted(resultados, key=resultados.get, reverse=True)[:2])
    return clasificados

clasificados_octavos = determinar_clasificados(resultados_grupos)
print("Equipos Clasificados a Octavos de Final:")
print([equipos[e] for e in clasificados_octavos])

# Simulación de las fases eliminatorias
def simular_eliminatorias(clasificados, fase):
    if len(clasificados) == 1:
        print(f"¡El campeón de la Copa América es {equipos[clasificados[0]]}!")
        return
    
    print(f"Simulando {fase}:")
    nuevos_clasificados = []
    for i in range(0, len(clasificados), 2):
        equipo1 = clasificados[i]
        equipo2 = clasificados[i + 1]
        esLocal = random.choice([0, 1])
        resultado = simular_partido(loaded_model, equipo1, esLocal, equipo2, 35, ranking_fifa[equipo1])
        if resultado == 2:
            nuevos_clasificados.append(equipo1)
            print(f"{equipos[equipo1]} gana a {equipos[equipo2]}")
        else:
            nuevos_clasificados.append(equipo2)
            print(f"{equipos[equipo2]} gana a {equipos[equipo1]}")
    
    if fase == "Final":
        simular_eliminatorias(nuevos_clasificados, "Ganador")
    elif fase == "Semifinales":
        simular_eliminatorias(nuevos_clasificados, "Final")
    elif fase == "Cuartos de Final":
        simular_eliminatorias(nuevos_clasificados, "Semifinales")
    elif fase == "Octavos de Final":
        simular_eliminatorias(nuevos_clasificados, "Cuartos de Final")

# Simular octavos de final
random.shuffle(clasificados_octavos)
simular_eliminatorias(clasificados_octavos, "Octavos de Final")
