import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, BatchNormalization
from tensorflow.keras.optimizers import Adam

def build_model(num_features, columnas_constantes):
    inputs = {col: Input(shape=(1,), name=col) for col in columnas_constantes}
    datos_variables_input = Input(shape=(18,), name='datosVariables')
    
    constantes = Concatenate()([inputs[col] for col in columnas_constantes])
    constantes = Dense(5, activation='relu')(constantes)
    variables = Dense(18, activation='relu')(datos_variables_input)
    combinacion = Concatenate()([constantes, variables])
    combinacion = Dense(46, activation='relu')(combinacion)
    combinacion = BatchNormalization()(combinacion)
    combinacion = Dense(23, activation='relu')(combinacion)
    combinacion = BatchNormalization()(combinacion)
    combinacion = Dense(11, activation='relu')(combinacion)
    combinacion = BatchNormalization()(combinacion)
    output = Dense(3, activation='softmax')(combinacion)

    model = tf.keras.Model(inputs=list(inputs.values()) + [datos_variables_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.003), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model