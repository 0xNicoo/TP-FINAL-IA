import tensorflow as tf
from base.build_model import build_model
from base.run_model import fit, evaluate


def run_a_model(X_train, X_test, y_train, y_test, columnas_variables, columnas_constantes):
    ##################################################################
    # Parseo de datos a tensor para que lo pueda procesar TensorFlow #
    ##################################################################
    X_train_tensors = {col: tf.convert_to_tensor(X_train[col].values, dtype=tf.float32) for col in columnas_constantes}
    X_train_tensors['datosVariables'] = tf.convert_to_tensor(X_train[columnas_variables].values, dtype=tf.float32)
    X_test_tensors = {col: tf.convert_to_tensor(X_test[col].values, dtype=tf.float32) for col in columnas_constantes}
    X_test_tensors['datosVariables'] = tf.convert_to_tensor(X_test[columnas_variables].values, dtype=tf.float32)
    y_train_tensor = tf.convert_to_tensor(y_train.values, dtype=tf.int64)
    y_test_tensor = tf.convert_to_tensor(y_test.values, dtype=tf.int64)

    model = build_model(X_train.shape[1],columnas_constantes)
    fit(model, X_train_tensors, y_train_tensor)
    evaluation = evaluate(model, X_test_tensors, y_test_tensor)

    return evaluation, model
