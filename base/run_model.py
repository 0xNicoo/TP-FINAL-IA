def fit(model, X_train_tensors, y_train_tensor, epochs=100, batch_size=32, validation_split=0.2):
    return model.fit(x=X_train_tensors, y=y_train_tensor, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

def evaluate(model, X_test_tensors, y_test_tensor):
    return model.evaluate(x=X_test_tensors, y=y_test_tensor)

def predict(model, X_pred_constantes, X_pred_variables, columnas_constantes):
    return model.predict([X_pred_constantes[col] for col in columnas_constantes] + [X_pred_variables])