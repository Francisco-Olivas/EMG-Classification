import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import randint, uniform
import joblib

# FILTRO PASA BANDA
def butter_bandpass_filter(data, lowcut=20.0, highcut=450.0, fs=1000.0, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# EXTRACCIÓN DE CARACTERÍSTICAS
def extraer_caracteristicas(senal, etiqueta, ventana=100, traslape=50):
    X, y = [], []
    for i in range(0, len(senal) - ventana, traslape):
        ventana_senal = senal[i:i+ventana]
        rms = np.sqrt(np.mean(np.square(ventana_senal)))
        media = np.mean(ventana_senal)
        varianza = np.var(ventana_senal)
        maximo = np.max(ventana_senal)
        minimo = np.min(ventana_senal)
        rango = maximo - minimo
        X.append([rms, media, varianza, maximo, minimo, rango])
        y.append(etiqueta)
    return X, y

# CARGA DE DATOS 
def cargar_dataset(nombre_archivo, etiqueta, fs=1000, ventana=100, traslape=50):
    df = pd.read_csv(nombre_archivo)
    emg = df["Valor Analogico"].astype(float).to_numpy()
    emg_filtrada = butter_bandpass_filter(emg, fs=fs)
    emg_rectificada = np.abs(emg_filtrada)  # Rectificación
    return extraer_caracteristicas(emg_rectificada, etiqueta, ventana=ventana, traslape=traslape)

#X y y mismo tamaño
def sincronizar_datos(X, y):
    min_len = min(len(X), len(y))
    return np.array(X[:min_len]), np.array(y[:min_len])

ventana = 100
traslape = 50

# CARGAR TODOS LOS DATOS
#X1, y1 = cargar_dataset('relajacion1.csv', 0, ventana=ventana, traslape=traslape)
#X2, y2 = cargar_dataset('medio agarre.csv', 1, ventana=ventana, traslape=traslape)
#X22, y22 = cargar_dataset('medio agarre2.csv', 1, ventana=ventana, traslape=traslape)
#X3, y3 = cargar_dataset('intenso1.csv', 2, ventana=ventana, traslape=traslape)
#X32, y32 = cargar_dataset('intenso2.csv', 2, ventana=ventana, traslape=traslape)
#X33, y33 = cargar_dataset('intenso3.csv', 2, ventana=ventana, traslape=traslape)
#X34, y34 = cargar_dataset('intenso4.csv', 2, ventana=ventana, traslape=traslape)
#X35, y35 = cargar_dataset('intenso5.csv', 2, ventana=ventana, traslape=traslape)
#X4, y4 = cargar_dataset('ligero agarre.csv', 3, ventana=ventana, traslape=traslape)
#X42, y42 = cargar_dataset('ligero agarre2.csv', 3, ventana=ventana, traslape=traslape)

X1, y1 = cargar_dataset('relajacion.csv', 0, ventana=ventana, traslape=traslape)
X2, y2 = cargar_dataset('indice.csv', 1, ventana=ventana, traslape=traslape)
X3, y3 = cargar_dataset('contraccion.csv', 2, ventana=ventana, traslape=traslape)
X32, y32 = cargar_dataset('contraccion2.csv', 2, ventana=ventana, traslape=traslape)
X33, y33 = cargar_dataset('contraccion3.csv', 2, ventana=ventana, traslape=traslape)


# UNIR Y SINCRONIZAR DATOS
#X = X1 + X2 + X22 + X3 + X32 + X33 + X34 + X35 + X4 + X42
#y = y1 + y2 + y22 + y3 + y32 + y33 + y34 + y35 + y4 + y42
X = X1 + X2 + X3 + X32 + X33
y = y1 + y2 + y3 + y32 + y33
X, y = sincronizar_datos(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# RANDOM FOREST
rf_pipeline = Pipeline([
    ('clf', RandomForestClassifier(random_state=42))
])

rf_params = {
    'clf__n_estimators': randint(50, 200),
    'clf__max_depth': randint(3, 20),
    'clf__min_samples_split': randint(2, 10),
}

rf_search = RandomizedSearchCV(rf_pipeline, rf_params, n_iter=20, cv=5, random_state=42, n_jobs=-1)

# SVM
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC())
])

svm_params = {
    'clf__C': uniform(0.1, 10),
    'clf__gamma': ['scale', 'auto'],
    'clf__kernel': ['rbf', 'linear']
}

svm_search = RandomizedSearchCV(svm_pipeline, svm_params, n_iter=20, cv=5, random_state=42, n_jobs=-1)

# ANN (Red Neuronal Artificial)
ann_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', MLPClassifier(
        max_iter=2000,
        early_stopping=True,
        tol=1e-4,
        random_state=42))
])

ann_params = {
    'clf__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'clf__activation': ['relu', 'tanh'],
    'clf__solver': ['adam', 'sgd'],
    'clf__alpha': uniform(0.0001, 0.01),
    'clf__learning_rate': ['constant', 'adaptive']
}

ann_search = RandomizedSearchCV(ann_pipeline, ann_params, n_iter=20, cv=5, random_state=42, n_jobs=-1)

# ENTRENAMIENTO
print("Entrenando Random Forest...")
rf_search.fit(X_train, y_train)

print("Entrenando SVM...")
svm_search.fit(X_train, y_train)
print("Entrenando ANN...")
ann_search.fit(X_train, y_train)

# RESULTADOS RANDOM FOREST
print("\nMejor modelo Random Forest:")
print(rf_search.best_params_)
rf_pred = rf_search.predict(X_test)
print(confusion_matrix(y_test, rf_pred))
#print(classification_report(y_test, rf_pred, target_names=["Relajación", "Media", "Intensa", "Ligero"]))
print(classification_report(y_test, rf_pred, target_names=["Relajación", "Indice", "Cerrar"]))

# RESULTADOS SVM
print("\nMejor modelo SVM:")
print(svm_search.best_params_)
svm_pred = svm_search.predict(X_test)
print(confusion_matrix(y_test, svm_pred))
#print(classification_report(y_test, svm_pred, target_names=["Relajación", "Media", "Intensa", "Ligero"]))
print(classification_report(y_test, svm_pred, target_names=["Relajación", "Indice", "Cerrar"]))

# RESULTADOS ANN
print("\nMejor modelo ANN:")
print(ann_search.best_params_)
ann_pred = ann_search.predict(X_test)
print(confusion_matrix(y_test, ann_pred))
#print(classification_report(y_test, ann_pred, target_names=["Relajación", "Media", "Intensa", "Ligero"]))
print(classification_report(y_test, ann_pred, target_names=["Relajación", "Indice", "Cerrar"]))

# GUARDAR MODELOS
joblib.dump(rf_search.best_estimator_, "modelo_random_forest.pkl")
joblib.dump(svm_search.best_estimator_, "modelo_svm.pkl")
joblib.dump(ann_search.best_estimator_, "modelo_ann.pkl")

