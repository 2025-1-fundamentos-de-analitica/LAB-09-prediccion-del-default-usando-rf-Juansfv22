# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

# --- Importación de librerías ---
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix
import pickle
import gzip
import os
import json

# --- Carga de datos comprimidos ---
zip_train_path = 'files/input/train_data.csv.zip'
zip_test_path = 'files/input/test_data.csv.zip'

df_train = pd.read_csv(zip_train_path, compression='zip', index_col=False)
df_test = pd.read_csv(zip_test_path, compression='zip', index_col=False)

# --- Renombrar columna objetivo ---
df_train.rename(columns={'default payment next month': 'default'}, inplace=True)
df_test.rename(columns={'default payment next month': 'default'}, inplace=True)

# --- Limpieza básica ---
df_train.drop(columns='ID', inplace=True)
df_test.drop(columns='ID', inplace=True)

df_train['EDUCATION'].replace(0, np.nan, inplace=True)
df_test['EDUCATION'].replace(0, np.nan, inplace=True)

df_train['MARRIAGE'].replace(0, np.nan, inplace=True)
df_test['MARRIAGE'].replace(0, np.nan, inplace=True)

df_train.dropna(inplace=True)
df_test.dropna(inplace=True)

df_train.loc[df_train['EDUCATION'] > 4, 'EDUCATION'] = 4
df_test.loc[df_test['EDUCATION'] > 4, 'EDUCATION'] = 4

# --- Conversión de variables categóricas ---
categorical_vars = ['EDUCATION', 'SEX', 'MARRIAGE']
for feature in categorical_vars:
    df_train[feature] = df_train[feature].astype('category')
    df_test[feature] = df_test[feature].astype('category')

# --- Separación de variables ---
X_train = df_train.drop(columns='default')
y_train = df_train['default']

X_test = df_test.drop(columns='default')
y_test = df_test['default']

X_train_cat = X_train.select_dtypes(include='category')
X_train_num = X_train.select_dtypes(exclude='category')

# --- Construcción del pipeline ---
cat_pipe = Pipeline([
    ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

num_pipe = Pipeline([
    ('std_scaler', StandardScaler())
])

column_prep = ColumnTransformer([
    ('num_features', num_pipe, X_train_num.columns),
    ('cat_features', cat_pipe, X_train_cat.columns)
])

full_pipeline = Pipeline([
    ('prep', column_prep),
    ('clf', RandomForestClassifier())
])

# --- GridSearch con validación cruzada ---
search_params = {
    'clf__n_estimators': [50, 100, 150],
    'clf__max_depth': [5, 10, 15]
}

grid = GridSearchCV(full_pipeline, param_grid=search_params, cv=10, scoring='balanced_accuracy')
grid.fit(X_train, y_train)

# --- Guardar modelo en disco ---
os.makedirs('files/models', exist_ok=True)
with gzip.open('files/models/model.pkl.gz', 'wb') as output_model:
    pickle.dump(grid, output_model)


y_pred_train = grid.predict(X_train)
y_pred_test = grid.predict(X_test)

train_metrics = {
    'type': 'metrics',
    'dataset': 'train',
    'precision': precision_score(y_train, y_pred_train),
    'balanced_accuracy': balanced_accuracy_score(y_train, y_pred_train),
    'recall': recall_score(y_train, y_pred_train),
    'f1_score': f1_score(y_train, y_pred_train)
}

test_metrics = {
    'type': 'metrics',
    'dataset': 'test',
    'precision': precision_score(y_test, y_pred_test),
    'balanced_accuracy': balanced_accuracy_score(y_test, y_pred_test),
    'recall': recall_score(y_test, y_pred_test),
    'f1_score': f1_score(y_test, y_pred_test)
}

# --- Matrices de confusión ---
train_cm = confusion_matrix(y_train, y_pred_train)
test_cm = confusion_matrix(y_test, y_pred_test)

train_cm_dict = {
    'type': 'cm_matrix',
    'dataset': 'train',
    'true_0': {"predicted_0": train_cm[0, 0], "predicted_1": train_cm[0, 1]},
    'true_1': {"predicted_0": train_cm[1, 0], "predicted_1": train_cm[1, 1]}
}

test_cm_dict = {
    'type': 'cm_matrix',
    'dataset': 'test',
    'true_0': {"predicted_0": test_cm[0, 0], "predicted_1": test_cm[0, 1]},
    'true_1': {"predicted_0": test_cm[1, 0], "predicted_1": test_cm[1, 1]}
}

# --- Guardar métricas en archivo JSON ---
all_metrics = [train_metrics, test_metrics, train_cm_dict, test_cm_dict]

os.makedirs('files/output', exist_ok=True)
with open('files/output/metrics.json', 'w') as metrics_file:
    for metric in all_metrics:
        metrics_file.write(json.dumps(metric) + '\n')