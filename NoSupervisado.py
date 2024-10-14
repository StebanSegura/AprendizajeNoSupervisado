import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Crear dataset simulado
data = {
    'estacion': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B'],
    'hora_dia': [8, 12, 18, 6, 9, 15, 19, 22],
    'dia_semana': ['lunes', 'martes', 'miércoles', 'jueves', 'viernes', 'sábado', 'domingo', 'lunes'],
    'cantidad_pasajeros': [200, 340, 400, 120, 450, 390, 500, 220],
    'clima': ['soleado', 'lluvioso', 'soleado', 'nublado', 'nublado', 'lluvioso', 'soleado', 'soleado'],
    'tiempo_espera': [5, 10, 2, 15, 7, 12, 3, 5]
}

df = pd.DataFrame(data)

# Convertir las variables categóricas en numéricas
df = pd.get_dummies(df, columns=['dia_semana', 'clima', 'estacion'])

# Definir el número de clusters (grupos)
kmeans = KMeans(n_clusters=3, random_state=42)

# Ajustar el modelo usando las características del dataset
kmeans.fit(df)

# Asignar etiquetas de cluster a cada fila en el dataset
df['cluster'] = kmeans.labels_

# Visualizar los resultados
print(df[['cantidad_pasajeros', 'tiempo_espera', 'cluster']])
# Visualización de los clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='cantidad_pasajeros', y='tiempo_espera', hue='cluster', data=df, palette='viridis', s=100)
plt.title("Agrupamiento de estaciones basado en pasajeros y tiempo de espera")
plt.xlabel('Cantidad de Pasajeros')
plt.ylabel('Tiempo de Espera')
plt.show()
