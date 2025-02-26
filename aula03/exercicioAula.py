# Importando as bibliotecas necessárias
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt


# Importando o conjunto de dados Wine
dados_vinho = load_wine()  # Carrega os dados
caracteristicas = dados_vinho.data  # Features (X)
rotulos = dados_vinho.target  # Labels (y)

# Dividindo os dados em treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(
    caracteristicas, rotulos, test_size=0.3, random_state=42, stratify=rotulos
)

# Criando o escalonador
escalonador = StandardScaler()

# Ajustando e transformando os dados de treino
X_treino = escalonador.fit_transform(X_treino)

# Transformando os dados de teste (usando o mesmo escalonamento do treino)
X_teste = escalonador.transform(X_teste)

# Criando o modelo KNN
modelo_knn = KNeighborsClassifier(n_neighbors=5)  # Aqui, 'n_neighbors' é o número de vizinhos (K)

# Treinando o modelo com os dados de treino
modelo_knn.fit(X_treino, y_treino)

# Prevendo as classes para os dados de teste
y_pred = modelo_knn.predict(X_teste)

# Acurácia
acuracia = accuracy_score(y_teste, y_pred)
print(f'Acurácia: {acuracia:.2f}')

# Matriz de Confusão
matriz_confusao = confusion_matrix(y_teste, y_pred)
print('Matriz de Confusão:')
print(matriz_confusao)

# Precisão, Recall e F1-Score (para cada classe)
precisao = precision_score(y_teste, y_pred, average='weighted')
recall = recall_score(y_teste, y_pred, average='weighted')
f1 = f1_score(y_teste, y_pred, average='weighted')

print(f'Precisão: {precisao:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-Score: {f1:.2f}')

import matplotlib.pyplot as plt
import seaborn as sns

# Matriz de Confusão
matriz_confusao = confusion_matrix(y_teste, y_pred)

# Plotando a Matriz de Confusão
plt.figure(figsize=(8, 6))
sns.heatmap(matriz_confusao, annot=True, fmt="d", cmap="Blues", xticklabels=dados_vinho.target_names, yticklabels=dados_vinho.target_names)
plt.xlabel('Classes Preditas')
plt.ylabel('Classes Reais')
plt.title('Matriz de Confusão')
plt.show()


from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Calculando as previsões de probabilidade para cada classe
y_score = modelo_knn.predict_proba(X_teste)

# Binaria as classes para a abordagem "one-vs-rest"
y_teste_binario = label_binarize(y_teste, classes=np.unique(y_teste))

# Plotando a Curva ROC para cada classe
plt.figure(figsize=(10, 8))

# Cores para o gráfico (uma cor para cada classe)
colors = ['blue', 'red', 'green']

for i in range(len(np.unique(y_teste))):  # Iterar sobre as classes
    fpr, tpr, _ = roc_curve(y_teste_binario[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)  # Calculando AUC para a classe

    plt.plot(fpr, tpr, color=colors[i], lw=2, label=f'Classe {i} (AUC = {roc_auc:.2f})')

# Linha diagonal para referência
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)

# Títulos e rótulos
plt.title('Curva ROC para Cada Classe')
plt.xlabel('Taxa de Falsos Positivos (FPR)')
plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
plt.legend(loc='lower right')

# Exibindo o gráfico
plt.show()
