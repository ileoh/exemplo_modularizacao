import matplotlib.pyplot as plt
import seaborn as sns

def plotar_resultados(y_test, y_pred):
    plt.figure(figsize=(8,6))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.xlabel("Valor Real")
    plt.ylabel("Valor Previsto")
    plt.title("Valores Reais vs. Valores Previstos")
    plt.show()

def plotar_residuos(y_test, y_pred):
    residuos = y_test - y_pred
    plt.figure(figsize=(8,6))
    sns.histplot(residuos, kde=True)
    plt.title("Distribuição dos Resíduos")
    plt.xlabel("Resíduos")
    plt.ylabel("Frequência")
    plt.show()
