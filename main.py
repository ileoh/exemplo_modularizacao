from data.carregamento import carregar_dados
from eda.analise_exploratoria import realizar_eda
from preprocessamento.preprocessamento import preprocessar_dados
from modelos.regressao_linear import treinar_modelo
from utils.metricas import calcular_metricas
from utils.visualizacao import plotar_resultados, plotar_residuos

def main():
    # Carregamento dos dados
    df = carregar_dados()

    # Análise exploratória
    realizar_eda(df)

    # Pré-processamento
    X_train, X_test, y_train, y_test = preprocessar_dados(df)

    # Treinamento do modelo
    model = treinar_modelo(X_train, y_train)

    # Avaliação do modelo
    y_pred = model.predict(X_test)
    mse, mae, r2 = calcular_metricas(y_test, y_pred)

    print(f"\nMean Squared Error (MSE): {mse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R² Score: {r2:.2f}")

    # Visualização dos resultados
    plotar_resultados(y_test, y_pred)
    plotar_residuos(y_test, y_pred)

    # Conclusão
    print("\nConclusão:")
    print(f"O modelo de Regressão Linear apresentou um R² de {r2:.2f}, indicando que {r2*100:.2f}% da variância nos dados é explicada pelo modelo.")

if __name__ == "__main__":
    main()
