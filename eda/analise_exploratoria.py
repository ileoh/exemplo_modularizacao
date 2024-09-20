import matplotlib.pyplot as plt
import seaborn as sns

def realizar_eda(df):
    print("Cabeçalho do DataFrame:")
    print(df.head())

    print("\nEstatísticas descritivas:")
    print(df.describe())

    print("\nVerificando valores nulos:")
    print(df.isnull().sum())

    # Visualização dos dados
    df.hist(figsize=(12, 10))
    plt.tight_layout()
    plt.show()

    # Matriz de correlação
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Matriz de Correlação')
    plt.show()
