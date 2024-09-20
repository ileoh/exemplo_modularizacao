import pandas as pd
from sklearn.datasets import fetch_california_housing

def carregar_dados():
    california = fetch_california_housing()
    df = pd.DataFrame(california.data, columns=california.feature_names)
    df['MedHouseVal'] = california.target
    return df
