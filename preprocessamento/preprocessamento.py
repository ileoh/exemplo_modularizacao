from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocessar_dados(df):
    X = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )

    return X_train, X_test, y_train, y_test
