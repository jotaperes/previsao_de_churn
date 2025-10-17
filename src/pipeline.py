# src/pipeline.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

def run_pipeline():
    """
    Executa o pipeline completo de carregamento de dados, pré-processamento,
    balanceamento (SMOTE), treinamento e avaliação dos modelos.
    """
    # Carregamento e Limpeza (sem alterações)
    print("Carregando e limpando os dados...")
    df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

    X = df.drop(['customerID', 'Churn'], axis=1)
    y = df['Churn']

    # Divisão em Treino e Teste (sem alterações)
    print("Dividindo os dados em treino e teste...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Definição do Pipeline de Pré-processamento (sem alterações)
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Treinamento e Avaliação dos Modelos
    models = {
        "Regressão Logística": LogisticRegression(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    }

    for name, model in models.items():
        print(f"\n--- Treinando e avaliando: {name} ---")
        
        # O pipeline inclui o SMOTE como uma etapa intermediária
        # Ele será aplicado SOMENTE nos dados de treino durante o .fit()
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('sampler', SMOTE(random_state=42)), # Adicionamos o SMOTE aqui
                                   ('classifier', model)])
        
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        
        print(f"Acurácia: {accuracy_score(y_test, y_pred):.2f}")
        print("Relatório de Classificação (com dados balanceados no treino):")
        print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    run_pipeline()