# Projeto de Previsão de Churn - Telecom

## 1. Visão Geral

Este projeto tem como objetivo desenvolver um modelo de Machine Learning capaz de prever a probabilidade de um cliente cancelar seu contrato com uma empresa de telecomunicações (processo conhecido como *churn*). A identificação precoce de clientes com alto risco de churn permite que a empresa tome ações proativas para retê-los, como ofertas personalizadas e melhorias no serviço.

O dataset utilizado foi o "Telco Customer Churn", disponível publicamente no Kaggle.

## 2. Estrutura do Repositório

-   `/data`: Armazena o dataset original.
-   `/notebooks`: Contém o notebook Jupyter com a Análise Exploratória de Dados (AED).
-   `/src`: Contém o script Python com o pipeline final de pré-processamento, treinamento e avaliação do modelo.
-   `requirements.txt`: Lista das dependências do projeto.

## 3. Metodologia

O projeto seguiu as seguintes etapas:

1.  **Análise Exploratória de Dados (AED):** Investigação inicial dos dados para entender as distribuições e identificar fatores que influenciam o churn.
2.  **Pré-processamento e Feature Engineering:** Limpeza, tratamento de valores ausentes e transformação de variáveis.
3.  **Balanceamento de Dados (SMOTE):** Aplicação da técnica *Synthetic Minority Over-sampling Technique* **apenas nos dados de treino** para corrigir o desbalanceamento entre as classes (clientes que cancelam vs. que não cancelam).
4.  **Modelagem:** Treinamento de três modelos de classificação (Regressão Logística, Random Forest e XGBoost).
5.  **Avaliação:** Análise de métricas com foco especial no **Recall** para a classe "Churn = Yes", pois é crucial para o negócio identificar o máximo de clientes com risco de cancelamento.

## 4. Como Executar o Projeto

1.  Clone este repositório:
    ```bash
    git clone [https://github.com/SEU_USUARIO/projeto_churn_telecom.git](https://github.com/SEU_USUARIO/projeto_churn_telecom.git)
    cd projeto_churn_telecom
    ```

2.  Crie e ative um ambiente virtual (recomendado):
    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows: venv\Scripts\activate
    ```

3.  Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```

4.  Para executar o pipeline de treinamento e ver os resultados, rode o script:
    ```bash
    python src/pipeline.py
    ```

## 5. Resultados

Após a aplicação do SMOTE para balancear os dados de treinamento, os modelos se tornaram mais eficazes em sua principal tarefa: identificar clientes com risco de cancelamento, como evidenciado pela melhora significativa na métrica de **Recall**.

Um insight interessante foi o desempenho da **Regressão Logística**. Embora tenha a menor acurácia geral, o modelo alcançou o **Recall mais alto (0.78)**. Isso significa que ele foi o mais bem-sucedido em encontrar os clientes que de fato iriam cancelar, o que pode ser extremamente valioso para o negócio, apesar de potencialmente classificar incorretamente alguns clientes fiéis.

O **XGBoost** apresentou o melhor equilíbrio geral entre as métricas, com uma boa acurácia e um recall competitivo.

A tabela abaixo resume o desempenho dos modelos nos dados de teste:

| Métrica | Regressão Logística | Random Forest | XGBoost |
| :--- | :---: | :---: | :---: |
| **Acurácia** | 0.73 | 0.77 | **0.78** |
| **Precisão (Churn=Yes)** | 0.50 | 0.57 | **0.57** |
| **Recall (Churn=Yes)** | **0.78** | 0.57 | 0.62 |
| **F1-Score (Churn=Yes)**| 0.61 | 0.57 | 0.60 |

### Conclusão dos Resultados

A escolha do melhor modelo dependeria do objetivo de negócio:
* **Para máxima identificação de churn:** Se a prioridade é contatar o maior número possível de clientes que podem cancelar, mesmo com o custo de contatar alguns que não iriam, a **Regressão Logística** seria a melhor escolha devido ao seu altíssimo Recall.
* **Para um equilíbrio geral:** Se o objetivo é um modelo mais balanceado, que erre menos no geral, o **XGBoost** se mostra a opção mais robusta.

## 6. Próximos Passos

-   Testar outros algoritmos, como Redes Neurais.
-   Realizar um ajuste fino de hiperparâmetros para otimizar o modelo Random Forest.
-   Implementar o modelo como uma API para consumo em um sistema de CRM.