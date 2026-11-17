# Detecção de Fraudes em Transações de Cartão de Crédito com Machine Learning

## Sobre o Projeto

Este projeto tem como objetivo aplicar técnicas de Machine Learning para detectar fraudes em transações de cartão de crédito utilizando uma base de dados real disponibilizada pelo TensorFlow.

O projeto percorre todo o fluxo de análise de dados, desde a exploração e preparação da base até o treinamento, avaliação e interpretação dos modelos.

---

## O Problema

A principal dificuldade deste conjunto de dados é o forte desbalanceamento entre as classes.

### Distribuição encontrada

| Classe | Quantidade |
|----------|----------:|
| Normal (0) | 284.315 |
| Fraude (1) | 492 |

### Distribuição percentual

| Classe | Percentual |
|----------|----------:|
| Normal | 99,83% |
| Fraude | 0,17% |

Em cenários como este, a acurácia pode ser enganosa.

Um modelo que classificasse todas as transações como legítimas teria mais de 99% de acerto e, ainda assim, não detectaria nenhuma fraude.

Por esse motivo, a avaliação foi baseada principalmente nas métricas:

- Recall
- Precision
- F1-Score

da classe de fraude.

---

## Preparação dos Dados

As etapas realizadas foram:

- Carregamento da base diretamente pela URL oficial.
- Análise da distribuição das classes.
- Criação da variável `LogAmount` utilizando `np.log1p()`.
- Separação entre treino e teste utilizando `train_test_split`.
- Utilização de `stratify` para manter a proporção de fraudes.
- Padronização dos dados utilizando `StandardScaler`.
- Utilização de pesos balanceados para compensar o desbalanceamento das classes.

---

## Modelos Utilizados

Foram avaliados três algoritmos.

### Logistic Regression

Utilizada como modelo base devido à simplicidade e facilidade de interpretação.

### Random Forest

Algoritmo baseado em ensemble de árvores de decisão.

### XGBoost

Modelo de Gradient Boosting amplamente utilizado em problemas de classificação e detecção de fraude.

---

## Comparação dos Modelos

### Logistic Regression

| Métrica | Fraude |
|----------|----------:|
| Precision | 0.06 |
| Recall | 0.92 |
| F1-Score | 0.11 |

Apesar de identificar 92% das fraudes, o modelo produziu uma quantidade muito elevada de falsos positivos, resultando em uma precisão extremamente baixa.

---

### Random Forest

| Métrica | Fraude |
|----------|----------:|
| Precision | 0.93 |
| Recall | 0.82 |
| F1-Score | 0.87 |

O Random Forest apresentou o melhor equilíbrio entre detecção e confiabilidade das previsões, alcançando o maior F1-Score entre os modelos avaliados.

---

### XGBoost

| Métrica | Fraude |
|----------|----------:|
| Precision | 0.72 |
| Recall | 0.85 |
| F1-Score | 0.78 |

O XGBoost apresentou desempenho consistente, detectando mais fraudes que o Random Forest, porém com menor precisão.

---

## Limiar de Decisão

Além do limiar padrão de 0,50, foi realizado um teste com limiar de 0,30.

### XGBoost com Threshold 0.30

| Métrica | Fraude |
|----------|----------:|
| Precision | 0.03 |
| Recall | 0.93 |
| F1-Score | 0.05 |

A redução do limiar aumentou a capacidade de detecção de fraudes, elevando o recall para 93%.

Entretanto, o número de falsos positivos aumentou significativamente, reduzindo a precisão para apenas 3%.

Esse resultado demonstra o trade-off existente entre maximizar a detecção de fraudes e minimizar alertas incorretos.

---

## ROC AUC

Resultado obtido:

```text
ROC AUC = 0.9728
```

Esse resultado indica excelente capacidade de separação entre transações legítimas e fraudulentas.

---

## Importância das Variáveis

As variáveis mais importantes segundo o modelo foram:

| Variável | Importância |
|----------|----------:|
| V14 | 0.520 |
| V10 | 0.091 |
| V12 | 0.034 |
| V8 | 0.033 |
| V13 | 0.033 |
| V4 | 0.032 |
| V20 | 0.031 |
| V21 | 0.027 |
| V19 | 0.020 |
| V3 | 0.019 |

Observa-se que a variável V14 teve destaque significativo, contribuindo com mais de 50% da importância total do modelo.

---

## Explicabilidade com SHAP

Foi utilizada a biblioteca SHAP para interpretar o comportamento do modelo.

A análise demonstrou que:

- V14 foi a variável mais relevante para a classificação de fraudes.
- V4, V10 e V12 também exerceram grande influência nas previsões.
- O modelo não depende de uma única variável, mas da combinação de diversos fatores para tomar suas decisões.

A utilização do SHAP aumenta a transparência do modelo e facilita a interpretação das classificações produzidas.

---

## Melhoria Implementada

Além do fluxo apresentado pela especialista da DIO, foram adicionadas as seguintes melhorias:

- Inclusão do modelo XGBoost.
- Comparação entre três algoritmos distintos.
- Ajuste de limiar de decisão para análise de recall.
- Busca de hiperparâmetros utilizando GridSearchCV.
- Análise de explicabilidade utilizando SHAP.
- Avaliação baseada em métricas adequadas para problemas desbalanceados.

---

## Evidências

O notebook apresenta:

- Distribuição das classes.
- Distribuição percentual das classes.
- Logistic Regression Classification Report.
- Random Forest Classification Report.
- XGBoost Classification Report.
- XGBoost com Threshold 0.30.
- ROC Curve.
- Precision-Recall Curve.
- Importância das Variáveis.
- Gráfico SHAP de explicabilidade.

---

## Conclusão

O projeto demonstrou que a acurácia não é uma métrica adequada para problemas de fraude devido ao forte desbalanceamento do conjunto de dados.

A comparação entre os modelos mostrou que o Random Forest apresentou o melhor equilíbrio entre Recall, Precision e F1-Score, enquanto o XGBoost permitiu explorar estratégias para aumentar a detecção de fraudes através do ajuste do limiar de decisão.

A utilização de SHAP complementou a análise ao fornecer transparência sobre os fatores mais relevantes para as decisões do modelo.