# Relatório Acadêmico - Análise Salarial v2.0

**Data de Geração:** 2025-06-13 18:57:37  
**Tempo Total de Processamento:** 52.1 segundos  
**Fonte de Dados:** CSV  

## 📊 Resumo Executivo

Este relatório apresenta os resultados de uma análise completa de dados salariais utilizando técnicas de Machine Learning, clustering e análise de regras de associação.

### Principais Resultados:
- **Dataset:** 32,561 registros processados
- **Modelos Treinados:** 2
- **Melhor Modelo:** Random Forest (Acurácia: 0.8632)
- **Clusters Identificados:** 3
- **Regras de Associação:** 62599

## 🔍 Metodologia

### 1. Processamento de Dados
- Carregamento de 32,561 registros
- Limpeza e tratamento de valores ausentes
- Criação de features derivadas (faixas etárias, intensidade de trabalho, etc.)

### 2. Análise Exploratória
- Estatísticas descritivas completas
- Visualizações de distribuições
- Análise de correlações

### 3. Modelagem de Machine Learning

#### Resultados dos Modelos:

| Modelo | Acurácia | Precisão | Recall | F1-Score |
|--------|----------|----------|--------|-----------|
| Random Forest | 0.8632 | 0.8592 | 0.8632 | 0.8532 |
| Logistic Regression | 0.8587 | 0.8531 | 0.8587 | 0.8539 |

### 4. Análise de Clustering
- **Número Ótimo de Clusters:** 3
- **Algoritmo:** K-Means
- **Silhouette Score:** 0.6063

### 5. Regras de Associação
- **Total de Regras Encontradas:** 62599
- **Algoritmo:** Apriori
- **Métricas:** Confiança e Lift

#### Top 3 Regras:
- SE workclass_Private, relationship_Own-child, education_Some-college, salary_<=50K, race_White → ENTÃO country_United-States, age_Jovem, edu_level_Médio, marital_Never-married (Conf: 0.807, Lift: 12.612)
- SE country_United-States, age_Jovem, edu_level_Médio, marital_Never-married → ENTÃO workclass_Private, relationship_Own-child, education_Some-college, salary_<=50K, race_White (Conf: 0.501, Lift: 12.612)
- SE relationship_Own-child, education_Some-college, workclass_Private, race_White → ENTÃO age_Jovem, country_United-States, salary_<=50K, edu_level_Médio, marital_Never-married (Conf: 0.801, Lift: 12.597)

## 📈 Conclusões

### Principais Insights:
1. **Performance de Modelos:** O modelo Random Forest apresentou a melhor performance com 0.8632 de acurácia
2. **Segmentação:** Identificados 3 grupos distintos na população
3. **Padrões:** 62599 regras de associação revelam relações interessantes

### Recomendações:
- Foco em features mais importantes para predição
- Análise detalhada dos clusters identificados
- Aplicação das regras de associação em estratégias de negócio

## 📊 Estatísticas Técnicas

- **Tempo de Processamento:** 52.1 segundos
- **Visualizações Criadas:** 7
- **Arquivos Gerados:** Modelos, preprocessadores, relatórios e gráficos
- **Fonte de Dados:** CSV

## 📁 Estrutura de Saídas

```
output/
├── images/                    # Visualizações
│   ├── summary_dashboard_v2.png
│   ├── model_comparison_v2.png
│   ├── clustering_analysis_v2.png
│   └── feature_importance_v2.png
├── analysis/                  # Análises
│   ├── relatorio_academico_v2.md
│   ├── advanced_metrics_v2.csv
│   ├── association_rules_v2.csv
│   └── clustering_results_v2.csv
└── logs/                      # Logs do processo
    └── projeto_salario_v2.log

data/processed/               # Modelos treinados
├── preprocessor_v2.joblib
├── random_forest_model_v2.joblib
├── logistic_regression_model_v2.joblib
└── target_encoder_v2.joblib
```

---
**Relatório gerado automaticamente pelo Pipeline Acadêmico v2.0**
