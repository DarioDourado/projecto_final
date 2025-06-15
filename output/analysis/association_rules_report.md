# 📋 Relatório de Regras de Associação

**Data:** 2025-06-14 20:01:39

## ⚙️ Parâmetros

- **Suporte mínimo:** 0.01
- **Confiança mínima:** 0.5
- **Lift mínimo:** 1.0

## 📊 Resumo Executivo

- **Transações processadas:** 32,561
- **Algoritmos executados:** 3
- **Total de regras geradas:** 71218
- **Melhor algoritmo:** Apriori

## 🏆 Comparação de Algoritmos

| Algoritmo | Regras | Confiança Média | Lift Médio | Regras Alta Qualidade |
|-----------|--------|-----------------|------------|----------------------|
| Apriori | 33114 | 0.790 | 1.841 | 16786 |
| FP-Growth | 4990 | 0.799 | 1.687 | 2613 |
| Eclat | 33114 | 0.790 | 1.841 | 16786 |

## 🔍 Top 5 Regras - Apriori

### 1. education= Bachelors → education-num_bin=education-num_high
- **Suporte:** 0.164
- **Confiança:** 1.000
- **Lift:** 3.565

### 2. education= HS-grad → education-num_bin=education-num_medium
- **Suporte:** 0.323
- **Confiança:** 1.000
- **Lift:** 1.567

### 3. education= 11th → education-num_bin=education-num_medium
- **Suporte:** 0.036
- **Confiança:** 1.000
- **Lift:** 1.567

### 4. education= Masters → education-num_bin=education-num_high
- **Suporte:** 0.053
- **Confiança:** 1.000
- **Lift:** 3.565

### 5. education= 9th → education-num_bin=education-num_low
- **Suporte:** 0.016
- **Confiança:** 1.000
- **Lift:** 12.310

## 🔍 Top 5 Regras - FP-Growth

### 1. education= Bachelors → education-num_bin=education-num_high
- **Suporte:** 0.164
- **Confiança:** 1.000
- **Lift:** 3.565

### 2. education= HS-grad → education-num_bin=education-num_medium
- **Suporte:** 0.323
- **Confiança:** 1.000
- **Lift:** 1.567

### 3. education= 11th → education-num_bin=education-num_medium
- **Suporte:** 0.036
- **Confiança:** 1.000
- **Lift:** 1.567

### 4. education= Masters → education-num_bin=education-num_high
- **Suporte:** 0.053
- **Confiança:** 1.000
- **Lift:** 3.565

### 5. education= 9th → education-num_bin=education-num_low
- **Suporte:** 0.016
- **Confiança:** 1.000
- **Lift:** 12.310

## 🔍 Top 5 Regras - Eclat

### 1. education= Bachelors → education-num_bin=education-num_high
- **Suporte:** 0.164
- **Confiança:** 1.000
- **Lift:** 3.565

### 2. education= Masters → education-num_bin=education-num_high
- **Suporte:** 0.053
- **Confiança:** 1.000
- **Lift:** 3.565

### 3. education= Assoc-acdm → education-num_bin=education-num_high
- **Suporte:** 0.033
- **Confiança:** 1.000
- **Lift:** 3.565

### 4. education= Doctorate → education-num_bin=education-num_high
- **Suporte:** 0.013
- **Confiança:** 1.000
- **Lift:** 3.565

### 5. education= Prof-school → education-num_bin=education-num_high
- **Suporte:** 0.018
- **Confiança:** 1.000
- **Lift:** 3.565

