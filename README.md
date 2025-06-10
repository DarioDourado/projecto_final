# 📊 Dashboard Interativo - Análise e Previsão Salarial

Um dashboard completo em **Streamlit** para análise exploratória de dados e previsão de salários usando algoritmos de Machine Learning. Este projeto implementa um sistema de autenticação, visualizações interativas e modelos preditivos para determinar se uma pessoa ganha mais ou menos de $50K por ano.

## 🎯 Características Principais

- **🔐 Sistema de Autenticação**: Login com diferentes níveis de acesso
- **📊 Análise Exploratória**: Visualizações interativas dos dados
- **🤖 Machine Learning**: Modelo Random Forest para previsões
- **🔮 Previsões**: Interface para previsões individuais e em lote
- **📈 Interpretabilidade**: Análise da importância das features
- **🎨 Interface Moderna**: Design responsivo com fundo transparente
- **🌍 Multi-idioma**: Interface em português com dados em inglês

## 🏗️ Estrutura do Projeto

```
📁 ProjectoFinal/
├── 📄 dashboard_app.py              # Dashboard principal (Streamlit)
├── 📄 projeto_salario.py            # Pipeline de ML e geração de modelos
├── 📄 projeto_salario_interpretabilidade.py  # Scripts de interpretabilidade
├── 📄 4-Carateristicas_salario.csv  # Dataset principal
├── 🤖 random_forest_model.joblib    # Modelo treinado
├── ⚙️ preprocessor.joblib           # Pipeline de pré-processamento
├── 📋 feature_info.joblib           # Informações das features
├── 📊 sample_data.joblib            # Dados de exemplo para SHAP
├── 📁 imagens/                      # Gráficos e visualizações
│   ├── hist_age.png
│   ├── correlacao.png
│   ├── feature_importance_rf.png
│   └── ...
├── 📁 Data/                         # Dados originais
├── 📁 docs/                         # Documentação
├── 📁 env/                          # Ambiente virtual
└── 📄 README.md                     # Este arquivo
```

## 🚀 Instalação e Configuração

### 1. Pré-requisitos

- Python 3.8+
- pip (gerenciador de pacotes)

### 2. Clonar/Baixar o Projeto

```bash
# Se usando Git
git clone <url-do-repositorio>
cd ProjectoFinal

# Ou baixar e extrair o arquivo ZIP
```

### 3. Criar Ambiente Virtual

```bash
# Criar ambiente virtual
python -m venv env

# Ativar ambiente virtual
# Windows:
env\Scripts\activate
# macOS/Linux:
source env/bin/activate
```

### 4. Instalar Dependências

```bash
# Instalar pacotes essenciais
pip install streamlit pandas numpy matplotlib seaborn scikit-learn joblib

# Instalar pacotes opcionais (recomendado)
pip install shap psutil mlxtend

# Ou usar requirements.txt se disponível
pip install -r requirements.txt
```

## 📋 Dependências Principais

| Pacote         | Versão  | Descrição                         |
| -------------- | ------- | --------------------------------- |
| `streamlit`    | ≥1.28.0 | Framework para dashboard web      |
| `pandas`       | ≥1.5.0  | Manipulação de dados              |
| `numpy`        | ≥1.24.0 | Computação numérica               |
| `scikit-learn` | ≥1.3.0  | Machine Learning                  |
| `matplotlib`   | ≥3.6.0  | Visualizações                     |
| `seaborn`      | ≥0.12.0 | Visualizações estatísticas        |
| `joblib`       | ≥1.3.0  | Serialização de modelos           |
| `shap`         | ≥0.42.0 | Interpretabilidade (opcional)     |
| `psutil`       | ≥5.9.0  | Informações do sistema (opcional) |

## 🏃‍♂️ Como Executar

### 1. Preparar os Dados e Modelos

```bash
# Executar pipeline de ML (primeira vez)
python projeto_salario.py
```

Este comando irá:

- ✅ Carregar e limpar os dados
- ✅ Treinar o modelo Random Forest
- ✅ Gerar visualizações
- ✅ Salvar modelos e preprocessadores
- ✅ Criar análises de interpretabilidade

### 2. Iniciar o Dashboard

```bash
# Iniciar aplicação Streamlit
streamlit run dashboard_app.py
```

O dashboard abrirá automaticamente no navegador em `http://localhost:8501`

### 3. Fazer Login

Use uma das contas disponíveis:

| Utilizador | Password    | Nível de Acesso                             |
| ---------- | ----------- | ------------------------------------------- |
| `admin`    | `admin123`  | 👑 **Administrador** - Acesso total         |
| `analista` | `dados2024` | 📊 **Analista** - Análises + Previsões      |
| `user`     | `user123`   | 👤 **Utilizador** - Funcionalidades básicas |
| `demo`     | `demo`      | 🔍 **Demo** - Apenas visualização           |

## 🎛️ Funcionalidades por Nível de Acesso

### 👑 Administrador (`admin`)

- ✅ Visualizar todos os dados
- ✅ Aceder a todas as visualizações
- ✅ Ver modelos e métricas
- ✅ Fazer previsões individuais e em lote
- ✅ Upload de ficheiros CSV
- ✅ Informações do sistema

### 📊 Analista (`analista`)

- ✅ Visualizar todos os dados
- ✅ Aceder a todas as visualizações
- ✅ Ver modelos e métricas
- ✅ Fazer previsões individuais e em lote
- ✅ Upload de ficheiros CSV
- ❌ Informações do sistema

### 👤 Utilizador (`user`)

- ✅ Visualizar dados básicos
- ✅ Aceder a visualizações
- ❌ Ver modelos e métricas
- ✅ Fazer previsões individuais
- ❌ Upload de ficheiros CSV
- ❌ Informações do sistema

### 🔍 Demo (`demo`)

- ✅ Visualizar dados básicos
- ✅ Aceder a visualizações
- ❌ Ver modelos e métricas
- ❌ Fazer previsões
- ❌ Upload de ficheiros CSV
- ❌ Informações do sistema

## 📊 Dataset e Variáveis

### Sobre o Dataset

- **Fonte**: Adult Census Income Dataset
- **Registos**: ~32,000 pessoas
- **Objetivo**: Prever se o salário anual é >$50K ou ≤$50K
- **Tipo**: Classificação binária

### Variáveis do Dataset

#### 🔢 Variáveis Numéricas

- **age**: Idade (17-90 anos)
- **fnlwgt**: Peso demográfico final
- **education-num**: Anos de educação (1-16)
- **capital-gain**: Ganhos de capital ($)
- **capital-loss**: Perdas de capital ($)
- **hours-per-week**: Horas trabalhadas por semana (1-99)

#### 📝 Variáveis Categóricas

- **workclass**: Tipo de empregador (Private, Self-emp, Gov, etc.)
- **education**: Nível educacional (HS-grad, Bachelors, Masters, etc.)
- **marital-status**: Estado civil (Married, Divorced, Single, etc.)
- **occupation**: Ocupação profissional (Tech-support, Sales, etc.)
- **relationship**: Relacionamento familiar (Husband, Wife, Child, etc.)
- **race**: Etnia (White, Black, Asian-Pac-Islander, etc.)
- **sex**: Sexo (Male, Female)
- **native-country**: País de origem

#### 🎯 Variável Alvo

- **salary**: ≤50K ou >50K (binária)

## 🤖 Modelos Implementados

### Random Forest (Principal)

- **Algoritmo**: Random Forest Classifier
- **Features**: Todas as variáveis disponíveis
- **Pré-processamento**: StandardScaler + OneHotEncoder
- **Métricas**: Accuracy, Precision, Recall, F1-Score
- **Interpretabilidade**: Feature importance + SHAP values

### Outros Modelos (Opcionais)

- Regressão Logística
- Gradient Boosting
- Support Vector Machine

## 📈 Visualizações Disponíveis

### 📊 Distribuições

- Histogramas das variáveis numéricas
- Gráficos de barras das variáveis categóricas
- Matriz de correlação

### 🔍 Análise de Modelos

- Importância das features (Random Forest)
- Coeficientes (Regressão Logística)
- SHAP values para interpretabilidade

### 🎯 Clustering

- K-Means clustering
- Visualização PCA 2D
- Análise de segmentação

## 🔮 Como Fazer Previsões

### Previsão Individual

1. Aceder à seção "🔮 Previsão com Novos Dados"
2. Preencher os campos do formulário
3. Clicar em "🎯 FAZER PREVISÃO"
4. Ver resultado e probabilidade

### Previsão em Lote (CSV)

1. Preparar ficheiro CSV com as colunas corretas
2. Fazer upload na seção correspondente
3. Clicar em "🎯 Fazer Previsões em Lote"
4. Baixar resultados em CSV

### Formato do CSV para Upload

```csv
age,workclass,fnlwgt,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country
39,State-gov,77516,Bachelors,13,Never-married,Adm-clerical,Not-in-family,White,Male,2174,0,40,United-States
50,Self-emp-not-inc,83311,Bachelors,13,Married-civ-spouse,Exec-managerial,Husband,White,Male,0,0,13,United-States
```

## 🗄️ Integração com Base de Dados

### Migração CSV → Base de Dados Relacional

O projeto suporta migração dos dados CSV para uma estrutura de base de dados normalizada para melhor performance e escalabilidade.

#### 🏗️ Estrutura da Base de Dados

```sql
-- Tabelas de Dimensão (Lookup Tables)
CREATE TABLE workclass (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(50) UNIQUE NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabela Principal (Fatos)
CREATE TABLE person (
    id INT PRIMARY KEY AUTO_INCREMENT,
    age INT NOT NULL CHECK (age BETWEEN 17 AND 100),
    workclass_id INT,
    education_id INT,
    -- ... outras colunas
    FOREIGN KEY (workclass_id) REFERENCES workclass(id)
);
```

#### 📦 Dependências Adicionais

```bash
# MySQL
pip install mysql-connector-python

# PostgreSQL (alternativa)
pip install psycopg2-binary
```

#### ⚙️ Configuração da Base de Dados

1. **Criar Base de Dados MySQL**:

```sql
CREATE DATABASE salary_analysis CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'salary_user'@'localhost' IDENTIFIED BY 'senha_forte';
GRANT ALL PRIVILEGES ON salary_analysis.* TO 'salary_user'@'localhost';
```

2. **Configurar Variáveis de Ambiente**:

```bash
export DB_HOST=localhost
export DB_NAME=salary_analysis
export DB_USER=salary_user
export DB_PASSWORD=senha_forte
export USE_DATABASE=true
```

#### 🚀 Executar com Base de Dados

```bash
# Migrar dados CSV para base de dados
python main.py --migrate

# Usar base de dados como fonte
python main.py --database

# Usar arquivo CSV (padrão)
python main.py --csv

# Dashboard com base de dados
USE_DATABASE=true streamlit run app.py
```

#### ✅ Vantagens da Base de Dados

| Aspecto            | CSV                         | Base de Dados              |
| ------------------ | --------------------------- | -------------------------- |
| **Performance**    | Lenta para grandes datasets | Otimizada com índices      |
| **Integridade**    | Sem validação               | Foreign keys + constraints |
| **Concorrência**   | Limitada                    | Suporte completo           |
| **Consultas**      | Pandas limitado             | SQL completo               |
| **Escalabilidade** | Memória limitante           | Escalável                  |

#### 📊 Exemplos de Consultas SQL

```sql
-- Análise de salário por educação
SELECT
    e.name as education_level,
    sr.name as salary_range,
    COUNT(*) as count
FROM person p
JOIN education e ON p.education_id = e.id
JOIN salary_range sr ON p.salary_range_id = sr.id
GROUP BY e.name, sr.name;
```

#### 🔧 Comandos Úteis

```bash
# Verificar configuração
python -c "from src.database.connection import DatabaseConnection; print('✅ Configuração OK' if DatabaseConnection().connect() else '❌ Erro')"

# Migração completa
python main.py --migrate --database

# Estatísticas da base de dados
python -c "from src.database.models import SalaryDataModel; print(SalaryDataModel().get_statistics())"
```

## 🛠️ Scripts Principais

### `projeto_salario.py`

Pipeline principal que:

- Carrega e limpa dados
- Aplica tipagem otimizada
- Treina modelos de ML
- Gera visualizações
- Salva artefactos

### `dashboard_app.py`

Aplicação Streamlit que:

- Implementa autenticação
- Mostra visualizações interativas
- Permite previsões
- Gere diferentes níveis de acesso

### `projeto_salario_interpretabilidade.py`

Script adicional para:

- Análise SHAP detalhada
- Interpretabilidade avançada
- Gráficos de explicação

## 🔧 Resolução de Problemas

### Erro: "Dados não encontrados"

```bash
# Solução: Executar pipeline primeiro
python projeto_salario.py
```

### Erro: "Modelo não encontrado"

```bash
# Verificar se os arquivos foram gerados
ls *.joblib
# Deve mostrar: random_forest_model.joblib, preprocessor.joblib, etc.
```

### Erro: "SHAP não disponível"

```bash
# Instalar SHAP (opcional)
pip install shap
```

### Problemas de Memória

```bash
# Instalar psutil para monitorização
pip install psutil
```

### Pasta 'imagens' não encontrada

```bash
# As imagens são geradas automaticamente pelo pipeline
python projeto_salario.py
```

## 📝 Logs e Debugging

### Ativar Logs Detalhados

```bash
# Executar com logs verbosos
streamlit run dashboard_app.py --logger.level=debug
```

### Verificar Status do Sistema

- Use a conta `admin` para aceder às "Informações do Sistema"
- Verifique o status de todos os componentes
- Monitor de memória e performance

## 🔄 Atualizações e Manutenção

### Retraining do Modelo

```bash
# Retreinar com novos dados
python projeto_salario.py
```

### Backup dos Modelos

```bash
# Criar backup dos artefactos importantes
cp *.joblib backup/
cp -r imagens/ backup/
```

### Limpar Cache

```bash
# Limpar cache do Streamlit
streamlit cache clear
```

## 📚 Documentação Adicional

### Recursos Técnicos

- **Streamlit**: [docs.streamlit.io](https://docs.streamlit.io)
- **Scikit-learn**: [scikit-learn.org](https://scikit-learn.org)
- **SHAP**: [shap.readthedocs.io](https://shap.readthedocs.io)

### Tutoriais Relacionados

- **Machine Learning**: [Documentação Scikit-learn](https://scikit-learn.org/stable/tutorial/index.html)
- **Dashboard Streamlit**: [30 days of Streamlit](https://30days.streamlit.app)

## 🤝 Contribuições

### Como Contribuir

1. Fork do projeto
2. Criar branch para feature (`git checkout -b feature/AmazingFeature`)
3. Commit das mudanças (`git commit -m 'Add AmazingFeature'`)
4. Push para branch (`git push origin feature/AmazingFeature`)
5. Abrir Pull Request

### Áreas para Melhoria

- [ ] Novos algoritmos de ML
- [ ] Mais visualizações interativas
- [ ] Integração com bases de dados
- [ ] Testes automatizados
- [ ] Deploy em cloud

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## 📞 Contacto

**Desenvolvedor**: [Seu Nome]
**Email**: [seu.email@example.com]
**LinkedIn**: [Seu LinkedIn]

---

### 🚀 Versão: 1.0.0

### 📅 Última Atualização: Dezembro 2024

---

**⭐ Se este projeto foi útil, considere dar uma estrela no repositório!**# projecto_final
