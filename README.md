# 📊 Sistema de Análise Salarial - Dashboard Multilingual Acadêmico

**Sistema Acadêmico Modular de Análise e Predição Salarial com Implementação Completa de DBSCAN, APRIORI, FP-GROWTH e ECLAT**

## 🎓 Sobre o Projeto Acadêmico

Este é um **sistema completo de análise salarial** desenvolvido para fins acadêmicos, implementando rigorosamente os algoritmos **DBSCAN**, **APRIORI**, **FP-GROWTH** e **ECLAT** conforme especificações científicas. O projeto demonstra a aplicação prática de técnicas avançadas de Data Science em cenários reais de análise salarial.

### 🏆 Algoritmos Principais Implementados

- 🎯 **DBSCAN** - Clustering baseado em densidade
- 📋 **APRIORI** - Regras de associação clássicas
- 🚀 **FP-GROWTH** - Mineração eficiente de padrões
- ⚡ **ECLAT** - Algoritmo de intersecção de conjuntos

## 🚀 Início Rápido

### **Método 1: Execução Automática (Recomendado)**

```bash
# Clonar repositório
https://github.com/DarioDourado/projecto_final.git
cd projecto_final


# Criar ambiente virtual (Windows)
python -m venv .venv

# Ativar ambiente virtual (Linux/macOS)
source .venv/bin/activate


# Ativar ambiente virtual (Windows)
.venv\Scripts\activate

# Instalar dependências
pip install -r requirements.txt

# Executar data diretamente
python main.py

# Executar dashboard diretamente
streamlit run app.py
```

### **Método 2: Pipeline Completo**

```bash
# 1. Configurar ambiente
python setup_scripts/setup_all.py

# 2. Executar pipeline acadêmico
python main.py

# 3. Iniciar dashboard multilingual
streamlit run app.py
```

### **Método 3: Deploy Streamlit Cloud**

1. Fork este repositório
2. Conectar no [Streamlit Community](https://streamlit.io/cloud)
3. Deploy automático (usa fallback CSV)
4. Acesso via URL gerada

## 🎯 Funcionalidades Acadêmicas

### ✅ **Algoritmos Científicos Implementados**

- **Machine Learning**: Random Forest, Logistic Regression
- **Clustering**: DBSCAN + K-Means com comparação
- **Association Rules**: APRIORI + FP-GROWTH + ECLAT
- **Métricas**: Accuracy, Precision, Recall, F1-Score, Silhouette

### ✅ **Sistema Completo**

- 🌍 **Dashboard Multilingual** (Português/English)
- 🔐 **Autenticação Robusta** (admin, user, guest)
- 💾 **Dual Storage** (MySQL + CSV fallback)
- 📊 **Visualizações Interativas** (Plotly)
- 🤖 **Predição em Tempo Real**
- 📈 **Métricas Avançadas** e relatórios
- 🔄 **Pipeline Reprodutível**

## 📁 Estrutura Acadêmica do Projeto

```
📁 projecto_final/
├── 📄 main.py                    # 🎓 Pipeline Acadêmico Principal
├── 📄 app.py                     # 🌐 Dashboard Multilingual
├── 📄 run_dashboard.py           # 🚀 Script de Inicialização
├── 📄 show_results.py            # 📊 Visualização de Resultados
├── 📄 requirements.txt           # 📋 Dependências Otimizadas
├── 📄 .env                       # ⚙️ Configurações (criar se necessário)
│
├── 📁 src/                       # 🧩 Código Modular Acadêmico
│   ├── 📁 analysis/              # 🎯 Algoritmos Científicos
│   │   ├── clustering.py         # DBSCAN + K-Means
│   │   └── association_rules.py  # APRIORI + FP-GROWTH + ECLAT
│   ├── 📁 pipelines/             # 🔄 Pipelines ML
│   ├── 📁 database/              # 💾 Integração SQL
│   ├── 📁 components/            # 🎨 UI Components
│   ├── 📁 auth/                  # 🔐 Sistema Autenticação
│   ├── 📁 pages/                 # 📄 Páginas Dashboard
│   ├── 📁 utils/                 # 🛠️ Utilitários (i18n, logging)
│   └── 📁 evaluation/            # 📊 Métricas Avançadas
│
├── 📁 bkp/                       # 💾 Dados e Backups
│   ├── 4-Carateristicas_salario.csv  # Dataset Original
│   ├── projeto_salario.py        # Versão Acadêmica v2.0
│   └── app_original_backup.py    # Backup Dashboard
│
├── 📁 data/                      # 📊 Dados Processados
│   ├── 📁 raw/                   # Dados brutos
│   └── 📁 processed/             # Dados limpos + modelos
│
├── 📁 output/                    # 📈 Resultados Científicos
│   ├── 📁 analysis/              # CSVs dos algoritmos
│   │   ├── dbscan_results.csv    # Resultados DBSCAN
│   │   ├── apriori_rules.csv     # Regras APRIORI
│   │   ├── fp_growth_rules.csv   # Regras FP-GROWTH
│   │   └── eclat_rules.csv       # Regras ECLAT
│   ├── 📁 images/                # Visualizações
│   └── 📁 logs/                  # Logs do sistema
│
├── 📁 translate/                 # 🌍 Suporte Multilingual
│   ├── pt.json                   # Português
│   └── en.json                   # English
│
├── 📁 config/                    # ⚙️ Configurações
├── 📁 setup_scripts/             # 🔧 Scripts de Configuração
├── 📁 tests/                     # 🧪 Testes Automatizados
└── 📄 Relatorio_rascunho.txt     # 📚 Relatório Acadêmico
```

## 🔑 Credenciais de Demonstração

### **Contas Pré-configuradas:**

- **👨‍💼 Admin**: `admin` / `admin123` (Acesso total)
- **👤 Demo**: `demo` / `demo123` (Usuário padrão)
- **🎭 Guest**: `guest` / `guest123` (Visitante)

## ⚙️ Configuração Avançada

### **💾 MySQL (Produção - Opcional):**

```bash
# 1. Criar arquivo .env
echo "DB_HOST=localhost
DB_NAME=salary_analysis
DB_USER=salary_user
DB_PASSWORD=senha_forte" > .env

# 2. Configurar estrutura
python main.py --setup-db

# 3. Migrar dados CSV → SQL
python main.py --migrate

# 4. Executar pipeline completo
python main.py
```

### **📄 CSV (Desenvolvimento/Streamlit Cloud):**

```bash
# Sistema funciona automaticamente com CSV
# Dados em: bkp/4-Carateristicas_salario.csv
streamlit run app.py
```

## 📊 Páginas do Dashboard Acadêmico

| Página                      | Acesso     | Algoritmos                             | Descrição                           |
| --------------------------- | ---------- | -------------------------------------- | ----------------------------------- |
| 📊 **Visão Geral**          | Todos      | -                                      | Dashboard principal com métricas    |
| 🔍 **Análise Exploratória** | User/Admin | Estatística                            | Visualizações e correlações         |
| 🤖 **Modelos ML**           | User/Admin | **Random Forest, Logistic Regression** | Treinamento e avaliação             |
| 🎯 **Clustering**           | User/Admin | **DBSCAN + K-Means**                   | Análise de agrupamentos             |
| 📋 **Regras de Associação** | User/Admin | **APRIORI + FP-GROWTH + ECLAT**        | Mineração de padrões                |
| 🔮 **Predição**             | User/Admin | ML Models                              | Interface de predição em tempo real |
| 📊 **Métricas Avançadas**   | User/Admin | Todos                                  | KPIs e dashboard científico         |
| 📁 **Relatórios**           | User/Admin | -                                      | Exportação e análises detalhadas    |
| ⚙️ **Administração**        | Admin      | -                                      | Gestão de usuários e sistema        |

## 🎓 Resultados Científicos Gerados

### **📈 Algoritmos Executados:**

- `output/analysis/dbscan_results.csv` - Clustering DBSCAN
- `output/analysis/apriori_rules.csv` - Regras APRIORI
- `output/analysis/fp_growth_rules.csv` - Regras FP-GROWTH
- `output/analysis/eclat_rules.csv` - Regras ECLAT
- `output/pipeline_results.json` - Resumo completo
- `output/relatorio_academico_completo.txt` - Relatório final

### **🔬 Métricas Implementadas:**

- **ML**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Clustering**: Silhouette Score, Inertia, N° Clusters
- **Association**: Confidence, Lift, Support
- **Performance**: Tempo execução, Memória utilizada

## 🌐 Suporte Multilingual

- 🇵🇹 **Português** (padrão acadêmico)
- 🇺🇸 **English** (internacional)

O sistema detecta automaticamente o idioma e permite troca dinâmica via interface.

## 🛠️ Resolução de Problemas

### **❌ Erro de Dependências:**

```bash
# Reinstalar dependências
pip install --upgrade -r requirements.txt

# Verificar instalação
python -c "import streamlit, pandas, sklearn; print('✅ OK')"
```

### **❌ Erro de Dados:**

```bash
# Verificar estrutura de dados
python diagnose.py

# Reprocessar dados
python main.py --force-csv

# Verificar saídas
ls output/analysis/
```

### **❌ Erro de Autenticação:**

```bash
# Resetar configurações
rm -rf config/

# Reiniciar aplicação
streamlit run app.py
```

### **❌ Erro MySQL:**

```bash
# Verificar serviço
brew services list | grep mysql  # macOS
sudo service mysql status        # Linux

# Usar fallback CSV
python main.py --force-csv
```

## 🔬 Validação Científica

### **📊 Reprodutibilidade:**

```bash
# Pipeline completo reprodutível
python main.py
# Verificar outputs idênticos em output/analysis/
```

### **🧪 Testes Automatizados:**

```bash
# Executar testes
python -m pytest tests/

# Verificar integridade
python diagnose.py
```

### **📈 Benchmarks:**

- **Accuracy Random Forest**: ~84.08%
- **Accuracy Logistic Regression**: ~81.85%
- **Tempo Pipeline Completo**: < 2 minutos
- **Regras de Associação**: 25-30 por algoritmo

## 🎯 Deploy Streamlit Community Cloud

O sistema foi **otimizado para Streamlit Community**:

1. **Fork** este repositório no GitHub
2. **Connect** no [Streamlit Community](https://streamlit.io/cloud)
3. **Deploy** automático (usa CSV fallback inteligente)
4. **Acesso** via URL pública gerada

**✅ Não requer configuração de banco de dados!**

## 📚 Referências Acadêmicas

O projeto implementa algoritmos baseados em:

- **DBSCAN**: Ester et al. (1996)
- **APRIORI**: Agrawal & Srikant (1994)
- **FP-GROWTH**: Han et al. (2000)
- **ECLAT**: Zaki (2000)

Metodologia fundamentada em literatura científica consolidada.

## 🏗️ Arquitetura Técnica

### **🔧 Tecnologias Principais:**

- **Backend**: Python 3.8+, Pandas, Scikit-learn
- **Frontend**: Streamlit, Plotly, HTML/CSS
- **Database**: MySQL 8.0+ (opcional), CSV fallback
- **ML**: Random Forest, Logistic Regression
- **Algorithms**: mlxtend (association rules), sklearn (clustering)

### **📦 Dependências Core:**

```txt
streamlit>=1.28.0
pandas>=1.5.3
scikit-learn>=1.0.0
plotly>=5.15.0
mlxtend>=0.22.0
mysql-connector-python>=8.1.0
python-dotenv>=1.0.0
```

## 📞 Suporte e Documentação

### **🐛 Para Problemas:**

1. Verificar logs: `logs/app.log`
2. Executar diagnóstico: `python diagnose.py`
3. Consultar documentação: `Relatorio_rascunho.txt`
4. Reiniciar sistema: `python run_dashboard.py`

### **📖 Documentação Adicional:**

- `docs/` - Documentação técnica completa
- `output/analysis/relatorio_academico_v2.md` - Relatório científico
- `bkp/projeto_salario.py` - Implementação acadêmica de referência

## 🎖️ Status do Projeto

- ✅ **Algoritmos Científicos**: 100% implementados
- ✅ **Dashboard Multilingual**: Funcional
- ✅ **Sistema de Autenticação**: Robusto
- ✅ **Pipeline Reprodutível**: Validado
- ✅ **Deploy Streamlit Cloud**: Otimizado
- ✅ **Documentação**: Completa

---

## 📄 Licença

Este projeto foi desenvolvido para fins **acadêmicos e educacionais**. Consulte o arquivo LICENSE para detalhes.

---

**💰 Sistema de Análise Salarial Acadêmico v6.2** - Implementação Completa dos Algoritmos DBSCAN, APRIORI, FP-GROWTH e ECLAT

**🎓 Projeto Acadêmico** | **📊 Data Science** | **🤖 Machine Learning** | **🌐 Dashboard Multilingual**

---

_Desenvolvido com rigor científico e metodologia acadêmica consolidada_
