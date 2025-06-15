# 📊 Sistema de Análise Salarial - Dashboard Multilingual

**Sistema Académico Modular de Análise e Predição Salarial com Suporte SQL/CSV**

## 🚀 Início Rápido

### **Método 1: Script Automático**
```bash
# Executar dashboard diretamente
python run_dashboard.py
```

### **Método 2: Manual**
```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Configurar banco (opcional)
python main.py --setup-db

# 3. Executar pipeline
python main.py

# 4. Iniciar dashboard
streamlit run app.py
```

## 🌍 Funcionalidades

- ✅ **Dashboard Multilingual** (Português/English)
- ✅ **Sistema de Autenticação** completo
- ✅ **Análise SQL + CSV** (fallback automático)
- ✅ **Machine Learning** avançado
- ✅ **Visualizações Interativas** (Plotly)
- ✅ **Clustering e Regras de Associação**
- ✅ **Sistema de Predição** em tempo real
- ✅ **Métricas Avançadas** e relatórios

## 📁 Estrutura do Projeto

```
📁 projecto_final/
├── 📄 app.py                     # ✅ Dashboard Principal (Multilingual)
├── 📄 main.py                    # Pipeline SQL/CSV
├── 📄 run_dashboard.py           # Script de inicialização
├── 📄 .env                       # Configurações (criar se necessário)
├── 📁 src/                       # Código modular
│   ├── 📁 auth/                  # Sistema de autenticação
│   ├── 📁 components/            # Componentes UI
│   ├── 📁 data/                  # Gestão de dados
│   ├── 📁 pages/                 # Páginas do dashboard
│   ├── 📁 utils/                 # Utilitários (i18n, etc)
│   └── 📁 database/              # Integração SQL
├── 📁 translate/                 # Traduções (pt.json, en.json)
├── 📁 config/                    # Configurações e usuários
├── 📁 data/                      # Dados CSV e processados
└── 📁 output/                    # Resultados e visualizações
```

## 🔑 Credenciais Padrão

### **Contas de Demonstração:**
- **Admin**: `admin` / `admin123`
- **Usuário**: `demo` / `demo123`  
- **Visitante**: `guest` / `guest123`

## ⚙️ Configuração Avançada

### **Banco de Dados MySQL (Opcional):**
```bash
# Configurar variáveis de ambiente
export DB_HOST=localhost
export DB_NAME=salary_analysis
export DB_USER=salary_user
export DB_PASSWORD=senha_forte

# Criar estrutura do banco
python main.py --setup-db

# Migrar dados CSV → SQL
python main.py --migrate
```

### **Só CSV (Streamlit Community):**
```bash
# Sistema funciona automaticamente com CSV
# Não requer configuração de banco
streamlit run app.py
```

## 📊 Páginas Disponíveis

| Página | Acesso | Descrição |
|--------|--------|-----------|
| 📊 **Visão Geral** | Todos | Dashboard principal com métricas |
| 🔍 **Análise Exploratória** | User/Admin | Visualizações e correlações |
| 🤖 **Modelos ML** | User/Admin | Treinamento e avaliação |
| 🔮 **Predição** | User/Admin | Interface de predição |
| 🎯 **Clustering** | User/Admin | Análise de agrupamentos |
| 📋 **Regras de Associação** | User/Admin | Padrões comportamentais |
| 📊 **Métricas Avançadas** | User/Admin | KPIs e dashboard completo |
| 📁 **Relatórios** | User/Admin | Exportação e análises |
| ⚙️ **Administração** | Admin | Gestão de usuários e sistema |

## 🌐 Suporte de Idiomas

- 🇵🇹 **Português** (padrão)
- 🇺🇸 **English**

O sistema detecta automaticamente e permite troca dinâmica de idiomas.

## 🛠️ Resolução de Problemas

### **Erro de Dependências:**
```bash
pip install --upgrade streamlit pandas plotly scikit-learn mysql-connector-python
```

### **Erro de Autenticação:**
```bash
# Apagar configurações
rm -rf config/
# Reiniciar aplicação
streamlit run app.py
```

### **Erro de Dados:**
```bash
# Verificar estrutura
python main.py --setup-db
# Reprocessar dados  
python main.py
```

## 🎯 Deploy Streamlit Community

O sistema foi otimizado para **Streamlit Community Cloud**:

1. **Fork** este repositório
2. **Connect** no Streamlit Community
3. **Deploy** automático (usa CSV fallback)
4. **Acesso** via URL gerada

**Não requer configuração de banco de dados!**

---

## 📞 Suporte

Para problemas ou dúvidas:
1. Verificar logs em `logs/app.log`
2. Executar diagnóstico: `python diagnose.py`
3. Reiniciar sistema: `python run_dashboard.py`

---

**💰 Sistema de Análise Salarial v6.1** - Dashboard Multilingual Modular

---

**4.13. Limitações Metodológicas e Threats to Validity**

#### 4.13.1. Limitações dos Dados
- **Temporal:** Dataset representa apenas um momento no tempo
- **Geográfica:** Possível concentração em determinadas regiões
- **Sectorial:** Pode não representar todos os sectores económicos

#### 4.13.2. Limitações Algorítmicas
- **Class Imbalance:** 75.9% vs 24.1% pode enviesar modelos
- **Feature Selection:** Ausência de variáveis contextuais importantes
- **Validation:** Cross-validation pode não captar todos os patterns

#### 4.13.3. Limitações de Generalização
- **External Validity:** Resultados podem não generalizar para outros países
- **Temporal Validity:** Padrões podem mudar com transformações económicas
