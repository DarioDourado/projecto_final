# projeto_salario.py

# ================================================
# 1. IMPORTAÇÃO DE BIBLIOTECAS
# ================================================
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# Configuração global para fundo transparente
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none'

# ================================================
# 2. CARREGAMENTO E LIMPEZA DOS DADOS
# ================================================
df = pd.read_csv('4-Carateristicas_salario.csv')
df = df.drop_duplicates()

print("\nResumo dos dados:")
print(df.info())
print("\nDescrição estatística:")
print(df.describe())
print("\nValores ausentes por coluna:")
print(df.isnull().sum())

# ================================================
# 2.1. LIMPEZA E TIPAGEM DOS DADOS
# ================================================
print("\n" + "="*60)
print("LIMPEZA E TIPAGEM DOS DADOS")
print("="*60)

def limpar_e_tipar_dados(df):
    """Limpar dados e aplicar tipagem correta às colunas"""
    
    # Fazer cópia para não alterar o original
    df_clean = df.copy()
    
    # Limpar espaços em branco em todas as colunas de texto
    for col in df_clean.select_dtypes(include=['object']).columns:
        df_clean[col] = df_clean[col].astype(str).str.strip()
    
    # Tratar valores '?' como NaN
    df_clean = df_clean.replace('?', pd.NA)
    
    print("🔧 Aplicando tipagem correta às colunas...")
    
    # ===== TIPAGEM DAS VARIÁVEIS NUMÉRICAS =====
    numerical_columns_types = {
        'age': 'int16',           # Idade: 17-90 (int16 suficiente)
        'fnlwgt': 'int32',        # Peso final: pode ser grande (int32)
        'education-num': 'int8',  # Anos educação: 1-16 (int8 suficiente)
        'capital-gain': 'int32',  # Ganho capital: pode ser grande
        'capital-loss': 'int16',  # Perda capital: menor range
        'hours-per-week': 'int8'  # Horas semana: 1-99 (int8 suficiente)
    }
    
    for col, dtype in numerical_columns_types.items():
        if col in df_clean.columns:
            try:
                # Converter para numérico primeiro, depois para o tipo específico
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                df_clean[col] = df_clean[col].astype(dtype)
                print(f"✅ {col}: convertido para {dtype}")
            except Exception as e:
                print(f"⚠️ Erro ao converter {col}: {e}")
    
    # ===== TIPAGEM DAS VARIÁVEIS CATEGÓRICAS =====
    categorical_columns = [
        'workclass', 'education', 'marital-status', 'occupation',
        'relationship', 'race', 'sex', 'native-country', 'salary'
    ]
    
    for col in categorical_columns:
        if col in df_clean.columns:
            try:
                # Converter para categoria para economizar memória
                df_clean[col] = df_clean[col].astype('category')
                print(f"✅ {col}: convertido para category")
            except Exception as e:
                print(f"⚠️ Erro ao converter {col}: {e}")
    
    # ===== VALIDAÇÃO DE RANGES =====
    print("\n🔍 Validando ranges das variáveis...")
    
    # Validar idade
    if 'age' in df_clean.columns:
        invalid_age = (df_clean['age'] < 17) | (df_clean['age'] > 100)
        if invalid_age.any():
            print(f"⚠️ Encontradas {invalid_age.sum()} idades inválidas (fora de 17-100)")
            df_clean.loc[invalid_age, 'age'] = pd.NA
    
    # Validar anos de educação
    if 'education-num' in df_clean.columns:
        invalid_edu = (df_clean['education-num'] < 1) | (df_clean['education-num'] > 16)
        if invalid_edu.any():
            print(f"⚠️ Encontrados {invalid_edu.sum()} anos de educação inválidos (fora de 1-16)")
            df_clean.loc[invalid_edu, 'education-num'] = pd.NA
    
    # Validar horas por semana
    if 'hours-per-week' in df_clean.columns:
        invalid_hours = (df_clean['hours-per-week'] < 1) | (df_clean['hours-per-week'] > 99)
        if invalid_hours.any():
            print(f"⚠️ Encontradas {invalid_hours.sum()} horas/semana inválidas (fora de 1-99)")
            df_clean.loc[invalid_hours, 'hours-per-week'] = pd.NA
    
    # Validar ganhos/perdas de capital (não podem ser negativos)
    for col in ['capital-gain', 'capital-loss']:
        if col in df_clean.columns:
            invalid_capital = df_clean[col] < 0
            if invalid_capital.any():
                print(f"⚠️ Encontrados {invalid_capital.sum()} valores negativos em {col}")
                df_clean.loc[invalid_capital, col] = 0
    
    return df_clean

def get_memory_usage(df):
    """Calcular uso de memória do DataFrame"""
    memory_usage = df.memory_usage(deep=True).sum()
    return memory_usage / 1024 / 1024  # MB

# Mostrar uso de memória antes da otimização
memory_before = get_memory_usage(df)
print(f"💾 Uso de memória antes da otimização: {memory_before:.2f} MB")

# Aplicar limpeza e tipagem
df = limpar_e_tipar_dados(df)

# Mostrar uso de memória após otimização
memory_after = get_memory_usage(df)
print(f"💾 Uso de memória após otimização: {memory_after:.2f} MB")
print(f"📉 Redução: {((memory_before - memory_after) / memory_before * 100):.1f}%")

print("\n✅ Dados limpos e tipados com sucesso!")
print("\nInfo dos tipos de dados:")
print(df.dtypes)

# Verificar valores ausentes após limpeza
missing_values = df.isnull().sum()
if missing_values.any():
    print(f"\n⚠️ Valores ausentes após limpeza:")
    print(missing_values[missing_values > 0])

# ================================================
# 3. ANÁLISE EXPLORATÓRIA DE DADOS (EDA)
# ================================================
# Criar diretório para imagens se não existir
os.makedirs("imagens", exist_ok=True)

# Distribuições das variáveis numéricas
numerical_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
for col in numerical_cols:
    plt.figure(figsize=(6, 4))
    # Configurar fundo transparente
    plt.gca().patch.set_alpha(0.0)
    plt.gcf().patch.set_alpha(0.0)
    
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'Distribuição de {col}')
    plt.tight_layout()
    plt.savefig(f'imagens/hist_{col}.png', transparent=True, bbox_inches='tight')
    plt.close()

# Matriz de correlação
corr = df[numerical_cols].corr()
plt.figure(figsize=(8, 6))
# Configurar fundo transparente
plt.gca().patch.set_alpha(0.0)
plt.gcf().patch.set_alpha(0.0)

sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Matriz de Correlação entre Variáveis Numéricas")
plt.tight_layout()
plt.savefig('imagens/correlacao.png', transparent=True, bbox_inches='tight')
plt.close()

# Distribuição de salário por variáveis categóricas
categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 
                    'relationship', 'race', 'sex', 'native-country']
for col in categorical_cols:
    plt.figure(figsize=(10, 5))
    # Configurar fundo transparente
    plt.gca().patch.set_alpha(0.0)
    plt.gcf().patch.set_alpha(0.0)
    
    sns.countplot(data=df, x=col, hue='salary')
    plt.title(f'Distribuição de Salário por {col}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'imagens/bar_{col}.png', transparent=True, bbox_inches='tight')
    plt.close()

print("\nGráficos salvos na pasta 'imagens'")

# ================================================
# 4. PRÉ-PROCESSAMENTO
# ================================================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Separar variáveis numéricas e categóricas
numerical_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
categorical_features = ['workclass', 'education', 'marital-status', 'occupation',
                        'relationship', 'race', 'sex', 'native-country']

# DIAGNÓSTICO DETALHADO DOS DADOS
print("\n" + "="*60)
print("DIAGNÓSTICO DETALHADO DOS DADOS")
print("="*60)

# Verificar distribuição original antes da codificação
print("\nDistribuição original de salary:")
print(df['salary'].value_counts())
print("Valores únicos originais:", df['salary'].unique())

# Verificar se há espaços ou caracteres especiais
print("\nAnálise detalhada dos valores de salary:")
for valor in df['salary'].unique():
    print(f"'{valor}' (tipo: {type(valor)}, comprimento: {len(str(valor))})")

# Limpar valores de salary primeiro
df['salary'] = df['salary'].astype(str).str.strip()
print("\nApós limpeza de espaços:")
print(df['salary'].value_counts())

# Codificar variável-alvo com verificação mais robusta
def codificar_salary(valor):
    valor = str(valor).strip().lower()
    if '>50k' in valor or valor == '>50k':
        return 1
    elif '<=50k' in valor or valor == '<=50k':
        return 0
    else:
        print(f"⚠️ Valor não reconhecido: '{valor}'")
        return 0

df['salary'] = df['salary'].apply(codificar_salary)

# Verificar distribuição após codificação
print("\nDistribuição após codificação:")
print(df['salary'].value_counts())
print("Percentual de cada classe:")
print(df['salary'].value_counts(normalize=True) * 100)

# Verificar se há ambas as classes
if df['salary'].nunique() < 2:
    print("⚠️ ERRO: Apenas uma classe encontrada nos dados!")
    print("Valores únicos em salary:", df['salary'].unique())
    
    # Investigar mais a fundo
    df_original = pd.read_csv('4-Carateristicas_salario.csv')
    print("\nValores únicos no arquivo original:")
    for valor in df_original['salary'].unique():
        print(f"'{valor}' (aparece {(df_original['salary'] == valor).sum()} vezes)")
    
    # Tentar uma codificação diferente
    print("\nTentando codificação alternativa...")
    df['salary'] = df_original['salary'].apply(lambda x: 1 if '>' in str(x) else 0)
    print("Nova distribuição:")
    print(df['salary'].value_counts())

else:
    print("✅ Ambas as classes presentes nos dados")

# Separar X e y
X = df[numerical_features + categorical_features]
y = df['salary']

# Verificar valores ausentes
print(f"\nValores ausentes em X: {X.isnull().sum().sum()}")
print(f"Valores ausentes em y: {y.isnull().sum()}")

# Tratar valores '?' como NaN primeiro
X = X.replace('?', pd.NA)
print(f"Valores ausentes após tratar '?': {X.isnull().sum().sum()}")

# Remover registros com valores ausentes se necessário
if X.isnull().sum().sum() > 0:
    print("Removendo registros com valores ausentes...")
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]
    print(f"Registros restantes: {len(X)}")

# Verificar distribuição final antes da divisão
print(f"\nDistribuição final de y antes da divisão:")
print(y.value_counts())
print("Classes únicas:", y.unique())

# PARAR SE NÃO HÁ AMBAS AS CLASSES
if y.nunique() < 2:
    print("\n❌ ERRO CRÍTICO: Impossível continuar com apenas uma classe!")
    print("Verifique os dados originais e a codificação da variável 'salary'")
    exit()

# Pré-processador para colunas (OneHot para categóricas, StandardScaler para numéricas)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
    ])

# Divisão treino/teste com stratify para manter proporção das classes
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y  # Importante: mantém proporção das classes
)

# Verificar distribuição nas partições
print(f"\nDistribuição em y_train:")
print(y_train.value_counts())
print(f"\nDistribuição em y_test:")
print(y_test.value_counts())

# Ajustar transformações
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Novo: Garantir que X_test_processed é do tipo float
X_test_processed = X_test_processed.astype(float)

print("\nPré-processamento completo. Formato final dos dados:")
print("X_train:", X_train_processed.shape)
print("X_test:", X_test_processed.shape)
print("y_train classes únicas:", y_train.unique())
print("y_test classes únicas:", y_test.unique())

# ================================================
# 5. BALANCEAMENTO DOS DADOS (SMOTE) - CONDICIONAL
# ================================================
print("\n" + "="*60)
print("BALANCEAMENTO DOS DADOS")
print("="*60)

# Verificar se precisa de balanceamento
class_counts = y_train.value_counts()
minority_ratio = min(class_counts) / max(class_counts)

print(f"Proporção da classe minoritária: {minority_ratio:.2f}")

if minority_ratio < 0.3:  # Se a classe minoritária representa menos de 30%
    print("Dataset desbalanceado. Aplicando SMOTE...")
    from imblearn.over_sampling import SMOTE
    
    # Verificar se há pelo menos 2 classes antes do SMOTE
    if len(y_train.unique()) >= 2:
        sm = SMOTE(random_state=42)
        X_train_processed, y_train = sm.fit_resample(X_train_processed, y_train)
        
        print("✅ SMOTE aplicado com sucesso")
        print("Distribuição após SMOTE:")
        print(y_train.value_counts())
    else:
        print("⚠️ Impossível aplicar SMOTE: apenas uma classe nos dados de treino")
else:
    print("Dataset já balanceado. SMOTE não necessário.")

# ================================================
# 6. MODELAGEM PREDITIVA
# ================================================
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import joblib

# Função para treinar e avaliar modelos
def avaliar_modelo(nome, modelo, X_train, y_train, X_test, y_test):
    # Verificar se há pelo menos 2 classes nos dados de treino
    if len(y_train.unique()) < 2:
        print(f"\n⚠️ {nome}: Impossível treinar - apenas uma classe nos dados de treino")
        print(f"Classes em y_train: {y_train.unique()}")
        return None
    
    try:
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        print(f"\n{'='*20}\n{nome}\n{'='*20}")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("Matriz de Confusão:")
        print(confusion_matrix(y_test, y_pred))
        return modelo
    except Exception as e:
        print(f"\n⚠️ Erro ao treinar {nome}: {str(e)}")
        return None

# Inicializar modelos
modelos = {
    "Regressão Logística": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5)
}

# Avaliar todos os modelos e salvar
modelos_treinados = {}
for nome, modelo in modelos.items():
    modelo_treinado = avaliar_modelo(nome, modelo, X_train_processed, y_train, X_test_processed, y_test)
    if modelo_treinado is not None:
        modelos_treinados[nome] = modelo_treinado

# ================================================
# SALVAR MODELOS E PREPROCESSOR
# ================================================
print("\n" + "="*60)
print("SALVANDO MODELOS E PREPROCESSOR")
print("="*60)

# Criar diretório models se não existir
os.makedirs("models", exist_ok=True)

# Salvar o modelo Random Forest (melhor para interpretabilidade)
if "Random Forest" in modelos_treinados:
    joblib.dump(modelos_treinados["Random Forest"], "random_forest_model.joblib")
    print("✅ Modelo Random Forest salvo como 'random_forest_model.joblib'")

# Salvar o preprocessor
joblib.dump(preprocessor, "preprocessor.joblib")
print("✅ Preprocessor salvo como 'preprocessor.joblib'")

# Gerar nomes das features após o preprocessamento
ohe = preprocessor.named_transformers_['cat']
cat_feature_names = ohe.get_feature_names_out(categorical_features)
feature_names = numerical_features + list(cat_feature_names)

# Salvar informações sobre as features
feature_info = {
    'numerical_features': numerical_features,
    'categorical_features': categorical_features,
    'feature_names': feature_names
}
joblib.dump(feature_info, "feature_info.joblib")
print("✅ Informações das features salvas como 'feature_info.joblib'")

# Salvar alguns dados de exemplo para testes
sample_data = X_test.head(5)
joblib.dump(sample_data, "sample_data.joblib")
print("✅ Dados de exemplo salvos como 'sample_data.joblib'")

print(f"\n📁 Todos os arquivos salvos com sucesso!")
print("Arquivos disponíveis para o dashboard:")
print("- random_forest_model.joblib")
print("- preprocessor.joblib") 
print("- feature_info.joblib")
print("- sample_data.joblib")

# ================================================
# 7. CLUSTERING E REGRAS DE ASSOCIAÇÃO
# ================================================
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

# Usar PCA para reduzir a dimensionalidade para 2D para visualização
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_processed.toarray() if hasattr(X_train_processed, 'toarray') else X_train_processed)

# KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_pca)

# Plot dos clusters
plt.figure(figsize=(8, 6))
# Configurar fundo transparente
plt.gca().patch.set_alpha(0.0)
plt.gcf().patch.set_alpha(0.0)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', s=10)
plt.title("Visualização dos Clusters com KMeans (PCA 2D)")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.tight_layout()
plt.savefig("imagens/kmeans_clusters.png", transparent=True, bbox_inches='tight')
plt.close()
print("\nClustering com KMeans concluído. Gráfico salvo em 'imagens/kmeans_clusters.png'.")

# -------------------------------------
# Regras de associação (via Apriori)
# -------------------------------------
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Criar um dataset amostral com transações binárias (apenas para demonstração simples)
amostras = df[['education', 'occupation', 'relationship', 'sex', 'salary']].astype(str).values.tolist()
te = TransactionEncoder()
transacoes = te.fit(amostras).transform(amostras)
df_transacoes = pd.DataFrame(transacoes, columns=te.columns_)

# Aplicar Apriori
freq_items = apriori(df_transacoes, min_support=0.05, use_colnames=True)
regras = association_rules(freq_items, metric="confidence", min_threshold=0.6)

# Mostrar as 5 principais regras
print("\nTop 5 Regras de Associação:")
print(regras[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())

# ================================================
# 8. INTERPRETAÇÃO DOS MODELOS
# ================================================
print("\n" + "="*60)
print("INTERPRETAÇÃO DOS MODELOS")
print("="*60)

if "Random Forest" in modelos_treinados:
    # -------------------------------
    # FEATURE IMPORTANCE - Random Forest
    # -------------------------------
    importances = modelos_treinados["Random Forest"].feature_importances_
    
    # Obter nomes das colunas após transformação
    ohe = preprocessor.named_transformers_['cat']
    cat_feature_names = ohe.get_feature_names_out(categorical_features)
    feature_names = numerical_features + list(cat_feature_names)
    
    # Combinar com os valores
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    # Mostrar top 15
    print("\nTop 15 Features mais importantes (Random Forest):")
    print(feature_importance_df.head(15))
    
    # Plot com fundo transparente
    plt.figure(figsize=(10, 6))
    # Configurar fundo transparente
    plt.gca().patch.set_alpha(0.0)
    plt.gcf().patch.set_alpha(0.0)
    
    sns.barplot(data=feature_importance_df.head(15), x='Importance', y='Feature')
    plt.title("Importância das Features - Random Forest")
    plt.tight_layout()
    plt.savefig("imagens/feature_importance_rf.png", transparent=True, bbox_inches='tight')
    plt.close()
    
    # -------------------------------
    # COEFICIENTES - Regressão Logística (se disponível)
    # -------------------------------
    if "Regressão Logística" in modelos_treinados:
        coef = modelos_treinados["Regressão Logística"].coef_[0]
        
        coef_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coef
        }).sort_values(by='Coefficient', key=abs, ascending=False)
        
        print("\nTop 15 Coeficientes mais importantes (Regressão Logística):")
        print(coef_df.head(15))
        
        # Plot com fundo transparente
        plt.figure(figsize=(10, 6))
        # Configurar fundo transparente
        plt.gca().patch.set_alpha(0.0)
        plt.gcf().patch.set_alpha(0.0)
        
        sns.barplot(data=coef_df.head(15), x='Coefficient', y='Feature')
        plt.title("Coeficientes - Regressão Logística")
        plt.tight_layout()
        plt.savefig("imagens/coefficients_lr.png", transparent=True, bbox_inches='tight')
        plt.close()
    
    # -------------------------------
    # SHAP VALUES (corrigido)
    # -------------------------------
    try:
        import shap
        
        # Converter para array denso se necessário e garantir tipo float
        X_train_shap = X_train_processed
        if hasattr(X_train_shap, 'toarray'):
            X_train_shap = X_train_shap.toarray()
        
        # Garantir que é float64
        X_train_shap = X_train_shap.astype(np.float64)
        
        # Usar apenas uma amostra menor para SHAP (mais rápido)
        sample_size = min(1000, X_train_shap.shape[0])
        X_sample = X_train_shap[:sample_size]
        
        print(f"\nGerando SHAP values para {sample_size} amostras...")
        
        explainer = shap.TreeExplainer(modelos_treinados["Random Forest"])
        shap_values = explainer.shap_values(X_sample)
        
        # Se o modelo retorna valores SHAP para cada classe, usar a classe positiva
        if isinstance(shap_values, list):
            shap_values_plot = shap_values[1]  # Classe positiva (>50K)
        else:
            shap_values_plot = shap_values
        
        # Summary plot com fundo transparente
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values_plot, X_sample, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig("imagens/shap_summary_rf.png", transparent=True, bbox_inches='tight')
        plt.close()
        
        # Bar plot SHAP com fundo transparente
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values_plot, X_sample, feature_names=feature_names, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig("imagens/shap_bar_rf.png", transparent=True, bbox_inches='tight')
        plt.close()
        
        print("✅ Gráficos SHAP gerados com sucesso.")
        
    except ImportError:
        print("\n⚠️ Pacote SHAP não instalado. Execute: pip install shap")
    except Exception as e:
        print(f"\n⚠️ Erro ao gerar gráficos SHAP: {str(e)}")
        print("Continuando sem SHAP...")

else:
    print("⚠️ Modelo Random Forest não foi treinado com sucesso.")

print("\n🎉 Pipeline completo executado!")
print("\n📁 Arquivos gerados:")
print("- Dados: 4-Carateristicas_salario.csv")
print("- Modelo: random_forest_model.joblib")
print("- Preprocessor: preprocessor.joblib")
print("- Features: feature_info.joblib")
print("- Amostras: sample_data.joblib")
print("- Gráficos: pasta imagens/ (todos com fundo transparente)")
