# ================================================
# 8. INTERPRETAÇÃO DOS MODELOS
# ================================================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib
import os

# Configuração global para fundo transparente
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none'

# Criar diretório para imagens se não existir
os.makedirs("imagens", exist_ok=True)

# Carregar modelo e dados salvos
try:
    modelo = joblib.load("random_forest_model.joblib")
    preprocessor = joblib.load("preprocessor.joblib")
    feature_info = joblib.load("feature_info.joblib")
    
    feature_names = feature_info['feature_names']
    
    print("✅ Modelos e dados carregados com sucesso!")
    
except FileNotFoundError as e:
    print(f"❌ Arquivo não encontrado: {e}")
    print("Execute primeiro: python projeto_salario.py")
    exit()

# -------------------------------
# FEATURE IMPORTANCE - Random Forest
# -------------------------------
importances = modelo.feature_importances_

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
# SHAP VALUES (corrigido)
# -------------------------------
try:
    import shap
    
    # Carregar dados de exemplo
    sample_data = joblib.load("sample_data.joblib")
    
    # Preprocessar os dados de exemplo
    sample_processed = preprocessor.transform(sample_data)
    
    # Converter para array denso se necessário
    if hasattr(sample_processed, 'toarray'):
        sample_processed = sample_processed.toarray()
    
    # Garantir tipo float64
    sample_processed = sample_processed.astype(np.float64)
    
    print(f"\nGerando SHAP values para {sample_processed.shape[0]} amostras...")
    
    explainer = shap.TreeExplainer(modelo)
    shap_values = explainer.shap_values(sample_processed)
    
    # Se o modelo retorna valores SHAP para cada classe
    if isinstance(shap_values, list):
        shap_values_plot = shap_values[1]  # Classe positiva
    else:
        shap_values_plot = shap_values
    
    # Summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_plot, sample_processed, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig("imagens/shap_summary_rf.png", transparent=True, bbox_inches='tight')
    plt.close()
    
    # Bar plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_plot, sample_processed, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig("imagens/shap_bar_rf.png", transparent=True, bbox_inches='tight')
    plt.close()
    
    print("✅ Gráficos SHAP gerados com sucesso.")
    
except ImportError:
    print("\n⚠️ Pacote SHAP não instalado. Execute: pip install shap")
except Exception as e:
    print(f"\n⚠️ Erro ao gerar gráficos SHAP: {str(e)}")

print("\n🎉 Interpretabilidade concluída!")
print("📁 Gráficos salvos na pasta 'imagens/'")
