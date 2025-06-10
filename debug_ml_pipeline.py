"""Script para diagnosticar problemas no pipeline ML"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent / "src"))

def debug_ml_pipeline():
    """Diagnosticar problemas no pipeline ML"""
    print("🔍 DIAGNÓSTICO DO PIPELINE ML")
    print("="*40)
    
    try:
        # 1. Testar carregamento de dados
        print("1. Testando carregamento de dados...")
        from src.pipelines.data_pipeline import DataPipelineSQL
        
        data_pipeline = DataPipelineSQL()
        df = data_pipeline.run()
        
        if df is None:
            print("❌ Dados não carregados")
            return
        
        print(f"✅ Dados carregados: {len(df)} registros")
        print(f"📋 Colunas: {list(df.columns)}")
        print(f"📊 Shape: {df.shape}")
        
        # 2. Verificar dados
        print("\n2. Verificando qualidade dos dados...")
        print(f"   Valores nulos por coluna:")
        null_counts = df.isnull().sum()
        for col, count in null_counts.items():
            if count > 0:
                print(f"     {col}: {count}")
        
        # 3. Testar preparação de dados
        print("\n3. Testando preparação de dados...")
        from src.pipelines.ml_pipeline import MLPipeline
        
        ml_pipeline = MLPipeline()
        X, y = ml_pipeline._prepare_data(df)
        
        if X is None or y is None:
            print("❌ Falha na preparação dos dados")
            return
        
        print(f"✅ Features preparadas: {X.shape}")
        print(f"✅ Target preparado: {y.shape}")
        
        # 4. Testar divisão treino/teste
        print("\n4. Testando divisão treino/teste...")
        X_train, X_test, y_train, y_test = ml_pipeline._safe_train_test_split(X, y)
        
        print(f"✅ Treino: X({X_train.shape}) y({y_train.shape})")
        print(f"✅ Teste: X({X_test.shape}) y({y_test.shape})")
        
        # 5. Testar modelo específico
        print("\n5. Testando Logistic Regression isoladamente...")
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score
        
        # Normalizar
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Treinar
        lr = LogisticRegression(random_state=42, max_iter=1000, solver='liblinear')
        lr.fit(X_train_scaled, y_train)
        
        # Predizer
        y_pred = lr.predict(X_test_scaled)
        
        # Validar
        print(f"   Predições shape: {y_pred.shape}")
        print(f"   Teste shape: {y_test.shape}")
        print(f"   Compatível: {len(y_pred) == len(y_test)}")
        
        if len(y_pred) == len(y_test):
            accuracy = accuracy_score(y_test, y_pred)
            print(f"✅ Acurácia: {accuracy:.4f}")
        else:
            print("❌ Incompatibilidade de tamanhos!")
            print(f"   y_pred: {len(y_pred)}")
            print(f"   y_test: {len(y_test)}")
        
        print("\n🎉 Diagnóstico concluído!")
        
    except Exception as e:
        print(f"❌ Erro durante diagnóstico: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_ml_pipeline()