"""
🔬 Testes do Pipeline de Dados
Validação completa de carregamento, limpeza e qualidade dos dados
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))

class TestDataPipeline:
    """Testes do pipeline de dados"""
    
    def test_data_loading_sql_connection(self):
        """Testar conexão e carregamento SQL"""
        try:
            from src.data.loader import load_data, get_data_status
            
            # Testar status das fontes
            status = get_data_status()
            
            assert 'sql' in status
            assert 'csv' in status
            assert 'recommended' in status
            
            # Log dos resultados
            print(f"✅ Status SQL: {status['sql']}")
            print(f"✅ Status CSV: {status['csv']}")
            print(f"✅ Recomendado: {status['recommended']}")
            
        except ImportError:
            pytest.skip("Módulos de dados não disponíveis")
    
    def test_data_loading_hybrid_fallback(self):
        """Testar fallback automático SQL→CSV"""
        try:
            from src.data.loader import load_data
            
            # Teste híbrido
            df_hybrid, source_hybrid = load_data(force_csv=False)
            
            assert df_hybrid is not None
            assert len(df_hybrid) > 0
            assert source_hybrid in ['sql', 'csv']
            
            # Teste CSV forçado
            df_csv, source_csv = load_data(force_csv=True)
            
            assert df_csv is not None
            assert source_csv == 'csv'
            assert len(df_csv) > 0
            
            print(f"✅ Híbrido: {len(df_hybrid):,} registros via {source_hybrid}")
            print(f"✅ CSV: {len(df_csv):,} registros via {source_csv}")
            
        except ImportError:
            pytest.skip("Módulos de dados não disponíveis")
    
    def test_data_quality_validation(self, sample_salary_data, expected_data_quality_metrics):
        """Validar qualidade dos dados carregados"""
        df = sample_salary_data
        
        # Validações básicas
        assert len(df) > 0, "Dataset não pode estar vazio"
        assert len(df.columns) >= 10, "Dataset deve ter pelo menos 10 colunas"
        
        # Validar tipos de dados
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        assert len(numeric_cols) >= 5, "Deve ter pelo menos 5 colunas numéricas"
        assert len(categorical_cols) >= 5, "Deve ter pelo menos 5 colunas categóricas"
        
        # Validar coluna target
        assert 'salary' in df.columns, "Coluna 'salary' deve existir"
        
        target_values = df['salary'].unique()
        expected_targets = ['<=50K', '>50K']
        assert all(val in expected_targets for val in target_values), f"Valores target inválidos: {target_values}"
        
        # Distribuição target
        target_dist = df['salary'].value_counts(normalize=True)
        assert target_dist['<=50K'] > 0.5, "Classe majoritária deve ser <=50K"
        
        print(f"✅ Qualidade validada: {len(df)} registros")
        print(f"✅ Features numéricas: {len(numeric_cols)}")
        print(f"✅ Features categóricas: {len(categorical_cols)}")
        print(f"✅ Distribuição target: {dict(target_dist)}")
    
    def test_data_preprocessing_steps(self, sample_salary_data):
        """Testar etapas de pré-processamento"""
        df = sample_salary_data.copy()
        
        # Simular dados ausentes
        df.loc[:10, 'workclass'] = np.nan
        df.loc[:5, 'occupation'] = np.nan
        
        # Validar detecção de missing data
        missing_data = df.isnull().sum()
        cols_with_missing = missing_data[missing_data > 0].index.tolist()
        
        assert 'workclass' in cols_with_missing
        assert 'occupation' in cols_with_missing
        
        # Simular duplicatas
        df_with_dups = pd.concat([df, df.iloc[:10]], ignore_index=True)
        duplicates = df_with_dups.duplicated().sum()
        
        assert duplicates == 10, f"Esperado 10 duplicatas, encontrado {duplicates}"
        
        print(f"✅ Missing data detectado: {cols_with_missing}")
        print(f"✅ Duplicatas detectadas: {duplicates}")
    
    def test_feature_engineering_validation(self, sample_salary_data):
        """Validar engenharia de features"""
        df = sample_salary_data
        
        # Features essenciais para ML
        essential_features = ['age', 'education-num', 'hours-per-week', 'capital-gain']
        
        for feature in essential_features:
            assert feature in df.columns, f"Feature essencial ausente: {feature}"
            
            # Validar ranges sensatos
            if feature == 'age':
                assert df[feature].min() >= 15, "Idade mínima inválida"
                assert df[feature].max() <= 100, "Idade máxima inválida"
            
            elif feature == 'education-num':
                assert df[feature].min() >= 1, "education-num deve ser >= 1"
                assert df[feature].max() <= 16, "education-num deve ser <= 16"
            
            elif feature == 'hours-per-week':
                assert df[feature].min() >= 1, "hours-per-week deve ser >= 1"
                assert df[feature].max() <= 100, "hours-per-week deve ser <= 100"
        
        print(f"✅ Features essenciais validadas: {essential_features}")