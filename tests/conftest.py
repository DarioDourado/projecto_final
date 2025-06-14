"""
🧪 Configuração de Testes para Data Science
Fixtures compartilhadas e configurações globais
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
import logging
import sys
from datetime import datetime

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent.parent / "src"))

@pytest.fixture(scope="session")
def test_data_dir():
    """Diretório temporário para dados de teste"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture(scope="session") 
def sample_salary_data():
    """Dataset de exemplo para testes"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'age': np.random.randint(18, 80, n_samples),
        'fnlwgt': np.random.randint(10000, 500000, n_samples),
        'education-num': np.random.randint(1, 16, n_samples),
        'capital-gain': np.random.randint(0, 10000, n_samples),
        'capital-loss': np.random.randint(0, 5000, n_samples),
        'hours-per-week': np.random.randint(20, 80, n_samples),
        'workclass': np.random.choice(['Private', 'Self-emp-not-inc', 'Federal-gov'], n_samples),
        'education': np.random.choice(['Bachelors', 'Masters', 'Doctorate'], n_samples),
        'marital-status': np.random.choice(['Married-civ-spouse', 'Never-married'], n_samples),
        'occupation': np.random.choice(['Tech-support', 'Craft-repair', 'Sales'], n_samples),
        'relationship': np.random.choice(['Husband', 'Wife', 'Own-child'], n_samples),
        'race': np.random.choice(['White', 'Black', 'Asian-Pac-Islander'], n_samples),
        'sex': np.random.choice(['Male', 'Female'], n_samples),
        'native-country': np.random.choice(['United-States', 'Canada', 'Mexico'], n_samples),
        'salary': np.random.choice(['<=50K', '>50K'], n_samples, p=[0.76, 0.24])
    }
    
    return pd.DataFrame(data)

@pytest.fixture
def mock_ml_results():
    """Resultados de ML mockados para testes"""
    return {
        'Random Forest': {
            'accuracy': 0.8408,
            'precision': 0.7234,
            'recall': 0.6789,
            'f1_score': 0.7004,
            'roc_auc': 0.8123
        },
        'Logistic Regression': {
            'accuracy': 0.8185,
            'precision': 0.6987,
            'recall': 0.6543,
            'f1_score': 0.6758,
            'roc_auc': 0.7891
        }
    }

@pytest.fixture
def expected_data_quality_metrics():
    """Métricas esperadas de qualidade de dados"""
    return {
        'total_records': 32561,
        'total_columns': 15,
        'numeric_columns': 6,
        'categorical_columns': 9,
        'missing_data_columns': ['workclass', 'occupation', 'native-country'],
        'duplicate_percentage': 0.1,
        'target_distribution': {'<=50K': 0.759, '>50K': 0.241}
    }

@pytest.fixture
def model_performance_thresholds():
    """Thresholds mínimos para performance dos modelos"""
    return {
        'accuracy': 0.80,
        'precision': 0.65,
        'recall': 0.60,
        'f1_score': 0.65,
        'roc_auc': 0.75
    }

@pytest.fixture
def setup_test_logging():
    """Setup de logging para testes"""
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.INFO)
    
    # Handler para capturar logs
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger