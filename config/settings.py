"""Configurações centralizadas do projeto"""

import os
from pathlib import Path

# Paths do projeto
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = BASE_DIR / "output"
IMAGES_DIR = OUTPUT_DIR / "images"
MODELS_DIR = BASE_DIR / "models" / "trained"
LOGS_DIR = OUTPUT_DIR / "logs"

# Configurações de dados
DATA_CONFIG = {
    'source_file': '4-Carateristicas_salario.csv',
    'target_column': 'salary',
    'test_size': 0.2,
    'random_state': 42
}

# Configurações de modelos
MODEL_CONFIG = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42,
        'n_jobs': -1
    },
    'logistic_regression': {
        'random_state': 42,
        'max_iter': 1000
    }
}

# Configurações de visualização
VISUALIZATION_CONFIG = {
    'style': 'modern',
    'dpi': 300,
    'figsize': (12, 8),
    'save_format': 'png'
}

# Paleta de cores moderna
MODERN_COLORS = {
    'primary': '#007bff',
    'secondary': '#6c757d', 
    'success': '#28a745',
    'danger': '#dc3545',
    'warning': '#ffc107',
    'info': '#17a2b8',
    'light': '#f8f9fa',
    'dark': '#343a40'
}

# Tipagem de colunas
COLUMN_TYPES = {
    'numerical': {
        'age': 'int16',
        'fnlwgt': 'int32',
        'education-num': 'int8',
        'capital-gain': 'int32',
        'capital-loss': 'int16',
        'hours-per-week': 'int8'
    },
    'categorical': [
        'workclass', 'education', 'marital-status', 'occupation',
        'relationship', 'race', 'sex', 'native-country', 'salary'
    ]
}

# Validação de ranges
VALIDATION_RANGES = {
    'age': (17, 100),
    'education-num': (1, 16),
    'hours-per-week': (1, 99)
}

# Criar diretórios se não existirem
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DIR, 
                  IMAGES_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)