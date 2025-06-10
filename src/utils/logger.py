"""Configuração de logging melhorada"""

import logging
import sys
from pathlib import Path

def setup_logging(level=logging.INFO):
    """Configurar logging detalhado"""
    
    # Criar diretório de logs
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configurar formato
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Handler para arquivo
    file_handler = logging.FileHandler(log_dir / "ml_pipeline.log")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Handler para console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    
    # Configurar logger raiz
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return root_logger

def log_success(message):
    """Log com checkmark verde"""
    logging.info(f"✅ {message}")

def log_function_start(function_name):
    """Log início de função"""
    logging.info(f"🔄 Iniciando: {function_name}")

def log_function_end(function_name):
    """Log fim de função com sucesso"""
    logging.info(f"✅ Concluído: {function_name}")

def log_function(func):
    """Decorator para logging de início e fim de função"""
    def wrapper(*args, **kwargs):
        logging.info(f"🔄 Iniciando: {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logging.info(f"✅ Concluído: {func.__name__}")
            return result
        except Exception as e:
            logging.error(f"❌ Erro em {func.__name__}: {e}")
            raise
    return wrapper

def get_memory_usage(df):
    """Calcular uso de memória do DataFrame"""
    memory_usage = df.memory_usage(deep=True).sum()
    return memory_usage / 1024 / 1024  # MB