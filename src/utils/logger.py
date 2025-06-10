"""Sistema de logging configurado"""

import logging
import sys
from datetime import datetime
from pathlib import Path

def setup_logging(level=logging.INFO):
    """Configurar sistema de logging"""
    
    # Criar diretório de logs
    logs_dir = Path("output/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Nome do arquivo de log com timestamp
    log_filename = logs_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Configurar formatação
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configurar handler para arquivo
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    
    # Configurar handler para console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Configurar logger principal
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Remover handlers existentes para evitar duplicatas
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Adicionar novos handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Configurar loggers específicos para reduzir ruído
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    return logger

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