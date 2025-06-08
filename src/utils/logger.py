"""Sistema de logging com emojis - Extraído do projeto_salario.py"""

import logging
import sys
import os
from pathlib import Path
from datetime import datetime

class EmojiFormatter(logging.Formatter):
    """Formatter personalizado com emojis"""
    
    emoji_mapping = {
        'DEBUG': '🔍',
        'INFO': 'ℹ️',
        'WARNING': '⚠️',
        'ERROR': '❌',
        'CRITICAL': '🚨',
        'SUCCESS': '✅'
    }
    
    def format(self, record):
        emoji = self.emoji_mapping.get(record.levelname, 'ℹ️')
        return f"{emoji} {record.getMessage()}"

def setup_logging():
    """Configurar sistema de logging"""
    logger = logging.getLogger()
    
    # Limpar handlers existentes para evitar duplicação
    logger.handlers = []
    
    # Handler para console
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(EmojiFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    
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