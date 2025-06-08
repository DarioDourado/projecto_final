import logging
import sys
from datetime import datetime

def setup_logging(log_level=logging.INFO, log_file=None):
    """
    Configura o sistema de logging para a aplicação.
    
    Args:
        log_level: Nível de logging (DEBUG, INFO, WARNING, ERROR)
        log_file: Caminho para arquivo de log (opcional)
    
    Returns:
        logger: Instância do logger configurado
    """
    # Configuração do formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configuração do logger
    logger = logging.getLogger('ml_pipeline')
    logger.setLevel(log_level)
    
    # Handler para console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler para arquivo (se especificado)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger