"""Conexão com banco de dados"""

import logging
import mysql.connector
from mysql.connector import Error
from pathlib import Path
import os
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class DatabaseConnection:
    """Gerenciador de conexão com banco de dados"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._load_config()
        self.connection = None
        self.cursor = None
    
    def _load_config(self) -> Dict[str, Any]:
        """Carregar configuração do banco"""
        return {
            'host': os.getenv('DB_HOST', 'localhost'),
            'database': os.getenv('DB_NAME', 'salary_analysis'),
            'user': os.getenv('DB_USER', 'salary_user'),
            'password': os.getenv('DB_PASSWORD', 'senha_forte'),
            'charset': 'utf8mb4',
            'collation': 'utf8mb4_unicode_ci'
        }
    
    def connect(self) -> bool:
        """Conectar ao banco de dados"""
        try:
            self.connection = mysql.connector.connect(**self.config)
            self.cursor = self.connection.cursor(dictionary=True)
            logger.info("✅ Conectado ao banco de dados MySQL")
            return True
        except Error as e:
            logger.error(f"❌ Erro ao conectar: {e}")
            return False
    
    def disconnect(self):
        """Desconectar do banco"""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        logger.info("🔌 Desconectado do banco de dados")
    
    def execute_query(self, query: str, params: tuple = None) -> Optional[list]:
        """Executar query SELECT"""
        try:
            self.cursor.execute(query, params or ())
            return self.cursor.fetchall()
        except Error as e:
            logger.error(f"❌ Erro na query: {e}")
            return None
    
    def execute_update(self, query: str, params: tuple = None) -> bool:
        """Executar query INSERT/UPDATE/DELETE"""
        try:
            self.cursor.execute(query, params or ())
            self.connection.commit()
            return True
        except Error as e:
            logger.error(f"❌ Erro na atualização: {e}")
            self.connection.rollback()
            return False
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()