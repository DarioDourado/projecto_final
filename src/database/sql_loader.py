"""
Carregador de dados SQL com fallback para CSV
"""

import logging
import pandas as pd
from pathlib import Path
import sqlite3
from typing import Optional, Dict, Any

class SQLDataLoader:
    """Carregador de dados SQL com configurações flexíveis"""
    
    def __init__(self, connection_string: Optional[str] = None):
        """
        Inicializar carregador SQL
        
        Args:
            connection_string: String de conexão personalizada (opcional)
        """
        self.logger = logging.getLogger(__name__)
        self.connection_string = connection_string
        self.connection = None
        
    def _get_default_connection_string(self) -> str:
        """Obter string de conexão padrão"""
        # Exemplo para SQLite local
        db_path = Path("data/database/salary_data.db")
        return f"sqlite:///{db_path}"
    
    def _create_test_database(self) -> bool:
        """Criar base de dados de teste se não existir"""
        try:
            db_path = Path("data/database")
            db_path.mkdir(parents=True, exist_ok=True)
            
            # Criar SQLite de teste
            conn = sqlite3.connect(db_path / "salary_data.db")
            
            # Verificar se tabela já existe
            cursor = conn.cursor()
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='salary_data'
            """)
            
            if cursor.fetchone() is None:
                # Criar tabela se não existir
                cursor.execute("""
                    CREATE TABLE salary_data (
                        id INTEGER PRIMARY KEY,
                        age INTEGER,
                        workclass TEXT,
                        fnlwgt INTEGER,
                        education TEXT,
                        education_num INTEGER,
                        marital_status TEXT,
                        occupation TEXT,
                        relationship TEXT,
                        race TEXT,
                        sex TEXT,
                        capital_gain INTEGER,
                        capital_loss INTEGER,
                        hours_per_week INTEGER,
                        native_country TEXT,
                        salary TEXT
                    )
                """)
                conn.commit()
                self.logger.info("✅ Tabela SQL criada")
            
            conn.close()
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao criar base de dados: {e}")
            return False
    
    def connect(self) -> bool:
        """Estabelecer conexão com a base de dados"""
        try:
            connection_string = self.connection_string or self._get_default_connection_string()
            
            # Para SQLite
            if "sqlite:" in connection_string:
                db_path = connection_string.replace("sqlite:///", "")
                
                # Criar base de dados se não existir
                if not Path(db_path).exists():
                    self._create_test_database()
                
                self.connection = sqlite3.connect(db_path)
                self.logger.info(f"✅ Conectado ao SQLite: {db_path}")
                return True
            
            # Para outras bases de dados (PostgreSQL, MySQL, etc.)
            else:
                # Implementar outras conexões conforme necessário
                import sqlalchemy
                engine = sqlalchemy.create_engine(connection_string)
                self.connection = engine.connect()
                self.logger.info("✅ Conectado à base de dados SQL")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Falha na conexão SQL: {e}")
            self.connection = None
            return False
    
    def load_salary_data(self) -> Optional[pd.DataFrame]:
        """
        Carregar dados salariais da base de dados
        
        Returns:
            DataFrame com dados ou None se falhar
        """
        try:
            if not self.connect():
                self.logger.warning("⚠️ Conexão SQL falhou")
                return None
            
            # Query para dados salariais
            query = """
                SELECT 
                    age, workclass, fnlwgt, education, education_num,
                    marital_status, occupation, relationship, race, sex,
                    capital_gain, capital_loss, hours_per_week, 
                    native_country, salary
                FROM salary_data
                WHERE age IS NOT NULL
                ORDER BY age
            """
            
            # Executar query
            df = pd.read_sql_query(query, self.connection)
            
            if len(df) > 0:
                self.logger.info(f"✅ Dados SQL carregados: {len(df):,} registros")
                return df
            else:
                self.logger.warning("⚠️ Query SQL retornou dados vazios")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Erro ao carregar dados SQL: {e}")
            return None
        
        finally:
            if self.connection:
                self.connection.close()
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Testar conexão e retornar informações
        
        Returns:
            Dicionário com status da conexão
        """
        result = {
            'connected': False,
            'error': None,
            'database_type': 'unknown',
            'tables_count': 0,
            'records_count': 0
        }
        
        try:
            if self.connect():
                result['connected'] = True
                
                # Identificar tipo de base de dados
                if hasattr(self.connection, 'execute'):
                    result['database_type'] = 'sqlite'
                    
                    # Contar tabelas
                    cursor = self.connection.cursor()
                    cursor.execute("""
                        SELECT COUNT(*) FROM sqlite_master 
                        WHERE type='table'
                    """)
                    result['tables_count'] = cursor.fetchone()[0]
                    
                    # Contar registros na tabela principal
                    try:
                        cursor.execute("SELECT COUNT(*) FROM salary_data")
                        result['records_count'] = cursor.fetchone()[0]
                    except:
                        result['records_count'] = 0
                
                self.logger.info(f"✅ Teste SQL: {result}")
                
        except Exception as e:
            result['error'] = str(e)
            self.logger.error(f"❌ Teste SQL falhou: {e}")
        
        finally:
            if self.connection:
                self.connection.close()
        
        return result