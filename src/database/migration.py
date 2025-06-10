"""Sistema de migração CSV → Banco de Dados"""

import pandas as pd
import mysql.connector
from mysql.connector import Error
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
from .connection import DatabaseConnection

logger = logging.getLogger(__name__)

class DatabaseMigrator:
    """Migrador de dados CSV para MySQL"""
    
    def __init__(self, db_connection: DatabaseConnection = None):
        self.db = db_connection or DatabaseConnection()
    
    def create_database_structure(self) -> bool:
        """Criar estrutura completa do banco"""
        
        tables = {
            'workclass': """
                CREATE TABLE IF NOT EXISTS workclass (
                    id INT PRIMARY KEY AUTO_INCREMENT,
                    name VARCHAR(50) UNIQUE NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """,
            
            'education': """
                CREATE TABLE IF NOT EXISTS education (
                    id INT PRIMARY KEY AUTO_INCREMENT,
                    name VARCHAR(50) UNIQUE NOT NULL,
                    education_num INT,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """,
            
            'marital_status': """
                CREATE TABLE IF NOT EXISTS marital_status (
                    id INT PRIMARY KEY AUTO_INCREMENT,
                    name VARCHAR(50) UNIQUE NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """,
            
            'occupation': """
                CREATE TABLE IF NOT EXISTS occupation (
                    id INT PRIMARY KEY AUTO_INCREMENT,
                    name VARCHAR(50) UNIQUE NOT NULL,
                    category VARCHAR(50),
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """,
            
            'relationship': """
                CREATE TABLE IF NOT EXISTS relationship (
                    id INT PRIMARY KEY AUTO_INCREMENT,
                    name VARCHAR(50) UNIQUE NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """,
            
            'race': """
                CREATE TABLE IF NOT EXISTS race (
                    id INT PRIMARY KEY AUTO_INCREMENT,
                    name VARCHAR(50) UNIQUE NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """,
            
            'country': """
                CREATE TABLE IF NOT EXISTS country (
                    id INT PRIMARY KEY AUTO_INCREMENT,
                    name VARCHAR(100) UNIQUE NOT NULL,
                    code VARCHAR(3),
                    continent VARCHAR(50),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """,
            
            'salary_range': """
                CREATE TABLE IF NOT EXISTS salary_range (
                    id INT PRIMARY KEY AUTO_INCREMENT,
                    name VARCHAR(20) UNIQUE NOT NULL,
                    min_amount DECIMAL(10,2),
                    max_amount DECIMAL(10,2),
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """,
            
            'person': """
                CREATE TABLE IF NOT EXISTS person (
                    id INT PRIMARY KEY AUTO_INCREMENT,
                    age INT NOT NULL CHECK (age BETWEEN 17 AND 100),
                    fnlwgt INT,
                    workclass_id INT,
                    education_id INT,
                    marital_status_id INT,
                    occupation_id INT,
                    relationship_id INT,
                    race_id INT,
                    sex ENUM('Male', 'Female') NOT NULL,
                    capital_gain DECIMAL(10,2) DEFAULT 0,
                    capital_loss DECIMAL(10,2) DEFAULT 0,
                    hours_per_week INT CHECK (hours_per_week BETWEEN 1 AND 99),
                    native_country_id INT,
                    salary_range_id INT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    
                    FOREIGN KEY (workclass_id) REFERENCES workclass(id),
                    FOREIGN KEY (education_id) REFERENCES education(id),
                    FOREIGN KEY (marital_status_id) REFERENCES marital_status(id),
                    FOREIGN KEY (occupation_id) REFERENCES occupation(id),
                    FOREIGN KEY (relationship_id) REFERENCES relationship(id),
                    FOREIGN KEY (race_id) REFERENCES race(id),
                    FOREIGN KEY (native_country_id) REFERENCES country(id),
                    FOREIGN KEY (salary_range_id) REFERENCES salary_range(id),
                    
                    INDEX idx_age (age),
                    INDEX idx_education (education_id),
                    INDEX idx_salary (salary_range_id),
                    INDEX idx_workclass (workclass_id),
                    INDEX idx_occupation (occupation_id)
                )
            """
        }
        
        table_order = ['workclass', 'education', 'marital_status', 'occupation', 
                      'relationship', 'race', 'country', 'salary_range', 'person']
        
        try:
            with self.db as db:
                for table_name in table_order:
                    if db.execute_update(tables[table_name]):
                        logger.info(f"✅ Tabela '{table_name}' criada/verificada")
                    else:
                        logger.error(f"❌ Erro ao criar tabela '{table_name}'")
                        return False
            
            logger.info("🎉 Estrutura do banco criada com sucesso!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao criar estrutura: {e}")
            return False
    
    def migrate_csv_to_database(self, csv_path: str) -> bool:
        """Migrar dados CSV para banco"""
        
        try:
            # Carregar CSV
            df = pd.read_csv(csv_path)
            logger.info(f"📊 CSV carregado: {len(df)} registros")
            
            # Limpar dados
            df = self._clean_data(df)
            
            # Criar estrutura
            if not self.create_database_structure():
                return False
            
            # Popular lookup tables
            if not self._populate_lookup_tables(df):
                return False
            
            # Migrar dados principais
            if not self._migrate_person_data(df):
                return False
            
            logger.info("🎉 Migração concluída com sucesso!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro na migração: {e}")
            return False
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpar dados antes da migração"""
        
        # Substituir valores problemáticos
        df = df.replace(['?', 'Unknown', 'nan', np.nan], None)
        
        # Limpar strings
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace('None', None)
        
        logger.info("✅ Dados limpos para migração")
        return df
    
    def _populate_lookup_tables(self, df: pd.DataFrame) -> bool:
        """Popular tabelas de lookup"""
        
        lookup_mappings = {
            'workclass': df['workclass'].dropna().unique(),
            'marital_status': df['marital-status'].dropna().unique(),
            'occupation': df['occupation'].dropna().unique(),
            'relationship': df['relationship'].dropna().unique(),
            'race': df['race'].dropna().unique(),
            'country': df['native-country'].dropna().unique()
        }
        
        # Education (com education_num)
        education_data = df[['education', 'education-num']].dropna().drop_duplicates()
        
        # Salary ranges
        salary_ranges = {
            '<=50K': (0, 50000),
            '>50K': (50001, None)
        }
        
        try:
            with self.db as db:
                # Popular tabelas simples
                for table, values in lookup_mappings.items():
                    for value in values:
                        sql = f"INSERT IGNORE INTO {table} (name) VALUES (%s)"
                        db.execute_update(sql, (value,))
                
                # Popular education
                for _, row in education_data.iterrows():
                    sql = "INSERT IGNORE INTO education (name, education_num) VALUES (%s, %s)"
                    db.execute_update(sql, (row['education'], row['education-num']))
                
                # Popular salary_range
                for salary_name in df['salary'].dropna().unique():
                    min_amt, max_amt = salary_ranges.get(salary_name, (None, None))
                    sql = "INSERT IGNORE INTO salary_range (name, min_amount, max_amount) VALUES (%s, %s, %s)"
                    db.execute_update(sql, (salary_name, min_amt, max_amt))
            
            logger.info("✅ Tabelas de lookup populadas")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao popular lookup tables: {e}")
            return False
    
    def _migrate_person_data(self, df: pd.DataFrame) -> bool:
        """Migrar dados da tabela person"""
        
        # Obter mapeamentos
        mappings = self._get_lookup_mappings()
        if not mappings:
            return False
        
        success_count = 0
        error_count = 0
        
        try:
            with self.db as db:
                for index, row in df.iterrows():
                    try:
                        # Mapear IDs
                        person_data = (
                            int(row['age']) if pd.notna(row['age']) else None,
                            int(row['fnlwgt']) if pd.notna(row['fnlwgt']) else None,
                            mappings['workclass'].get(row.get('workclass')),
                            mappings['education'].get(row.get('education')),
                            mappings['marital_status'].get(row.get('marital-status')),
                            mappings['occupation'].get(row.get('occupation')),
                            mappings['relationship'].get(row.get('relationship')),
                            mappings['race'].get(row.get('race')),
                            row['sex'] if pd.notna(row['sex']) else None,
                            float(row['capital-gain']) if pd.notna(row['capital-gain']) else 0,
                            float(row['capital-loss']) if pd.notna(row['capital-loss']) else 0,
                            int(row['hours-per-week']) if pd.notna(row['hours-per-week']) else None,
                            mappings['country'].get(row.get('native-country')),
                            mappings['salary_range'].get(row.get('salary'))
                        )
                        
                        sql = """
                            INSERT INTO person 
                            (age, fnlwgt, workclass_id, education_id, marital_status_id, 
                             occupation_id, relationship_id, race_id, sex, capital_gain, 
                             capital_loss, hours_per_week, native_country_id, salary_range_id)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """
                        
                        if db.execute_update(sql, person_data):
                            success_count += 1
                        else:
                            error_count += 1
                        
                        if success_count % 1000 == 0:
                            logger.info(f"📊 {success_count} registros migrados...")
                            
                    except Exception as e:
                        error_count += 1
                        if error_count % 100 == 0:
                            logger.warning(f"⚠️ {error_count} erros até agora")
                        continue
            
            logger.info(f"✅ Migração concluída: {success_count} sucessos, {error_count} erros")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro na migração de dados: {e}")
            return False
    
    def _get_lookup_mappings(self) -> dict:
        """Obter mapeamentos ID das tabelas de lookup"""
        mappings = {}
        
        try:
            with self.db as db:
                tables = ['workclass', 'education', 'marital_status', 'occupation', 
                         'relationship', 'race', 'country', 'salary_range']
                
                for table in tables:
                    results = db.execute_query(f"SELECT id, name FROM {table}")
                    if results:
                        mappings[table] = {row['name']: row['id'] for row in results}
                    else:
                        mappings[table] = {}
            
            return mappings
            
        except Exception as e:
            logger.error(f"❌ Erro ao obter mapeamentos: {e}")
            return {}