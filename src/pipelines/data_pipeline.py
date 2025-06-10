"""Pipeline de dados SQL-Only - Sem dependência de CSV"""

import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)

class DataPipelineSQL:
    """Pipeline de dados exclusivo para banco de dados"""
    
    def __init__(self, force_migration: bool = False):
        self.df = None
        self.force_migration = force_migration
        
        # Verificar se banco está configurado
        if not self._check_database_config():
            raise ValueError("❌ Configuração de banco de dados obrigatória!")
    
    def _check_database_config(self) -> bool:
        """Verificar se configuração de banco existe"""
        required_vars = ['DB_HOST', 'DB_NAME', 'DB_USER', 'DB_PASSWORD']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            logger.error(f"❌ Variáveis de ambiente ausentes: {missing_vars}")
            logger.info("💡 Configure: export DB_HOST=localhost DB_NAME=salary_analysis ...")
            return False
        
        return True
    
    def run(self) -> Optional[pd.DataFrame]:
        """Executar pipeline de dados SQL"""
        try:
            logger.info("🗄️ Carregando dados do banco de dados")
            
            # Verificar se dados existem no banco
            if not self._check_database_populated():
                if self.force_migration or self._ask_for_migration():
                    logger.info("🔄 Executando migração automática...")
                    if not self._run_migration():
                        raise ValueError("❌ Migração falhou - dados não disponíveis")
                else:
                    raise ValueError("❌ Banco vazio e migração cancelada")
            
            # Carregar dados do banco
            self.df = self._load_from_database()
            
            if self.df is not None and len(self.df) > 0:
                # Validação e limpeza
                self.df = self._validate_and_clean(self.df)
                logger.info(f"✅ Dados processados: {len(self.df)} registros")
                return self.df
            else:
                raise ValueError("❌ Nenhum dado válido encontrado no banco")
                
        except Exception as e:
            logger.error(f"❌ Erro no pipeline SQL: {e}")
            raise
    
    def _check_database_populated(self) -> bool:
        """Verificar se banco tem dados"""
        try:
            from src.database.models import SalaryAnalysisSQL
            
            sql_model = SalaryAnalysisSQL()
            stats = sql_model.get_advanced_statistics()
            
            total_records = stats.get('total_records', 0)
            
            if total_records > 0:
                logger.info(f"✅ Banco populado: {total_records} registros")
                return True
            else:
                logger.warning("⚠️ Banco vazio ou não acessível")
                return False
                
        except Exception as e:
            logger.warning(f"⚠️ Erro ao verificar banco: {e}")
            return False
    
    def _ask_for_migration(self) -> bool:
        """Perguntar se deve executar migração"""
        logger.info("❓ Banco vazio. Executar migração automática?")
        logger.info("💡 Para migração automática, execute: python main.py --migrate")
        
        # Em modo automatizado, sempre migrar
        if os.getenv('AUTO_MIGRATE', 'false').lower() == 'true':
            return True
        
        # Verificar se existe CSV para migrar
        csv_paths = [
            Path("data/raw/4-Carateristicas_salario.csv"),
            Path("bkp/4-Carateristicas_salario.csv"),
            Path("4-Carateristicas_salario.csv")
        ]
        
        csv_found = any(path.exists() for path in csv_paths)
        
        if csv_found:
            logger.info("📄 Arquivo CSV encontrado - executando migração automática")
            return True
        else:
            logger.error("❌ Nenhum CSV encontrado para migração")
            return False
    
    def _run_migration(self) -> bool:
        """Executar migração se necessário"""
        try:
            from src.database.migration import DatabaseMigrator
            
            # Procurar CSV
            csv_paths = [
                Path("data/raw/4-Carateristicas_salario.csv"),
                Path("bkp/4-Carateristicas_salario.csv"),
                Path("4-Carateristicas_salario.csv")
            ]
            
            csv_path = None
            for path in csv_paths:
                if path.exists():
                    csv_path = str(path)
                    break
            
            if not csv_path:
                logger.error("❌ Nenhum CSV encontrado para migração")
                return False
            
            logger.info(f"🚀 Migrando {csv_path} para banco...")
            
            migrator = DatabaseMigrator()
            success = migrator.migrate_csv_to_database(csv_path)
            
            if success:
                logger.info("✅ Migração concluída!")
            else:
                logger.error("❌ Migração falhou!")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Erro na migração: {e}")
            return False
    
    def _load_from_database(self) -> Optional[pd.DataFrame]:
        """Carregar dados do banco de dados"""
        try:
            from src.database.models import SalaryAnalysisSQL
            
            sql_model = SalaryAnalysisSQL()
            df = sql_model.get_dataset_for_ml()
            
            if df is not None and len(df) > 0:
                logger.info(f"✅ {len(df)} registros carregados do banco")
                return df
            else:
                logger.error("❌ Nenhum dado encontrado no banco")
                return None
                
        except Exception as e:
            logger.error(f"❌ Erro ao carregar do banco: {e}")
            return None
    
    def _validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validação e limpeza específica para dados SQL"""
        try:
            original_len = len(df)
            
            # Remover registros sem target
            df = df.dropna(subset=['salary'])
            
            # Validar colunas essenciais
            required_columns = ['age', 'salary', 'education-num']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.warning(f"⚠️ Colunas ausentes: {missing_columns}")
            
            # Limpeza de dados categóricos
            categorical_columns = ['workclass', 'education', 'marital-status', 
                                 'occupation', 'relationship', 'race', 'sex', 'native-country']
            
            for col in categorical_columns:
                if col in df.columns:
                    df[col] = df[col].fillna('Unknown')
                    df[col] = df[col].replace(['None', 'nan', ''], 'Unknown')
            
            # Limpeza de dados numéricos
            numeric_columns = ['age', 'fnlwgt', 'education-num', 'capital-gain', 
                             'capital-loss', 'hours-per-week']
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].fillna(df[col].median())
            
            # Validações de range
            if 'age' in df.columns:
                df = df[(df['age'] >= 17) & (df['age'] <= 100)]
            
            if 'hours-per-week' in df.columns:
                df = df[(df['hours-per-week'] >= 1) & (df['hours-per-week'] <= 99)]
            
            cleaned_len = len(df)
            removed = original_len - cleaned_len
            
            if removed > 0:
                logger.info(f"🧹 Limpeza: {removed} registros removidos ({removed/original_len*100:.1f}%)")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Erro na validação: {e}")
            return df
    
    def get_filtered_data(self, filters: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Obter dados filtrados via SQL"""
        try:
            from src.database.models import SQLQueryBuilder, SalaryAnalysisSQL
            
            query = SQLQueryBuilder.build_filtered_dataset_query(filters)
            
            sql_model = SalaryAnalysisSQL()
            df = sql_model.execute_custom_analysis(query)
            
            if df is not None:
                df = self._validate_and_clean(df)
                logger.info(f"✅ Dados filtrados: {len(df)} registros")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Erro ao filtrar dados: {e}")
            return None
    
    def get_statistics_from_sql(self) -> Dict[str, Any]:
        """Obter estatísticas diretamente do SQL"""
        try:
            from src.database.models import SalaryAnalysisSQL
            
            sql_model = SalaryAnalysisSQL()
            return sql_model.get_advanced_statistics()
            
        except Exception as e:
            logger.error(f"❌ Erro nas estatísticas SQL: {e}")
            return {}
    
    def create_analysis_views(self) -> bool:
        """Criar views SQL para análises específicas"""
        try:
            from src.database.models import SalaryAnalysisSQL
            
            sql_model = SalaryAnalysisSQL()
            
            views = {
                'high_earners_view': """
                    SELECT 
                        p.age,
                        w.name as workclass,
                        e.name as education,
                        e.education_num,
                        o.name as occupation,
                        p.sex,
                        p.hours_per_week,
                        p.capital_gain
                    FROM person p
                    JOIN workclass w ON p.workclass_id = w.id
                    JOIN education e ON p.education_id = e.id
                    JOIN occupation o ON p.occupation_id = o.id
                    JOIN salary_range sr ON p.salary_range_id = sr.id
                    WHERE sr.name = '>50K'
                """,
                
                'education_analysis_view': """
                    SELECT 
                        e.name as education,
                        e.education_num,
                        COUNT(*) as total_count,
                        AVG(p.age) as avg_age,
                        AVG(p.hours_per_week) as avg_hours,
                        SUM(CASE WHEN sr.name = '>50K' THEN 1 ELSE 0 END) as high_salary_count,
                        ROUND(
                            SUM(CASE WHEN sr.name = '>50K' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 
                            2
                        ) as high_salary_percentage
                    FROM person p
                    JOIN education e ON p.education_id = e.id
                    JOIN salary_range sr ON p.salary_range_id = sr.id
                    GROUP BY e.name, e.education_num
                    ORDER BY e.education_num
                """,
                
                'clustering_features_view': """
                    SELECT 
                        p.id,
                        p.age,
                        e.education_num,
                        p.hours_per_week,
                        p.capital_gain,
                        p.capital_loss,
                        CASE WHEN sr.name = '>50K' THEN 1 ELSE 0 END as salary_numeric
                    FROM person p
                    JOIN education e ON p.education_id = e.id
                    JOIN salary_range sr ON p.salary_range_id = sr.id
                    WHERE p.age IS NOT NULL 
                        AND e.education_num IS NOT NULL 
                        AND p.hours_per_week IS NOT NULL
                """
            }
            
            success_count = 0
            for view_name, query in views.items():
                if sql_model.create_custom_view(view_name, query):
                    success_count += 1
            
            logger.info(f"✅ {success_count}/{len(views)} views criadas")
            return success_count == len(views)
            
        except Exception as e:
            logger.error(f"❌ Erro ao criar views: {e}")
            return False