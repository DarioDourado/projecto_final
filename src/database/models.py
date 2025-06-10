"""Modelos SQL - Versão com Mapeamento ML Compatível"""

import pandas as pd
import logging
from typing import Optional, List, Dict, Any, Tuple
from .connection import DatabaseConnection

logger = logging.getLogger(__name__)

class SalaryAnalysisSQL:
    """Modelo principal para análises SQL sem dependência de CSV"""
    
    def __init__(self, db_connection: DatabaseConnection = None):
        self.db = db_connection or DatabaseConnection()
    
    def get_dataset_for_ml(self) -> Optional[pd.DataFrame]:
        """Obter dataset completo formatado para ML com mapeamento correto"""
        query = """
        SELECT 
            p.age,
            p.fnlwgt,
            w.name as workclass,
            e.name as education,
            e.education_num as 'education-num',
            ms.name as 'marital-status',
            o.name as occupation,
            r.name as relationship,
            rc.name as race,
            p.sex,
            p.capital_gain as 'capital-gain',
            p.capital_loss as 'capital-loss',
            p.hours_per_week as 'hours-per-week',
            c.name as 'native-country',
            sr.name as salary  -- Manter como 'salary' para o pipeline SQL detectar
        FROM person p
        LEFT JOIN workclass w ON p.workclass_id = w.id
        LEFT JOIN education e ON p.education_id = e.id
        LEFT JOIN marital_status ms ON p.marital_status_id = ms.id
        LEFT JOIN occupation o ON p.occupation_id = o.id
        LEFT JOIN relationship r ON p.relationship_id = r.id
        LEFT JOIN race rc ON p.race_id = rc.id
        LEFT JOIN country c ON p.native_country_id = c.id
        LEFT JOIN salary_range sr ON p.salary_range_id = sr.id
        WHERE sr.name IS NOT NULL
        ORDER BY p.id
        """
        
        try:
            with self.db as db:
                results = db.execute_query(query)
                if results:
                    df = pd.DataFrame(results)
                    logger.info(f"✅ Dataset ML: {len(df)} registros carregados")
                    logger.info(f"📋 Colunas SQL: {list(df.columns)}")
                    return df
                else:
                    logger.warning("⚠️ Nenhum dado encontrado para ML")
                    return None
        except Exception as e:
            logger.error(f"❌ Erro ao carregar dataset ML: {e}")
            return None
    
    def get_salary_distribution(self) -> Dict[str, Any]:
        """Distribuição de salários via SQL"""
        query = """
        SELECT 
            sr.name as salary_range,
            COUNT(*) as count,
            ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM person), 2) as percentage
        FROM person p
        JOIN salary_range sr ON p.salary_range_id = sr.id
        GROUP BY sr.name, sr.id
        ORDER BY sr.id
        """
        
        try:
            with self.db as db:
                return db.execute_query(query)
        except Exception as e:
            logger.error(f"❌ Erro na distribuição salarial: {e}")
            return []
    
    def get_age_salary_analysis(self) -> List[Dict]:
        """Análise idade vs salário"""
        query = """
        SELECT 
            CASE 
                WHEN age < 25 THEN '18-24'
                WHEN age < 35 THEN '25-34'
                WHEN age < 45 THEN '35-44'
                WHEN age < 55 THEN '45-54'
                ELSE '55+'
            END as age_group,
            sr.name as salary_range,
            COUNT(*) as count,
            AVG(age) as avg_age
        FROM person p
        JOIN salary_range sr ON p.salary_range_id = sr.id
        GROUP BY age_group, sr.name
        ORDER BY 
            CASE age_group
                WHEN '18-24' THEN 1
                WHEN '25-34' THEN 2
                WHEN '35-44' THEN 3
                WHEN '45-54' THEN 4
                ELSE 5
            END, sr.name
        """
        
        try:
            with self.db as db:
                return db.execute_query(query)
        except Exception as e:
            logger.error(f"❌ Erro na análise idade-salário: {e}")
            return []
    
    def get_education_salary_analysis(self) -> List[Dict]:
        """Análise educação vs salário"""
        query = """
        SELECT 
            e.name as education,
            e.education_num,
            sr.name as salary_range,
            COUNT(*) as count,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(PARTITION BY e.name), 2) as percentage_within_education
        FROM person p
        JOIN education e ON p.education_id = e.id
        JOIN salary_range sr ON p.salary_range_id = sr.id
        GROUP BY e.name, e.education_num, sr.name
        ORDER BY e.education_num, sr.name
        """
        
        try:
            with self.db as db:
                return db.execute_query(query)
        except Exception as e:
            logger.error(f"❌ Erro na análise educação-salário: {e}")
            return []
    
    def get_occupation_workclass_analysis(self) -> List[Dict]:
        """Análise ocupação vs classe trabalhadora"""
        query = """
        SELECT 
            o.name as occupation,
            w.name as workclass,
            sr.name as salary_range,
            COUNT(*) as count,
            AVG(p.hours_per_week) as avg_hours_per_week,
            AVG(p.age) as avg_age
        FROM person p
        JOIN occupation o ON p.occupation_id = o.id
        JOIN workclass w ON p.workclass_id = w.id
        JOIN salary_range sr ON p.salary_range_id = sr.id
        GROUP BY o.name, w.name, sr.name
        HAVING count >= 10  -- Filtrar grupos pequenos
        ORDER BY count DESC
        """
        
        try:
            with self.db as db:
                return db.execute_query(query)
        except Exception as e:
            logger.error(f"❌ Erro na análise ocupação-workclass: {e}")
            return []
    
    def get_clustering_features_data(self) -> Optional[pd.DataFrame]:
        """Dados para clustering (apenas features numéricas)"""
        query = """
        SELECT 
            p.age,
            p.fnlwgt,
            e.education_num,
            p.capital_gain,
            p.capital_loss,
            p.hours_per_week,
            sr.name as salary_target
        FROM person p
        JOIN education e ON p.education_id = e.id
        JOIN salary_range sr ON p.salary_range_id = sr.id
        WHERE p.age IS NOT NULL 
            AND e.education_num IS NOT NULL 
            AND p.hours_per_week IS NOT NULL
        """
        
        try:
            with self.db as db:
                results = db.execute_query(query)
                if results:
                    df = pd.DataFrame(results)
                    logger.info(f"✅ Dados clustering: {len(df)} registros")
                    return df
                return None
        except Exception as e:
            logger.error(f"❌ Erro dados clustering: {e}")
            return None
    
    def get_association_rules_data(self) -> Optional[pd.DataFrame]:
        """Dados categóricos para regras de associação"""
        query = """
        SELECT 
            CONCAT('age_', 
                CASE 
                    WHEN age < 30 THEN 'jovem'
                    WHEN age < 50 THEN 'adulto'
                    ELSE 'senior'
                END
            ) as age_category,
            CONCAT('workclass_', REPLACE(w.name, ' ', '_')) as workclass_cat,
            CONCAT('education_', REPLACE(e.name, ' ', '_')) as education_cat,
            CONCAT('marital_', REPLACE(ms.name, ' ', '_')) as marital_cat,
            CONCAT('occupation_', REPLACE(o.name, ' ', '_')) as occupation_cat,
            CONCAT('sex_', p.sex) as sex_cat,
            CONCAT('hours_', 
                CASE 
                    WHEN p.hours_per_week < 35 THEN 'part_time'
                    WHEN p.hours_per_week <= 45 THEN 'normal'
                    ELSE 'overtime'
                END
            ) as hours_cat,
            CONCAT('salary_', REPLACE(sr.name, ' ', '_')) as salary_cat
        FROM person p
        LEFT JOIN workclass w ON p.workclass_id = w.id
        LEFT JOIN education e ON p.education_id = e.id
        LEFT JOIN marital_status ms ON p.marital_status_id = ms.id
        LEFT JOIN occupation o ON p.occupation_id = o.id
        LEFT JOIN salary_range sr ON p.salary_range_id = sr.id
        WHERE w.name IS NOT NULL 
            AND e.name IS NOT NULL 
            AND sr.name IS NOT NULL
        """
        
        try:
            with self.db as db:
                results = db.execute_query(query)
                if results:
                    df = pd.DataFrame(results)
                    logger.info(f"✅ Dados associação: {len(df)} registros")
                    return df
                return None
        except Exception as e:
            logger.error(f"❌ Erro dados associação: {e}")
            return None
    
    def get_advanced_statistics(self) -> Dict[str, Any]:
        """Estatísticas avançadas via SQL"""
        queries = {
            'total_records': "SELECT COUNT(*) as total FROM person",
            
            'salary_stats': """
                SELECT 
                    sr.name as salary_range,
                    COUNT(*) as count,
                    AVG(p.age) as avg_age,
                    AVG(p.hours_per_week) as avg_hours,
                    AVG(e.education_num) as avg_education
                FROM person p
                JOIN salary_range sr ON p.salary_range_id = sr.id
                JOIN education e ON p.education_id = e.id
                GROUP BY sr.name
            """,
            
            'gender_analysis': """
                SELECT 
                    p.sex,
                    sr.name as salary_range,
                    COUNT(*) as count,
                    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(PARTITION BY p.sex), 2) as percentage
                FROM person p
                JOIN salary_range sr ON p.salary_range_id = sr.id
                GROUP BY p.sex, sr.name
                ORDER BY p.sex, sr.name
            """,
            
            'top_occupations_high_salary': """
                SELECT 
                    o.name as occupation,
                    COUNT(*) as high_salary_count,
                    ROUND(AVG(p.age), 1) as avg_age,
                    ROUND(AVG(p.hours_per_week), 1) as avg_hours
                FROM person p
                JOIN occupation o ON p.occupation_id = o.id
                JOIN salary_range sr ON p.salary_range_id = sr.id
                WHERE sr.name = '>50K'
                GROUP BY o.name
                HAVING high_salary_count >= 50
                ORDER BY high_salary_count DESC
                LIMIT 10
            """,
            
            'education_impact': """
                SELECT 
                    e.education_num,
                    e.name as education_name,
                    COUNT(*) as total_count,
                    SUM(CASE WHEN sr.name = '>50K' THEN 1 ELSE 0 END) as high_salary_count,
                    ROUND(
                        SUM(CASE WHEN sr.name = '>50K' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 
                        2
                    ) as high_salary_percentage
                FROM person p
                JOIN education e ON p.education_id = e.id
                JOIN salary_range sr ON p.salary_range_id = sr.id
                GROUP BY e.education_num, e.name
                ORDER BY e.education_num
            """
        }
        
        stats = {}
        
        try:
            with self.db as db:
                for key, query in queries.items():
                    result = db.execute_query(query)
                    if result:
                        if key == 'total_records':
                            stats[key] = result[0]['total']
                        else:
                            stats[key] = result
        except Exception as e:
            logger.error(f"❌ Erro nas estatísticas avançadas: {e}")
        
        return stats
    
    def create_custom_view(self, view_name: str, query: str) -> bool:
        """Criar view customizada"""
        try:
            with self.db as db:
                # Remover view se existir
                db.execute_update(f"DROP VIEW IF EXISTS {view_name}")
                
                # Criar nova view
                create_query = f"CREATE VIEW {view_name} AS {query}"
                success = db.execute_update(create_query)
                
                if success:
                    logger.info(f"✅ View '{view_name}' criada com sucesso")
                return success
        except Exception as e:
            logger.error(f"❌ Erro ao criar view '{view_name}': {e}")
            return False
    
    def execute_custom_analysis(self, query: str) -> Optional[pd.DataFrame]:
        """Executar análise SQL customizada"""
        try:
            with self.db as db:
                results = db.execute_query(query)
                if results:
                    return pd.DataFrame(results)
                return None
        except Exception as e:
            logger.error(f"❌ Erro na análise customizada: {e}")
            return None

class SQLQueryBuilder:
    """Construtor de queries SQL para análises específicas"""
    
    @staticmethod
    def build_filtered_dataset_query(filters: Dict[str, Any]) -> str:
        """Construir query com filtros dinâmicos"""
        base_query = """
        SELECT 
            p.age,
            w.name as workclass,
            e.name as education,
            e.education_num as 'education-num',
            ms.name as 'marital-status',
            o.name as occupation,
            r.name as relationship,
            rc.name as race,
            p.sex,
            p.capital_gain as 'capital-gain',
            p.capital_loss as 'capital-loss',
            p.hours_per_week as 'hours-per-week',
            c.name as 'native-country',
            sr.name as salary
        FROM person p
        LEFT JOIN workclass w ON p.workclass_id = w.id
        LEFT JOIN education e ON p.education_id = e.id
        LEFT JOIN marital_status ms ON p.marital_status_id = ms.id
        LEFT JOIN occupation o ON p.occupation_id = o.id
        LEFT JOIN relationship r ON p.relationship_id = r.id
        LEFT JOIN race rc ON p.race_id = rc.id
        LEFT JOIN country c ON p.native_country_id = c.id
        LEFT JOIN salary_range sr ON p.salary_range_id = sr.id
        WHERE 1=1
        """
        
        conditions = []
        
        # Filtros de idade
        if 'min_age' in filters:
            conditions.append(f"AND p.age >= {filters['min_age']}")
        if 'max_age' in filters:
            conditions.append(f"AND p.age <= {filters['max_age']}")
        
        # Filtros categóricos
        if 'salary' in filters and filters['salary']:
            salary_list = "', '".join(filters['salary'])
            conditions.append(f"AND sr.name IN ('{salary_list}')")
        
        if 'sex' in filters and filters['sex']:
            sex_list = "', '".join(filters['sex'])
            conditions.append(f"AND p.sex IN ('{sex_list}')")
        
        if 'workclass' in filters and filters['workclass']:
            workclass_list = "', '".join(filters['workclass'])
            conditions.append(f"AND w.name IN ('{workclass_list}')")
        
        # Adicionar condições
        if conditions:
            base_query += " " + " ".join(conditions)
        
        base_query += " ORDER BY p.id"
        
        return base_query
    
    @staticmethod
    def build_correlation_analysis_query() -> str:
        """Query para análise de correlação"""
        return """
        SELECT 
            p.age,
            p.fnlwgt,
            e.education_num,
            p.capital_gain,
            p.capital_loss,
            p.hours_per_week,
            CASE WHEN sr.name = '>50K' THEN 1 ELSE 0 END as salary_numeric
        FROM person p
        JOIN education e ON p.education_id = e.id
        JOIN salary_range sr ON p.salary_range_id = sr.id
        WHERE p.age IS NOT NULL 
            AND e.education_num IS NOT NULL 
            AND p.hours_per_week IS NOT NULL
        """
    
    @staticmethod
    def build_feature_importance_query() -> str:
        """Query para análise de importância de features"""
        return """
        SELECT 
            'age' as feature,
            AVG(CASE WHEN sr.name = '>50K' THEN p.age ELSE 0 END) as high_salary_avg,
            AVG(CASE WHEN sr.name = '<=50K' THEN p.age ELSE 0 END) as low_salary_avg,
            ABS(AVG(CASE WHEN sr.name = '>50K' THEN p.age ELSE 0 END) - 
                AVG(CASE WHEN sr.name = '<=50K' THEN p.age ELSE 0 END)) as difference
        FROM person p
        JOIN salary_range sr ON p.salary_range_id = sr.id
        
        UNION ALL
        
        SELECT 
            'education_num' as feature,
            AVG(CASE WHEN sr.name = '>50K' THEN e.education_num ELSE 0 END) as high_salary_avg,
            AVG(CASE WHEN sr.name = '<=50K' THEN e.education_num ELSE 0 END) as low_salary_avg,
            ABS(AVG(CASE WHEN sr.name = '>50K' THEN e.education_num ELSE 0 END) - 
                AVG(CASE WHEN sr.name = '<=50K' THEN e.education_num ELSE 0 END)) as difference
        FROM person p
        JOIN education e ON p.education_id = e.id
        JOIN salary_range sr ON p.salary_range_id = sr.id
        
        UNION ALL
        
        SELECT 
            'hours_per_week' as feature,
            AVG(CASE WHEN sr.name = '>50K' THEN p.hours_per_week ELSE 0 END) as high_salary_avg,
            AVG(CASE WHEN sr.name = '<=50K' THEN p.hours_per_week ELSE 0 END) as low_salary_avg,
            ABS(AVG(CASE WHEN sr.name = '>50K' THEN p.hours_per_week ELSE 0 END) - 
                AVG(CASE WHEN sr.name = '<=50K' THEN p.hours_per_week ELSE 0 END)) as difference
        FROM person p
        JOIN salary_range sr ON p.salary_range_id = sr.id
        
        ORDER BY difference DESC
        """