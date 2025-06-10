"""Utilitários para pipelines"""

import joblib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def load_real_feature_names(expected_length):
    """Carregar nomes reais das features"""
    try:
        # Tentar carregar feature_info
        feature_info_paths = [
            Path("data/processed/feature_info.joblib"),
            Path("bkp/feature_info.joblib"),
            Path("feature_info.joblib")
        ]
        
        for path in feature_info_paths:
            if path.exists():
                feature_info = joblib.load(path)
                if 'feature_names' in feature_info:
                    feature_names = feature_info['feature_names']
                    if len(feature_names) >= expected_length:
                        logger.info(f"✅ Nomes reais carregados: {len(feature_names)} features")
                        return feature_names[:expected_length]
        
        # Fallback para nomes baseados no dataset
        logger.warning("⚠️ feature_info.joblib não encontrado. Usando nomes baseados no dataset.")
        return get_dataset_based_feature_names(expected_length)
        
    except Exception as e:
        logger.error(f"❌ Erro ao carregar nomes das features: {e}")
        return get_dataset_based_feature_names(expected_length)

def get_dataset_based_feature_names(expected_length):
    """Gerar nomes baseados na estrutura do dataset"""
    numeric_features = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
    
    categorical_features = [
        'workclass_Federal_gov', 'workclass_Local_gov', 'workclass_Private', 
        'workclass_Self_emp_inc', 'workclass_Self_emp_not_inc', 'workclass_State_gov', 'workclass_Without_pay',
        'education_11th', 'education_12th', 'education_1st_4th', 'education_5th_6th', 'education_7th_8th', 
        'education_9th', 'education_Assoc_acdm', 'education_Assoc_voc', 'education_Bachelors', 
        'education_Doctorate', 'education_HS_grad', 'education_Masters', 'education_Preschool', 
        'education_Prof_school', 'education_Some_college',
        'marital_status_Divorced', 'marital_status_Married_AF_spouse', 'marital_status_Married_civ_spouse',
        'marital_status_Married_spouse_absent', 'marital_status_Never_married', 'marital_status_Separated', 
        'marital_status_Widowed',
        'occupation_Adm_clerical', 'occupation_Armed_Forces', 'occupation_Craft_repair', 
        'occupation_Exec_managerial', 'occupation_Farming_fishing', 'occupation_Handlers_cleaners',
        'occupation_Machine_op_inspct', 'occupation_Other_service', 'occupation_Priv_house_serv',
        'occupation_Prof_specialty', 'occupation_Protective_serv', 'occupation_Sales', 
        'occupation_Tech_support', 'occupation_Transport_moving',
        'relationship_Husband', 'relationship_Not_in_family', 'relationship_Other_relative',
        'relationship_Own_child', 'relationship_Unmarried', 'relationship_Wife',
        'race_Amer_Indian_Eskimo', 'race_Asian_Pac_Islander', 'race_Black', 'race_Other', 'race_White',
        'sex_Male',
        'native_country_Canada', 'native_country_China', 'native_country_Cuba', 'native_country_England',
        'native_country_Germany', 'native_country_India', 'native_country_Iran', 'native_country_Italy',
        'native_country_Jamaica', 'native_country_Japan', 'native_country_Mexico', 'native_country_Philippines',
        'native_country_Poland', 'native_country_Portugal', 'native_country_Puerto_Rico', 'native_country_South',
        'native_country_Taiwan', 'native_country_United_States', 'native_country_Vietnam'
    ]
    
    all_features = numeric_features + categorical_features
    
    if len(all_features) >= expected_length:
        return all_features[:expected_length]
    else:
        while len(all_features) < expected_length:
            all_features.append(f"Feature_{len(all_features)}")
        return all_features

def check_data_structure():
    """Verificar estrutura de dados"""
    project_root = Path(__file__).parent.parent.parent
    data_file = project_root / "data" / "raw" / "4-Carateristicas_salario.csv"
    
    if not data_file.exists():
        logger.warning("⚠️ Estrutura de dados não configurada!")
        logger.info("🔧 Execute: python setup_data_structure.py")
        
        try:
            import setup_scripts.setup_data_structure as setup_data_structure
            setup_data_structure.setup_data_structure()
            if data_file.exists():
                logger.info("✅ Estrutura configurada automaticamente!")
            else:
                raise FileNotFoundError("Configuração automática falhou")
        except Exception as e:
            logger.error(f"❌ Erro: {e}")
            logger.error("Manual: Coloque '4-Carateristicas_salario.csv' em 'data/raw/'")
            raise