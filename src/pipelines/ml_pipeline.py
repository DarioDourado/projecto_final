"""Pipeline de Machine Learning - Versão SQL Compatível"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import logging
import joblib
from pathlib import Path

class MLPipeline:
    """Pipeline de ML compatível com estrutura SQL"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.results = {}
        self.label_encoders = {}
        self.scaler = None
        
    def run(self, df):
        """Executar pipeline ML com dados SQL"""
        self.logger.info("🤖 Iniciando Pipeline de Machine Learning (SQL)")
        
        try:
            # Validar dados de entrada
            if df is None or len(df) == 0:
                raise ValueError("Dataset vazio")
            
            self.logger.info(f"📊 Dataset original: {len(df)} registros")
            self.logger.info(f"📋 Colunas disponíveis: {list(df.columns)}")
            
            # Preparar dados com mapeamento SQL
            X, y = self._prepare_data_sql(df)
            
            if X is None or y is None:
                raise ValueError("Falha na preparação dos dados SQL")
            
            self.logger.info(f"✅ Dados preparados: {X.shape[0]} registros, {X.shape[1]} features")
            
            # Divisão treino/teste
            X_train, X_test, y_train, y_test = self._safe_train_test_split(X, y)
            
            # Normalização
            X_train_scaled, X_test_scaled = self._scale_features(X_train, X_test)
            
            # Treinar modelos
            self._train_models_sql(X_train_scaled, X_test_scaled, y_train, y_test)
            
            return self.models, self.results
            
        except Exception as e:
            self.logger.error(f"❌ Erro no pipeline ML: {e}")
            raise
    
    def _prepare_data_sql(self, df):
        """Preparar dados com mapeamento de colunas SQL"""
        try:
            # Fazer cópia para não alterar original
            df_work = df.copy()
            
            self.logger.info("🔧 Preparando dados com estrutura SQL...")
            
            # Mapear colunas SQL para formato ML
            column_mapping = {
                'salary': 'income',  # Mapear salary -> income
                'education-num': 'education_num',
                'marital-status': 'marital_status',
                'capital-gain': 'capital_gain',
                'capital-loss': 'capital_loss',
                'hours-per-week': 'hours_per_week',
                'native-country': 'native_country'
            }
            
            # Aplicar mapeamento
            for old_col, new_col in column_mapping.items():
                if old_col in df_work.columns:
                    df_work[new_col] = df_work[old_col]
                    if old_col != new_col:  # Só remover se for diferente
                        df_work.drop(old_col, axis=1, inplace=True)
            
            self.logger.info(f"📋 Colunas após mapeamento: {list(df_work.columns)}")
            
            # Verificar colunas obrigatórias após mapeamento
            required_columns = ['age', 'education', 'workclass', 'occupation', 'income']
            missing_cols = [col for col in required_columns if col not in df_work.columns]
            
            if missing_cols:
                self.logger.error(f"❌ Colunas obrigatórias ausentes após mapeamento: {missing_cols}")
                self.logger.info(f"💡 Colunas disponíveis: {list(df_work.columns)}")
                return None, None
            
            # Remover valores nulos
            df_clean = df_work.dropna(subset=['income'])
            self.logger.info(f"📋 Após limpeza: {len(df_clean)} registros")
            
            if len(df_clean) == 0:
                raise ValueError("Nenhum registro válido após limpeza")
            
            # Preparar features (X) - usar colunas disponíveis inteligentemente
            available_columns = list(df_clean.columns)
            
            # Colunas prioritárias para ML
            priority_columns = [
                'age', 'education', 'workclass', 'occupation', 
                'education_num', 'marital_status', 'relationship', 
                'race', 'sex', 'capital_gain', 'capital_loss', 
                'hours_per_week', 'native_country'
            ]
            
            # Selecionar apenas colunas que existem
            feature_columns = [col for col in priority_columns if col in available_columns]
            
            # Garantir que temos pelo menos algumas features básicas
            basic_features = ['age', 'education', 'workclass', 'occupation']
            missing_basic = [col for col in basic_features if col not in feature_columns]
            
            if missing_basic:
                self.logger.warning(f"⚠️ Features básicas ausentes: {missing_basic}")
                # Tentar usar alternativas disponíveis
                additional_features = [col for col in available_columns 
                                     if col not in feature_columns 
                                     and col != 'income' 
                                     and df_clean[col].dtype in ['object', 'int64', 'float64']]
                feature_columns.extend(additional_features[:5])  # Adicionar até 5 features extras
            
            X = df_clean[feature_columns].copy()
            self.logger.info(f"✅ Features selecionadas: {feature_columns}")
            
            # Codificar variáveis categóricas
            categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
            
            for col in categorical_columns:
                if col in X.columns:
                    le = LabelEncoder()
                    # Tratar valores únicos
                    X[col] = X[col].astype(str).fillna('Unknown')
                    X[col] = le.fit_transform(X[col])
                    self.label_encoders[col] = le
                    self.logger.info(f"✅ {col}: {len(le.classes_)} categorias únicas")
            
            # Preparar target (y)
            y = df_clean['income'].copy()
            
            # Codificar target se necessário
            if y.dtype == 'object':
                le_target = LabelEncoder()
                y = le_target.fit_transform(y.astype(str))
                self.label_encoders['income'] = le_target
                self.logger.info(f"✅ Target: {len(le_target.classes_)} classes")
            
            # Validação final
            if len(X) != len(y):
                raise ValueError(f"Incompatibilidade X({len(X)}) vs y({len(y)})")
            
            self.logger.info(f"✅ Features shape: {X.shape}")
            self.logger.info(f"✅ Target shape: {y.shape}")
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"❌ Erro na preparação SQL: {e}")
            return None, None
    
    def _safe_train_test_split(self, X, y, test_size=0.2, random_state=42):
        """Divisão treino/teste com validação"""
        try:
            # Verificar se temos dados suficientes
            if len(X) < 10:
                raise ValueError(f"Dataset muito pequeno: {len(X)} registros")
            
            # Ajustar test_size se necessário
            min_test_size = max(1, int(len(X) * 0.1))  # Pelo menos 10% ou 1 registro
            max_test_size = int(len(X) * 0.3)  # Máximo 30%
            
            actual_test_size = min(max(min_test_size, int(len(X) * test_size)), max_test_size)
            actual_test_ratio = actual_test_size / len(X)
            
            # Verificar se stratify é possível
            unique_classes = np.unique(y)
            min_class_count = min([np.sum(y == cls) for cls in unique_classes])
            
            if min_class_count >= 2:  # Cada classe tem pelo menos 2 exemplos
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=actual_test_ratio, 
                    random_state=random_state, stratify=y
                )
                self.logger.info("✅ Divisão estratificada aplicada")
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=actual_test_ratio, 
                    random_state=random_state
                )
                self.logger.warning("⚠️ Divisão não estratificada (classes desbalanceadas)")
            
            self.logger.info(f"✅ Treino: {len(X_train)} | Teste: {len(X_test)}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            self.logger.error(f"❌ Erro na divisão treino/teste: {e}")
            raise
    
    def _scale_features(self, X_train, X_test):
        """Normalizar features com validação"""
        try:
            self.scaler = StandardScaler()
            
            # Verificar se temos features numéricas
            if X_train.shape[1] == 0:
                raise ValueError("Nenhuma feature para normalizar")
            
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.logger.info("✅ Features normalizadas")
            
            return X_train_scaled, X_test_scaled
            
        except Exception as e:
            self.logger.error(f"❌ Erro na normalização: {e}")
            return X_train, X_test  # Retornar dados originais se normalização falhar
    
    def _train_models_sql(self, X_train, X_test, y_train, y_test):
        """Treinar modelos com configuração otimizada para SQL"""
        models_config = {
            'Random Forest': RandomForestClassifier(
                n_estimators=50,  # Reduzido para performance
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=42,
                solver='liblinear'  # Mais rápido para datasets pequenos/médios
            )
        }
        
        for name, model in models_config.items():
            try:
                self.logger.info(f"🔄 Treinando {name}...")
                
                # Validar dados de treino
                self._validate_training_data(X_train, y_train, name)
                
                # Treinar modelo
                model.fit(X_train, y_train)
                
                # Fazer predições
                y_pred = self._safe_predict(model, X_test, name)
                
                if y_pred is not None:
                    # Calcular métricas
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    # Salvar modelo e resultados
                    self.models[name] = model
                    self.results[name] = {
                        'accuracy': accuracy,
                        'y_test': y_test,
                        'y_pred': y_pred,
                        'model': model
                    }
                    
                    # Adicionar probabilidades se disponível
                    if hasattr(model, 'predict_proba'):
                        try:
                            y_pred_proba = model.predict_proba(X_test)
                            self.results[name]['y_pred_proba'] = y_pred_proba
                        except:
                            pass
                    
                    self.logger.info(f"✅ {name}: Accuracy = {accuracy:.4f}")
                else:
                    self.logger.warning(f"⚠️ {name}: Falha na predição")
                    
            except Exception as e:
                self.logger.error(f"❌ Erro no treinamento {name}: {e}")
                continue
        
        # Salvar artefatos
        self._save_artifacts()
    
    def _validate_training_data(self, X_train, y_train, model_name):
        """Validar dados de treino"""
        if len(X_train) == 0 or len(y_train) == 0:
            raise ValueError(f"Dados de treino vazios para {model_name}")
        
        if len(X_train) != len(y_train):
            raise ValueError(f"Incompatibilidade X_train({len(X_train)}) vs y_train({len(y_train)})")
        
        if np.any(np.isnan(X_train)):
            self.logger.warning(f"⚠️ NaN encontrado em X_train para {model_name}")
            # Substituir NaN por média
            X_train = np.nan_to_num(X_train)
        
        if np.any(np.isnan(y_train)):
            raise ValueError(f"NaN encontrado em y_train para {model_name}")
    
    def _safe_predict(self, model, X_test, model_name):
        """Fazer predições com validação"""
        try:
            if hasattr(X_test, 'shape') and X_test.shape[0] == 0:
                self.logger.warning(f"⚠️ Conjunto de teste vazio para {model_name}")
                return None
            
            # Tratar NaN se existir
            if np.any(np.isnan(X_test)):
                self.logger.warning(f"⚠️ NaN em X_test para {model_name}, substituindo...")
                X_test = np.nan_to_num(X_test)
            
            y_pred = model.predict(X_test)
            return y_pred
            
        except Exception as e:
            self.logger.error(f"❌ Erro na predição {model_name}: {e}")
            return None
    
    def _save_artifacts(self):
        """Salvar modelos e preprocessadores"""
        try:
            # Criar diretório
            processed_dir = Path("data/processed")
            processed_dir.mkdir(parents=True, exist_ok=True)
            
            # Salvar modelos
            for name, model in self.models.items():
                model_file = processed_dir / f"{name.lower().replace(' ', '_')}_model.joblib"
                joblib.dump(model, model_file)
                self.logger.info(f"💾 {name} salvo: {model_file}")
            
            # Salvar preprocessadores
            if self.scaler:
                joblib.dump(self.scaler, processed_dir / "scaler.joblib")
            
            if self.label_encoders:
                joblib.dump(self.label_encoders, processed_dir / "label_encoders.joblib")
            
            self.logger.info("✅ Artefatos ML salvos")
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao salvar artefatos: {e}")