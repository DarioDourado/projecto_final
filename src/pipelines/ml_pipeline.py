"""Pipeline de Machine Learning - Versão SQL Compatível"""

import pandas as pd
import numpy as np
import logging
import joblib
from pathlib import Path

# Imports com fallback
try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class MLPipeline:
    """Pipeline de ML compatível com estrutura SQL"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.results = {}
        self.label_encoders = {}
        self.scaler = None
        
        if not SKLEARN_AVAILABLE:
            self.logger.warning("⚠️ scikit-learn não disponível - ML desabilitado")
    
    def run(self, df):
        """Executar pipeline ML com dados SQL"""
        if not SKLEARN_AVAILABLE:
            self.logger.error("❌ scikit-learn não disponível")
            return {}, {}
        
        self.logger.info("🤖 Iniciando Pipeline de Machine Learning")
        
        try:
            # Validar dados de entrada
            if df is None or len(df) == 0:
                raise ValueError("Dataset vazio")
            
            self.logger.info(f"📊 Dataset original: {len(df)} registros")
            self.logger.info(f"📋 Colunas disponíveis: {list(df.columns)}")
            
            # Preparar dados
            X, y = self._prepare_data_basic(df)
            
            if X is None or y is None:
                raise ValueError("Falha na preparação dos dados")
            
            self.logger.info(f"✅ Dados preparados: {X.shape[0]} registros, {X.shape[1]} features")
            
            # Divisão treino/teste
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Normalização
            X_train_scaled, X_test_scaled = self._scale_features(X_train, X_test)
            
            # Treinar modelos
            self._train_basic_models(X_train_scaled, X_test_scaled, y_train, y_test)
            
            return self.models, self.results
            
        except Exception as e:
            self.logger.error(f"❌ Erro no pipeline ML: {e}")
            return {}, {}
    
    def _prepare_data_basic(self, df):
        """Preparação básica de dados"""
        try:
            # Fazer cópia
            df_work = df.copy()
            
            # Mapeamento de colunas
            if 'salary' in df_work.columns:
                df_work['income'] = df_work['salary']
            
            # Target
            if 'income' not in df_work.columns:
                self.logger.error("❌ Coluna 'income' ou 'salary' não encontrada")
                return None, None
            
            # Features numéricas básicas
            numeric_features = []
            for col in ['age', 'fnlwgt', 'education-num', 'capital-gain', 
                       'capital-loss', 'hours-per-week']:
                if col in df_work.columns:
                    numeric_features.append(col)
            
            if not numeric_features:
                self.logger.error("❌ Nenhuma feature numérica encontrada")
                return None, None
            
            # Preparar X e y
            X = df_work[numeric_features].copy()
            y = df_work['income'].copy()
            
            # Limpeza
            X = X.fillna(X.median())
            
            # Codificar target
            if y.dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(y.astype(str))
                self.label_encoders['income'] = le
            
            self.logger.info(f"✅ Features: {list(X.columns)}")
            return X, y
            
        except Exception as e:
            self.logger.error(f"❌ Erro na preparação: {e}")
            return None, None
    
    def _scale_features(self, X_train, X_test):
        """Normalizar features"""
        try:
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.logger.info("✅ Features normalizadas")
            return X_train_scaled, X_test_scaled
            
        except Exception as e:
            self.logger.error(f"❌ Erro na normalização: {e}")
            return X_train, X_test
    
    def _train_basic_models(self, X_train, X_test, y_train, y_test):
        """Treinar modelos básicos"""
        models_config = {
            'Random Forest': RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                random_state=42,
                n_jobs=1  # Reduzido para compatibilidade
            ),
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=42,
                solver='liblinear'
            )
        }
        
        for name, model in models_config.items():
            try:
                self.logger.info(f"🔄 Treinando {name}...")
                
                # Treinar
                model.fit(X_train, y_train)
                
                # Predizer
                y_pred = model.predict(X_test)
                
                # Calcular métricas
                accuracy = accuracy_score(y_test, y_pred)
                
                # Salvar
                self.models[name] = model
                self.results[name] = {
                    'accuracy': accuracy,
                    'y_test': y_test,
                    'y_pred': y_pred,
                    'model': model
                }
                
                self.logger.info(f"✅ {name}: Accuracy = {accuracy:.4f}")
                
            except Exception as e:
                self.logger.error(f"❌ Erro no treinamento {name}: {e}")
                continue
        
        # Salvar artefatos
        self._save_artifacts()
    
    def _save_artifacts(self):
        """Salvar modelos"""
        try:
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