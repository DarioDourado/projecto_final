"""Análise de Regras de Associação"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

try:
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder
    MLXTEND_AVAILABLE = True
except ImportError:
    MLXTEND_AVAILABLE = False
    logging.warning("⚠️ mlxtend não disponível. Execute: pip install mlxtend")

class AssociationRulesAnalysis:
    """Análise de regras de associação para padrões salariais"""
    
    def __init__(self):
        self.rules = None
        self.frequent_itemsets = None
    
    def prepare_transaction_data(self, df):
        """Preparar dados para análise de associação"""
        if not MLXTEND_AVAILABLE:
            logging.error("❌ mlxtend necessário para regras de associação")
            return []
        
        logging.info("🔧 Preparando dados para análise de associação...")
        
        # Discretizar variáveis numéricas
        df_discrete = df.copy()
        
        # Idade em faixas
        if 'age' in df.columns:
            df_discrete['age_group'] = pd.cut(df['age'], 
                                            bins=[0, 30, 40, 50, 100], 
                                            labels=['jovem', 'adulto', 'maduro', 'senior'])
        
        # Horas em faixas
        if 'hours-per-week' in df.columns:
            df_discrete['hours_group'] = pd.cut(df['hours-per-week'], 
                                              bins=[0, 35, 45, 100], 
                                              labels=['part_time', 'full_time', 'overtime'])
        
        # Educação em faixas
        if 'education-num' in df.columns:
            df_discrete['education_level'] = pd.cut(df['education-num'], 
                                                  bins=[0, 9, 12, 16, 20], 
                                                  labels=['basico', 'medio', 'superior', 'pos_grad'])
        
        # Criar transações
        transactions = []
        for _, row in df_discrete.iterrows():
            transaction = []
            
            # Adicionar características categóricas disponíveis
            categorical_features = [
                ('workclass', 'workclass'),
                ('education', 'education'), 
                ('marital-status', 'marital'),
                ('occupation', 'occupation'),
                ('sex', 'sex'),
                ('age_group', 'age'),
                ('hours_group', 'hours'),
                ('education_level', 'edu_level'),
                ('salary', 'salary')
            ]
            
            for col, prefix in categorical_features:
                if col in row and pd.notna(row[col]):
                    transaction.append(f"{prefix}_{row[col]}")
            
            # Adicionar apenas se temos items suficientes
            if len(transaction) >= 3:
                transactions.append(transaction)
        
        logging.info(f"✅ {len(transactions)} transações preparadas")
        return transactions
    
    def find_association_rules(self, transactions, min_support=0.03, min_confidence=0.5):
        """Encontrar regras de associação"""
        if not MLXTEND_AVAILABLE or not transactions:
            return pd.DataFrame()
        
        logging.info("🔍 A procura de regras de associação...")
        
        try:
            # Codificar transações
            te = TransactionEncoder()
            te_ary = te.fit(transactions).transform(transactions)
            df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
            
            logging.info(f"  📊 Dataset codificado: {df_encoded.shape}")
            
            # Encontrar itens frequentes
            self.frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
            
            # Fix: Check if DataFrame is empty properly
            if self.frequent_itemsets.empty:
                logging.warning(f"⚠️ Nenhum itemset frequente encontrado com suporte >= {min_support}")
                return pd.DataFrame()
            
            logging.info(f"  📊 {len(self.frequent_itemsets)} itemsets frequentes encontrados")
            
            # Gerar regras
            self.rules = association_rules(self.frequent_itemsets, metric="confidence", min_threshold=min_confidence)
            
            # Fix: Check if DataFrame is empty properly
            if self.rules.empty:
                logging.warning(f"⚠️ Nenhuma regra encontrada com confiança >= {min_confidence}")
                return pd.DataFrame()
            
            # Filtrar regras relacionadas com salário
            # Fix: Use proper DataFrame filtering
            salary_mask = (
                self.rules['consequents'].astype(str).str.contains('salary_', na=False) |
                self.rules['antecedents'].astype(str).str.contains('salary_', na=False)
            )
            salary_rules = self.rules[salary_mask]
            
            logging.info(f"✅ {len(self.rules)} regras totais, {len(salary_rules)} relacionadas a salário")
            
            # Salvar análise
            self._save_rules_analysis(salary_rules)
            
            return salary_rules.sort_values('lift', ascending=False)
        
        except Exception as e:
            logging.error(f"❌ Erro na análise de associação: {e}")
            return pd.DataFrame()
    
    def _save_rules_analysis(self, rules):
        """Salvar análise das regras"""
        output_dir = Path("output/analysis")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not rules.empty:
            # Salvar CSV
            rules.to_csv(output_dir / "association_rules_salary.csv", index=False)
            
            # Criar relatório
            report = []
            report.append("# RELATÓRIO DE REGRAS DE ASSOCIAÇÃO\n\n")
            report.append(f"**Data:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
            report.append(f"**Total de regras:** {len(rules)}\n\n")
            
            # Top 5 regras
            top_rules = rules.head(5)
            report.append("## TOP 5 REGRAS:\n\n")
            
            for idx, rule in top_rules.iterrows():
                antecedents = ", ".join(list(rule['antecedents']))
                consequents = ", ".join(list(rule['consequents']))
                
                report.append(f"**Regra {idx + 1}:**\n")
                report.append(f"- SE: {antecedents}\n")
                report.append(f"- ENTÃO: {consequents}\n")
                report.append(f"- Confiança: {rule['confidence']:.3f}\n")
                report.append(f"- Lift: {rule['lift']:.3f}\n\n")
            
            # Salvar relatório
            with open(output_dir / "association_rules_report.md", 'w', encoding='utf-8') as f:
                f.writelines(report)
        
        logging.info("📊 Análise de regras de associação salva")