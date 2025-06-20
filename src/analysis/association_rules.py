"""Análise completa de regras de associação com Apriori, FP-Growth e Eclat"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from itertools import combinations
from typing import List, Dict, Any, Tuple, Set
import warnings
warnings.filterwarnings('ignore')

# Check if mlxtend is available
try:
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder
    MLXTEND_AVAILABLE = True
except ImportError:
    MLXTEND_AVAILABLE = False

class AssociationRulesAnalysis:
    """Análise completa de regras de associação com Apriori, FP-Growth e Eclat"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.apriori_results = {}
        self.fp_growth_results = {}
        self.eclat_results = {}
        
    def prepare_data(self, df: pd.DataFrame) -> List[List[str]]:
        """Preparar dados para análise de regras de associação"""
        self.logger.info("📊 Preparando dados para regras de associação...")
        
        try:
            transactions = []
            
            # Converter dados para formato transacional
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            # Remover target se existir
            if 'salary' in categorical_cols:
                categorical_cols.remove('salary')
            
            self.logger.info(f"📋 Colunas categóricas: {categorical_cols}")
            
            # Adicionar bins de variáveis numéricas
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'salary' in numeric_cols:
                numeric_cols.remove('salary')
            
            # Criar bins para variáveis numéricas
            df_processed = df.copy()
            
            for col in numeric_cols[:3]:  # Limitar a 3 para não explodir o número de itens
                try:
                    df_processed[f"{col}_bin"] = pd.cut(
                        df_processed[col], 
                        bins=3, 
                        labels=[f"{col}_low", f"{col}_medium", f"{col}_high"]
                    )
                    categorical_cols.append(f"{col}_bin")
                except Exception as e:
                    self.logger.warning(f"⚠️ Erro ao criar bins para {col}: {e}")
            
            # Criar transações
            for idx, row in df_processed.iterrows():
                transaction = []
                
                # Adicionar variáveis categóricas
                for col in categorical_cols:
                    if col in row and pd.notna(row[col]):
                        transaction.append(f"{col}_{row[col]}")
                
                # Adicionar target se existir
                if 'salary' in df.columns and pd.notna(row['salary']):
                    transaction.append(f"salary_{row['salary']}")
                
                # Adicionar apenas se temos items suficientes
                if len(transaction) >= 3:
                    transactions.append(transaction)
            
            self.logger.info(f"✅ {len(transactions)} transações preparadas")
            return transactions
            
        except Exception as e:
            self.logger.error(f"❌ Erro na preparação dos dados: {e}")
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
        
        logging.info("🔍 Buscando regras de associação...")
        
        try:
            # Codificar transações
            te = TransactionEncoder()
            te_ary = te.fit(transactions).transform(transactions)
            df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
            
            logging.info(f"  📊 Dataset codificado: {df_encoded.shape}")
            
            # Encontrar itens frequentes
            self.frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
            
            if self.frequent_itemsets.empty:
                logging.warning(f"⚠️ Nenhum itemset frequente encontrado com suporte >= {min_support}")
                return pd.DataFrame()
            
            logging.info(f"  📊 {len(self.frequent_itemsets)} itemsets frequentes encontrados")
            
            # Gerar regras
            self.rules = association_rules(self.frequent_itemsets, metric="confidence", min_threshold=min_confidence)
            
            if self.rules.empty:
                logging.warning(f"⚠️ Nenhuma regra encontrada com confiança >= {min_confidence}")
                return pd.DataFrame()
            
            # Filtrar regras relacionadas com salário
            salary_rules = self.rules[
                self.rules['consequents'].astype(str).str.contains('salary_') |
                self.rules['antecedents'].astype(str).str.contains('salary_')
            ]
            
            logging.info(f"✅ {len(self.rules)} regras totais, {len(salary_rules)} relacionadas a salário")
            
            # Salvar análise
            self._save_rules_analysis(salary_rules)
            
            return salary_rules.sort_values('lift', ascending=False)
            
        except Exception as e:
            self.logger.error(f"❌ Erro no Eclat: {e}")
            return {}

    def _eclat_recursive(self, current_itemsets: Dict, all_frequent: Dict, min_support: int, k: int):
        """Recursão do Eclat para gerar itemsets de tamanho k"""
        if k > 3:  # Limitar profundidade para performance
            return
        
        new_itemsets = {}
        items = list(current_itemsets.keys())
        
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                # Intersecção das listas de transações
                intersection = current_itemsets[items[i]] & current_itemsets[items[j]]
                
                if len(intersection) >= min_support:
                    # Criar novo itemset
                    new_itemset = f"{items[i]}+{items[j]}"
                    new_itemsets[new_itemset] = intersection
        
        if new_itemsets:
            all_frequent[k] = {frozenset(itemset.split('+')): len(tids) 
                              for itemset, tids in new_itemsets.items()}
            
            self.logger.info(f"   • L{k}: {len(new_itemsets)} itemsets frequentes de tamanho {k}")
            
            # Continuar recursão
            self._eclat_recursive(new_itemsets, all_frequent, min_support, k + 1)

    def _generate_rules_eclat(self, frequent_itemsets: Dict[int, Dict], n_transactions: int, min_confidence: float = 0.6) -> List[Dict]:
        """Gerar regras para Eclat (mesmo método do Apriori)"""
        return self._generate_rules_apriori(frequent_itemsets, n_transactions, min_confidence)

    def compare_algorithms(self) -> Dict[str, Any]:
        """Comparar resultados dos três algoritmos"""
        comparison = {
            'algorithms': ['Apriori', 'FP-Growth', 'Eclat'],
            'results': []
        }
        
        for alg_name, results in [
            ('Apriori', self.apriori_results),
            ('FP-Growth', self.fp_growth_results),
            ('Eclat', self.eclat_results)
        ]:
            if results and results.get('rules'):
                rules = results['rules']
                comparison['results'].append({
                    'Algorithm': alg_name,
                    'Rules_Found': len(rules),
                    'Avg_Confidence': np.mean([r['confidence'] for r in rules]),
                    'Avg_Lift': np.mean([r['lift'] for r in rules]),
                    'Max_Confidence': max([r['confidence'] for r in rules]),
                    'Execution_Status': 'Success'
                })
            else:
                comparison['results'].append({
                    'Algorithm': alg_name,
                    'Rules_Found': 0,
                    'Avg_Confidence': 0,
                    'Avg_Lift': 0,
                    'Max_Confidence': 0,
                    'Execution_Status': 'Failed'
                })
        
        return comparison

    def save_results(self, output_dir: str = "output/analysis"):
        """Salvar resultados das análises"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Salvar resultados de cada algoritmo
        for alg_name, results in [
            ('apriori', self.apriori_results),
            ('fp_growth', self.fp_growth_results),
            ('eclat', self.eclat_results)
        ]:
            if results and results.get('rules'):
                rules_df = pd.DataFrame(results['rules'])
                rules_df.to_csv(output_path / f"{alg_name}_rules.csv", index=False)
                
                self.logger.info(f"💾 {alg_name.upper()} salvo: {len(results['rules'])} regras")
        
        # Salvar comparação
        comparison = self.compare_algorithms()
        if comparison.get('results'):
            comparison_df = pd.DataFrame(comparison['results'])
            comparison_df.to_csv(output_path / "association_algorithms_comparison.csv", index=False)
            
            self.logger.info("💾 Comparação salva: association_algorithms_comparison.csv")

    def run_complete_analysis(self, df: pd.DataFrame, min_support: float = 0.01, min_confidence: float = 0.6) -> Dict[str, Any]:
        """Executar análise completa com todos os algoritmos"""
        self.logger.info("🚀 Iniciando análise completa de regras de associação...")
        
        # Preparar dados
        transactions = self.prepare_data(df)
        
        if not transactions:
            self.logger.error("❌ Nenhuma transação válida para análise")
            return {}
        
        results = {}
        
        # Executar algoritmos
        results['apriori'] = self.run_apriori(transactions, min_support, min_confidence)
        results['fp_growth'] = self.run_fp_growth(transactions, min_support, min_confidence)
        results['eclat'] = self.run_eclat(transactions, min_support, min_confidence)
        
        # Comparação
        results['comparison'] = self.compare_algorithms()
        
        # Salvar resultados
        self.save_results()
        
        return results
    
    def run_apriori(self, transactions: List[List[str]], min_support: float = 0.01, min_confidence: float = 0.6) -> Dict[str, Any]:
        """Executar algoritmo Apriori"""
        self.logger.info("🔍 Executando APRIORI...")
        
        try:
            if not MLXTEND_AVAILABLE:
                self.logger.warning("⚠️ MLxtend não disponível - usando implementação básica")
                return self._run_apriori_basic(transactions, min_support, min_confidence)
            
            # Usar MLxtend se disponível
            from mlxtend.preprocessing import TransactionEncoder
            from mlxtend.frequent_patterns import apriori, association_rules
            
            # Codificar transações
            te = TransactionEncoder()
            te_ary = te.fit(transactions).transform(transactions)
            df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
            
            # Encontrar itemsets frequentes
            frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
            
            if frequent_itemsets.empty:
                self.logger.warning(f"⚠️ APRIORI: Nenhum itemset frequente (suporte >= {min_support})")
                return {'rules': [], 'frequent_itemsets': [], 'status': 'no_patterns'}
            
            # Gerar regras
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
            
            if rules.empty:
                self.logger.warning(f"⚠️ APRIORI: Nenhuma regra (confiança >= {min_confidence})")
                return {'rules': [], 'frequent_itemsets': len(frequent_itemsets), 'status': 'no_rules'}
            
            # Converter para formato padrão
            rules_list = []
            for _, rule in rules.iterrows():
                rules_list.append({
                    'antecedents': list(rule['antecedents']),
                    'consequents': list(rule['consequents']),
                    'support': float(rule['support']),
                    'confidence': float(rule['confidence']),
                    'lift': float(rule['lift']),
                    'conviction': float(rule.get('conviction', 1.0))
                })
            
            result = {
                'rules': rules_list,
                'frequent_itemsets': len(frequent_itemsets),
                'total_rules': len(rules_list),
                'status': 'success'
            }
            
            self.apriori_results = result
            self.logger.info(f"✅ APRIORI: {len(rules_list)} regras encontradas")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Erro no APRIORI: {e}")
            return {'rules': [], 'status': 'error', 'error': str(e)}
    
    def run_fp_growth(self, transactions: List[List[str]], min_support: float = 0.01, min_confidence: float = 0.6) -> Dict[str, Any]:
        """Executar algoritmo FP-Growth"""
        self.logger.info("🌳 Executando FP-GROWTH...")
        
        try:
            if not MLXTEND_AVAILABLE:
                self.logger.warning("⚠️ MLxtend não disponível - usando implementação básica")
                return self._run_fp_growth_basic(transactions, min_support, min_confidence)
            
            # Usar MLxtend se disponível
            from mlxtend.preprocessing import TransactionEncoder
            from mlxtend.frequent_patterns import fpgrowth, association_rules
            
            # Codificar transações
            te = TransactionEncoder()
            te_ary = te.fit(transactions).transform(transactions)
            df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
            
            # Encontrar itemsets frequentes com FP-Growth
            frequent_itemsets = fpgrowth(df_encoded, min_support=min_support, use_colnames=True)
            
            if frequent_itemsets.empty:
                self.logger.warning(f"⚠️ FP-GROWTH: Nenhum itemset frequente (suporte >= {min_support})")
                return {'rules': [], 'frequent_itemsets': [], 'status': 'no_patterns'}
            
            # Gerar regras
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
            
            if rules.empty:
                self.logger.warning(f"⚠️ FP-GROWTH: Nenhuma regra (confiança >= {min_confidence})")
                return {'rules': [], 'frequent_itemsets': len(frequent_itemsets), 'status': 'no_rules'}
            
            # Converter para formato padrão
            rules_list = []
            for _, rule in rules.iterrows():
                rules_list.append({
                    'antecedents': list(rule['antecedents']),
                    'consequents': list(rule['consequents']),
                    'support': float(rule['support']),
                    'confidence': float(rule['confidence']),
                    'lift': float(rule['lift']),
                    'conviction': float(rule.get('conviction', 1.0))
                })
            
            result = {
                'rules': rules_list,
                'frequent_itemsets': len(frequent_itemsets),
                'total_rules': len(rules_list),
                'status': 'success'
            }
            
            self.fp_growth_results = result
            self.logger.info(f"✅ FP-GROWTH: {len(rules_list)} regras encontradas")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Erro no FP-GROWTH: {e}")
            return {'rules': [], 'status': 'error', 'error': str(e)}
    
    def run_eclat(self, transactions: List[List[str]], min_support: float = 0.01, min_confidence: float = 0.6) -> Dict[str, Any]:
        """Executar algoritmo Eclat"""
        self.logger.info("⚡ Executando ECLAT...")
        
        try:
            # Implementação básica do ECLAT
            min_support_count = int(min_support * len(transactions))
            
            # Encontrar items únicos e suas transações
            item_transactions = defaultdict(set)
            for tid, transaction in enumerate(transactions):
                for item in transaction:
                    item_transactions[item].add(tid)
            
            # Filtrar items frequentes
            frequent_items = {
                item: tids for item, tids in item_transactions.items()
                if len(tids) >= min_support_count
            }
            
            if not frequent_items:
                self.logger.warning(f"⚠️ ECLAT: Nenhum item frequente (suporte >= {min_support})")
                return {'rules': [], 'frequent_itemsets': 0, 'status': 'no_patterns'}
            
            # Encontrar itemsets frequentes de tamanho 2
            frequent_itemsets = {1: {frozenset([item]): len(tids) for item, tids in frequent_items.items()}}
            
            self._eclat_recursive(frequent_items, frequent_itemsets, min_support_count, 2)
            
            # Gerar regras
            rules = self._generate_rules_eclat(frequent_itemsets, len(transactions), min_confidence)
            
            if not rules:
                self.logger.warning(f"⚠️ ECLAT: Nenhuma regra (confiança >= {min_confidence})")
                return {'rules': [], 'frequent_itemsets': sum(len(fs) for fs in frequent_itemsets.values()), 'status': 'no_rules'}
            
            result = {
                'rules': rules,
                'frequent_itemsets': sum(len(fs) for fs in frequent_itemsets.values()),
                'total_rules': len(rules),
                'status': 'success'
            }
            
            self.eclat_results = result
            self.logger.info(f"✅ ECLAT: {len(rules)} regras encontradas")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Erro no ECLAT: {e}")
            return {'rules': [], 'status': 'error', 'error': str(e)}
    
    def _run_apriori_basic(self, transactions: List[List[str]], min_support: float, min_confidence: float) -> Dict[str, Any]:
        """Implementação básica do Apriori quando MLxtend não está disponível"""
        try:
            min_support_count = int(min_support * len(transactions))
            
            # Contar items individuais
            item_counts = defaultdict(int)
            for transaction in transactions:
                for item in transaction:
                    item_counts[item] += 1
            
            # Items frequentes de tamanho 1
            frequent_1 = {
                frozenset([item]): count for item, count in item_counts.items()
                if count >= min_support_count
            }
            
            if not frequent_1:
                return {'rules': [], 'status': 'no_patterns'}
            
            # Gerar itemsets de tamanho 2
            frequent_2 = {}
            items = list(frequent_1.keys())
            
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    candidate = items[i] | items[j]
                    count = sum(1 for t in transactions if candidate.issubset(set(t)))
                    
                    if count >= min_support_count:
                        frequent_2[candidate] = count
            
            frequent_itemsets = {1: frequent_1, 2: frequent_2}
            
            # Gerar regras
            rules = self._generate_rules_apriori(frequent_itemsets, len(transactions), min_confidence)
            
            return {
                'rules': rules,
                'frequent_itemsets': len(frequent_1) + len(frequent_2),
                'total_rules': len(rules),
                'status': 'success'
            }
            
        except Exception as e:
            return {'rules': [], 'status': 'error', 'error': str(e)}
    
    def _run_fp_growth_basic(self, transactions: List[List[str]], min_support: float, min_confidence: float) -> Dict[str, Any]:
        """Implementação básica do FP-Growth quando MLxtend não está disponível"""
        # Por simplicidade, usar a mesma implementação