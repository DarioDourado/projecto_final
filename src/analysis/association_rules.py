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

    #def run_apriori(self, transactions: List[List[str]], min_support: float = 0.01, min_confidence: float = 0.6) -> Dict[str, Any]:
        """Algoritmo Apriori implementação própria"""
        self.logger.info("🔍 Executando algoritmo Apriori...")
        
        try:
            n_transactions = len(transactions)
            min_support_count = int(min_support * n_transactions)
            
            # Gerar itemsets frequentes
            frequent_itemsets = self._generate_frequent_itemsets_apriori(transactions, min_support_count)
            
            if not frequent_itemsets:
                self.logger.warning("⚠️ Nenhum itemset frequente encontrado")
                return {}
            
            # Gerar regras
            rules = self._generate_rules_apriori(frequent_itemsets, n_transactions, min_confidence)
            
            self.apriori_results = {
                'algorithm': 'Apriori',
                'frequent_itemsets': frequent_itemsets,
                'rules': rules,
                'n_transactions': n_transactions,
                'min_support': min_support,
                'min_confidence': min_confidence
            }
            
            self.logger.info(f"✅ Apriori: {len(rules)} regras encontradas")
            return self.apriori_results
            
        except Exception as e:
            self.logger.error(f"❌ Erro no Apriori: {e}")
            return {}

    def _generate_frequent_itemsets_apriori(self, transactions: List[List[str]], min_support_count: int) -> Dict[int, Dict]:
        """Gerar itemsets frequentes usando Apriori"""
        frequent_itemsets = {}
        
        # L1: Itemsets de tamanho 1
        item_counts = defaultdict(int)
        for transaction in transactions:
            for item in transaction:
                item_counts[item] += 1
        
        l1 = {frozenset([item]): count for item, count in item_counts.items() 
              if count >= min_support_count}
        frequent_itemsets[1] = l1
        
        self.logger.info(f"   • L1: {len(l1)} itemsets frequentes de tamanho 1")
        
        # Gerar Lk para k > 1
        k = 2
        while frequent_itemsets.get(k-1):
            candidates = self._generate_candidates(list(frequent_itemsets[k-1].keys()), k)
            
            # Contar suporte dos candidatos
            candidate_counts = defaultdict(int)
            for transaction in transactions:
                transaction_set = set(transaction)
                for candidate in candidates:
                    if candidate.issubset(transaction_set):
                        candidate_counts[candidate] += 1
            
            # Filtrar por suporte mínimo
            lk = {itemset: count for itemset, count in candidate_counts.items() 
                  if count >= min_support_count}
            
            if lk:
                frequent_itemsets[k] = lk
                self.logger.info(f"   • L{k}: {len(lk)} itemsets frequentes de tamanho {k}")
            else:
                break
            
            k += 1
        
        return frequent_itemsets

    def _generate_candidates(self, frequent_itemsets: List[frozenset], k: int) -> List[frozenset]:
        """Gerar candidatos para Apriori (join step)"""
        candidates = []
        
        for i in range(len(frequent_itemsets)):
            for j in range(i + 1, len(frequent_itemsets)):
                # Join step
                union = frequent_itemsets[i] | frequent_itemsets[j]
                if len(union) == k:
                    candidates.append(union)
        
        return candidates

    def _generate_rules_apriori(self, frequent_itemsets: Dict[int, Dict], n_transactions: int, min_confidence: float = 0.6) -> List[Dict]:
        """Gerar regras de associação a partir dos itemsets frequentes"""
        rules = []
        
        for k in range(2, len(frequent_itemsets) + 1):
            for itemset, support_count in frequent_itemsets[k].items():
                itemset_list = list(itemset)
                
                # Gerar todas as possíveis divisões antecedent -> consequent
                for i in range(1, len(itemset_list)):
                    for antecedent_items in combinations(itemset_list, i):
                        antecedent = frozenset(antecedent_items)
                        consequent = itemset - antecedent
                        
                        if antecedent in frequent_itemsets.get(len(antecedent), {}):
                            antecedent_support = frequent_itemsets[len(antecedent)][antecedent]
                            
                            # Calcular métricas
                            support = support_count / n_transactions
                            confidence = support_count / antecedent_support
                            
                            if confidence >= min_confidence:
                                # Calcular lift
                                consequent_support = sum(
                                    count for itemset_c, count in frequent_itemsets.get(len(consequent), {}).items()
                                    if itemset_c == consequent
                                ) / n_transactions if len(consequent) == 1 else support
                                
                                lift = confidence / consequent_support if consequent_support > 0 else 0
                                
                                rules.append({
                                    'antecedents': antecedent,
                                    'consequents': consequent,
                                    'support': support,
                                    'confidence': confidence,
                                    'lift': lift,
                                    'antecedent_support': antecedent_support / n_transactions,
                                    'consequent_support': consequent_support
                                })
        
        return sorted(rules, key=lambda x: x['confidence'], reverse=True)

    #def run_fp_growth(self, transactions: List[List[str]], min_support: float = 0.01, min_confidence: float = 0.6) -> Dict[str, Any]:
        """Algoritmo FP-Growth (implementação simplificada)"""
        self.logger.info("🌳 Executando algoritmo FP-Growth...")
        
        try:
            n_transactions = len(transactions)
            min_support_count = int(min_support * n_transactions)
            
            # 1. Contar frequência dos itens
            item_counts = defaultdict(int)
            for transaction in transactions:
                for item in transaction:
                    item_counts[item] += 1
            
            # Filtrar itens frequentes e ordenar por frequência
            frequent_items = {item: count for item, count in item_counts.items() 
                            if count >= min_support_count}
            sorted_items = sorted(frequent_items.items(), key=lambda x: x[1], reverse=True)
            item_order = {item: idx for idx, (item, _) in enumerate(sorted_items)}
            
            # 2. Reordenar transações baseado na frequência
            ordered_transactions = []
            for transaction in transactions:
                # Filtrar e ordenar itens da transação
                filtered_items = [item for item in transaction if item in frequent_items]
                ordered_items = sorted(filtered_items, key=lambda x: item_order[x])
                if len(ordered_items) >= 2:
                    ordered_transactions.append(ordered_items)
            
            # 3. Construir FP-Tree (versão simplificada)
            frequent_itemsets = self._build_fp_tree(ordered_transactions, min_support_count)
            
            # 4. Gerar regras
            rules = self._generate_rules_fp_growth(frequent_itemsets, n_transactions, min_confidence)
            
            self.fp_growth_results = {
                'algorithm': 'FP-Growth',
                'frequent_itemsets': frequent_itemsets,
                'rules': rules,
                'n_transactions': n_transactions,
                'min_support': min_support,
                'min_confidence': min_confidence
            }
            
            self.logger.info(f"✅ FP-Growth: {len(rules)} regras encontradas")
            return self.fp_growth_results
            
        except Exception as e:
            self.logger.error(f"❌ Erro no FP-Growth: {e}")
            return {}

    def _build_fp_tree(self, transactions: List[List[str]], min_support: int) -> Dict[int, Dict]:
        """Construir FP-Tree simplificada e extrair itemsets frequentes"""
        # Implementação simplificada - usar combinações diretas
        frequent_itemsets = {}
        
        # Itemsets de tamanho 1
        item_counts = defaultdict(int)
        for transaction in transactions:
            for item in transaction:
                item_counts[item] += 1
        
        l1 = {frozenset([item]): count for item, count in item_counts.items() 
              if count >= min_support}
        frequent_itemsets[1] = l1
        
        # Itemsets de tamanho 2 e 3 (simplificado)
        for k in [2, 3]:
            candidates = defaultdict(int)
            for transaction in transactions:
                for itemset in combinations(transaction, k):
                    candidates[frozenset(itemset)] += 1
            
            lk = {itemset: count for itemset, count in candidates.items() 
                  if count >= min_support}
            
            if lk:
                frequent_itemsets[k] = lk
            else:
                break
        
        return frequent_itemsets

    def _generate_rules_fp_growth(self, frequent_itemsets: Dict[int, Dict], n_transactions: int, min_confidence: float = 0.6) -> List[Dict]:
        """Gerar regras para FP-Growth (mesmo método do Apriori)"""
        return self._generate_rules_apriori(frequent_itemsets, n_transactions, min_confidence)

    #def run_eclat(self, transactions: List[List[str]], min_support: float = 0.01, min_confidence: float = 0.6) -> Dict[str, Any]:
        """Algoritmo Eclat (baseado em intersecção de conjuntos)"""
        self.logger.info("🔗 Executando algoritmo Eclat...")
        
        try:
            n_transactions = len(transactions)
            min_support_count = int(min_support * n_transactions)
            
            # Criar mapeamento item -> conjunto de transações
            item_tids = defaultdict(set)
            for tid, transaction in enumerate(transactions):
                for item in transaction:
                    item_tids[item].add(tid)
            
            # Filtrar itens frequentes
            frequent_items = {item: tids for item, tids in item_tids.items() 
                            if len(tids) >= min_support_count}
            
            if not frequent_items:
                self.logger.warning("⚠️ Nenhum item frequente encontrado")
                return {}
            
            # Gerar itemsets frequentes usando intersecção
            frequent_itemsets = {1: {frozenset([item]): len(tids) for item, tids in frequent_items.items()}}
            
            self.logger.info(f"   • L1: {len(frequent_itemsets[1])} itemsets frequentes de tamanho 1")
            
            # Recursão para encontrar itemsets maiores
            self._eclat_recursive(frequent_items, frequent_itemsets, min_support_count, 2)
            
            # Gerar regras
            rules = self._generate_rules_eclat(frequent_itemsets, n_transactions, min_confidence)
            
            self.eclat_results = {
                'algorithm': 'Eclat',
                'frequent_itemsets': frequent_itemsets,
                'rules': rules,
                'n_transactions': n_transactions,
                'min_support': min_support,
                'min_confidence': min_confidence
            }
            
            self.logger.info(f"✅ Eclat: {len(rules)} regras encontradas")
            return self.eclat_results
            
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