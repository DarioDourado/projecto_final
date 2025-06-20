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
            
            # Criar bins para variáveis numéricas
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'salary' in numeric_cols:
                numeric_cols.remove('salary')
            
            df_processed = df.copy()
            
            # Simplificar bins para reduzir dimensionalidade
            for col in numeric_cols[:3]:  # Apenas top 3 colunas numéricas
                try:
                    # Usar quantis para criar bins mais balanceados
                    df_processed[f"{col}_bin"] = pd.qcut(
                        df_processed[col], 
                        q=3, 
                        labels=[f"{col}_low", f"{col}_medium", f"{col}_high"],
                        duplicates='drop'
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
                        # Simplificar nomes para evitar items muito específicos
                        item_name = f"{col}_{str(row[col]).replace(' ', '_')}"
                        transaction.append(item_name)
                
                # Adicionar target se existir
                if 'salary' in df.columns and pd.notna(row['salary']):
                    transaction.append(f"salary_{row['salary']}")
                
                # Adicionar apenas se temos items suficientes
                if len(transaction) >= 2:  # Reduzir requisito mínimo
                    transactions.append(transaction)
            
            self.logger.info(f"✅ {len(transactions)} transações preparadas")
            return transactions
            
        except Exception as e:
            self.logger.error(f"❌ Erro na preparação dos dados: {e}")
            return []
    
    
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
            # Fix: Wrong error message and return type
            self.logger.error(f"❌ Erro na análise de regras: {e}")  # Was: "Erro no Eclat"
            return pd.DataFrame() 

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

    def _generate_rules_apriori(self, frequent_itemsets: Dict[int, Dict], n_transactions: int, min_confidence: float = 0.6) -> List[Dict]:
        """Gerar regras de associação a partir de itemsets frequentes"""
        rules = []
        
        try:
            # Apenas itemsets de tamanho >= 2 podem gerar regras
            for size in range(2, len(frequent_itemsets) + 1):
                if size not in frequent_itemsets:
                    continue
                
                for itemset, support_count in frequent_itemsets[size].items():
                    if len(itemset) < 2:
                        continue
                    
                    # Gerar todas as combinações possíveis de antecedente/consequente
                    items = list(itemset)
                    
                    for i in range(1, len(items)):
                        for antecedent in combinations(items, i):
                            antecedent = frozenset(antecedent)
                            consequent = itemset - antecedent
                            
                            if not consequent:
                                continue
                            
                            # Calcular suporte do antecedente
                            antecedent_support = 0
                            for fs_size, fs_dict in frequent_itemsets.items():
                                if antecedent in fs_dict:
                                    antecedent_support = fs_dict[antecedent]
                                    break
                            
                            if antecedent_support == 0:
                                continue
                            
                            # Calcular métricas
                            support = support_count / n_transactions
                            confidence = support_count / antecedent_support
                            
                            if confidence >= min_confidence:
                                # Calcular lift
                                consequent_support = 0
                                for fs_size, fs_dict in frequent_itemsets.items():
                                    if consequent in fs_dict:
                                        consequent_support = fs_dict[consequent]
                                        break
                                
                                if consequent_support > 0:
                                    lift = confidence / (consequent_support / n_transactions)
                                else:
                                    lift = 1.0
                                
                                # Calcular conviction
                                if confidence < 1:
                                    conviction = (1 - (consequent_support / n_transactions)) / (1 - confidence)
                                else:
                                    conviction = float('inf')
                                
                                rules.append({
                                    'antecedents': list(antecedent),
                                    'consequents': list(consequent),
                                    'support': support,
                                    'confidence': confidence,
                                    'lift': lift,
                                    'conviction': conviction
                                })
            
            return rules
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao gerar regras: {e}")
            return []

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
    
        # Criar visualizações
        self.create_visualizations()

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
        return self._run_apriori_basic(transactions, min_support, min_confidence)
    
    def create_visualizations(self, output_dir: str = "output/imagens"):
        """Criar visualizações dos resultados"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("📊 Criando visualizações...")
        
        try:
            # 1. Gráfico de comparação de algoritmos
            self._plot_algorithm_comparison(output_path)
            
            # 2. Gráficos de regras por algoritmo
            if self.apriori_results.get('rules'):
                self._plot_rules_metrics(self.apriori_results['rules'], 'Apriori', output_path)
            
            if self.fp_growth_results.get('rules'):
                self._plot_rules_metrics(self.fp_growth_results['rules'], 'FP-Growth', output_path)
            
            if self.eclat_results.get('rules'):
                self._plot_rules_metrics(self.eclat_results['rules'], 'Eclat', output_path)
            
            # 3. Gráfico combinado de métricas
            self._plot_combined_metrics(output_path)
            
            self.logger.info(f"✅ Visualizações salvas em: {output_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao criar visualizações: {e}")

    def _plot_algorithm_comparison(self, output_path: Path):
        """Gráfico de comparação entre algoritmos"""
        import matplotlib.pyplot as plt
        
        comparison = self.compare_algorithms()
        if not comparison.get('results'):
            return
        
        df_comp = pd.DataFrame(comparison['results'])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Comparação de Algoritmos de Regras de Associação', fontsize=16)
        
        # Número de regras encontradas
        axes[0, 0].bar(df_comp['Algorithm'], df_comp['Rules_Found'], 
                       color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[0, 0].set_title('Número de Regras Encontradas')
        axes[0, 0].set_ylabel('Quantidade')
        
        # Confiança média
        axes[0, 1].bar(df_comp['Algorithm'], df_comp['Avg_Confidence'], 
                       color=['#96CEB4', '#FFEAA7', '#DDA0DD'])
        axes[0, 1].set_title('Confiança Média')
        axes[0, 1].set_ylabel('Confiança')
        
        # Lift médio
        axes[1, 0].bar(df_comp['Algorithm'], df_comp['Avg_Lift'], 
                       color=['#FD79A8', '#FDCB6E', '#6C5CE7'])
        axes[1, 0].set_title('Lift Médio')
        axes[1, 0].set_ylabel('Lift')
        
        # Status de execução
        status_counts = df_comp['Execution_Status'].value_counts()
        axes[1, 1].pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%')
        axes[1, 1].set_title('Status de Execução')
        
        plt.tight_layout()
        plt.savefig(output_path / 'algorithm_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_rules_metrics(self, rules: List[Dict], algorithm_name: str, output_path: Path):
        """Gráfico de métricas das regras para um algoritmo específico"""
        import matplotlib.pyplot as plt
        
        if not rules:
            return
        
        # Extrair métricas
        confidences = [rule['confidence'] for rule in rules[:20]]  # Top 20
        lifts = [rule['lift'] for rule in rules[:20]]
        supports = [rule['support'] for rule in rules[:20]]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Métricas das Regras - {algorithm_name}', fontsize=16)
        
        # Distribuição de confiança
        axes[0, 0].hist(confidences, bins=10, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Distribuição de Confiança')
        axes[0, 0].set_xlabel('Confiança')
        axes[0, 0].set_ylabel('Frequência')
        
        # Distribuição de lift
        axes[0, 1].hist(lifts, bins=10, alpha=0.7, color='lightgreen')
        axes[0, 1].set_title('Distribuição de Lift')
        axes[0, 1].set_xlabel('Lift')
        axes[0, 1].set_ylabel('Frequência')
        
        # Scatter plot: Confiança vs Lift
        axes[1, 0].scatter(confidences, lifts, alpha=0.6, c='red')
        axes[1, 0].set_title('Confiança vs Lift')
        axes[1, 0].set_xlabel('Confiança')
        axes[1, 0].set_ylabel('Lift')
        
        # Scatter plot: Suporte vs Confiança
        axes[1, 1].scatter(supports, confidences, alpha=0.6, c='purple')
        axes[1, 1].set_title('Suporte vs Confiança')
        axes[1, 1].set_xlabel('Suporte')
        axes[1, 1].set_ylabel('Confiança')
        
        plt.tight_layout()
        plt.savefig(output_path / f'{algorithm_name.lower()}_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_combined_metrics(self, output_path: Path):
        """Gráfico combinado de todas as métricas"""
        import matplotlib.pyplot as plt
        
        all_rules = []
        algorithms = []
        
        # Coletar todas as regras
        for alg_name, results in [
            ('Apriori', self.apriori_results),
            ('FP-Growth', self.fp_growth_results),
            ('Eclat', self.eclat_results)
        ]:
            if results.get('rules'):
                for rule in results['rules'][:10]:  # Top 10 de cada
                    all_rules.append(rule)
                    algorithms.append(alg_name)
        
        if not all_rules:
            return
        
        # Criar DataFrame
        df_rules = pd.DataFrame(all_rules)
        df_rules['Algorithm'] = algorithms
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Comparação de Métricas - Todos os Algoritmos', fontsize=16)
        
        # Boxplot de confiança por algoritmo
        df_rules.boxplot(column='confidence', by='Algorithm', ax=axes[0, 0])
        axes[0, 0].set_title('Confiança por Algoritmo')
        axes[0, 0].set_xlabel('Algoritmo')
        axes[0, 0].set_ylabel('Confiança')
        
        # Boxplot de lift por algoritmo
        df_rules.boxplot(column='lift', by='Algorithm', ax=axes[0, 1])
        axes[0, 1].set_title('Lift por Algoritmo')
        axes[0, 1].set_xlabel('Algoritmo')
        axes[0, 1].set_ylabel('Lift')
        
        # Scatter plot colorido por algoritmo
        for alg in df_rules['Algorithm'].unique():
            mask = df_rules['Algorithm'] == alg
            axes[1, 0].scatter(df_rules[mask]['confidence'], df_rules[mask]['lift'], 
                              label=alg, alpha=0.7)
        axes[1, 0].set_title('Confiança vs Lift por Algoritmo')
        axes[1, 0].set_xlabel('Confiança')
        axes[1, 0].set_ylabel('Lift')
        axes[1, 0].legend()
        
        # Distribuição de suporte
        axes[1, 1].hist([df_rules[df_rules['Algorithm'] == alg]['support'].values 
                        for alg in df_rules['Algorithm'].unique()], 
                       label=df_rules['Algorithm'].unique(), alpha=0.7)
        axes[1, 1].set_title('Distribuição de Suporte')
        axes[1, 1].set_xlabel('Suporte')
        axes[1, 1].set_ylabel('Frequência')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(output_path / 'combined_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _save_rules_analysis(self, rules_df: pd.DataFrame):
        """Salvar análise das regras de associação"""
        try:
            output_dir = Path("output/analysis")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if not rules_df.empty:
                # Salvar regras relacionadas a salário
                rules_df.to_csv(output_dir / "salary_association_rules.csv", index=False)
                self.logger.info(f"💾 Regras de salário salvas: {len(rules_df)} regras")
        except Exception as e:
            self.logger.error(f"❌ Erro ao salvar análise: {e}")