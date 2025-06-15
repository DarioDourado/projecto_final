"""Pipeline de Regras de Associação"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from src.analysis.association_rules import AssociationRulesAnalysis

class AssociationPipeline:
    """Pipeline para análise de regras de associação"""
    
    def __init__(self):
        self.analysis = AssociationRulesAnalysis()
        self.logger = logging.getLogger(__name__)
        
    def run(self, df: pd.DataFrame, min_support: float = 0.01, min_confidence: float = 0.6) -> Optional[Dict[str, Any]]:
        """Executar pipeline completo de regras de associação"""
        try:
            self.logger.info("📋 Iniciando pipeline de regras de associação...")
            
            # Validar dados
            if df is None or df.empty:
                self.logger.error("❌ DataFrame vazio ou None")
                return None
            
            # Executar análise completa
            results = self.analysis.run_complete_analysis(
                df, 
                min_support=min_support, 
                min_confidence=min_confidence
            )
            
            if results:
                self.logger.info("✅ Pipeline de regras de associação concluído")
                
                # Log estatísticas
                total_rules = 0
                for alg_name in ['apriori', 'fp_growth', 'eclat']:
                    if alg_name in results and results[alg_name].get('rules'):
                        rule_count = len(results[alg_name]['rules'])
                        total_rules += rule_count
                        self.logger.info(f"   📊 {alg_name.upper()}: {rule_count} regras")
                
                self.logger.info(f"   🔢 Total: {total_rules} regras encontradas")
                
                return results
            else:
                self.logger.warning("⚠️ Nenhuma regra encontrada")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Erro no pipeline de associação: {e}")
            return None