#!/usr/bin/env python3
"""
🚀 Sistema de Análise Salarial com Fallback SQL → CSV
Implementa: DBSCAN, APRIORI, FP-GROWTH, ECLAT
"""

import sys
from pathlib import Path

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent / "src"))

from src.pipelines.hybrid_pipeline import HybridPipelineSQL

def main():
    """Função principal com fallback SQL → CSV"""
    print("🔍 DEBUG: Iniciando pipeline híbrido com fallback...")
    
    try:
        print("🔍 DEBUG: Criando pipeline...")
        
        # Pipeline com fallback automático
        pipeline = HybridPipelineSQL(
            force_csv=False,        # False = tenta SQL primeiro
            log_level="INFO",       
            show_results=True,      
            auto_optimize=True      
        )
        
        print("🔍 DEBUG: Executando com fallback SQL → CSV...")
        
        # Executar pipeline (tentará SQL, se falhar usa CSV)
        results = pipeline.run()
        
        if results:
            print("🔍 DEBUG: ✅ Pipeline executado com sucesso!")
            print("🔍 DEBUG: 💾 Fallback SQL → CSV funcionando")
            print("🔍 DEBUG: 📊 Algoritmos executados:")
            print("🔍 DEBUG:    • DBSCAN (Clustering)")
            print("🔍 DEBUG:    • APRIORI (Association Rules)")  
            print("🔍 DEBUG:    • FP-GROWTH (Association Rules)")
            print("🔍 DEBUG:    • ECLAT (Association Rules)")
            print("🔍 DEBUG: 📁 Resultados em output/analysis/")
            return True
        else:
            print("🔍 DEBUG: ❌ Pipeline falhou")
            return False
            
    except Exception as e:
        print(f"🔍 DEBUG: ❌ Erro capturado: {e}")
        import traceback
        print("📋 TRACEBACK COMPLETO:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    print(f"🔍 DEBUG: Script finalizado com sucesso: {success}")