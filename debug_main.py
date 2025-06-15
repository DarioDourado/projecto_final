"""Debug do main.py para ver onde está travando"""

import sys
from pathlib import Path

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent / "src"))

def debug_main():
    """Debug passo a passo do main.py"""
    print("🔍 DEBUGANDO MAIN.PY...")
    print("=" * 50)
    
    try:
        print("1. Importando dependências...")
        import logging
        import pandas as pd
        import numpy as np
        from datetime import datetime
        print("   ✅ Dependências básicas OK")
        
        print("2. Testando classe HybridPipelineSQL...")
        from main import HybridPipelineSQL
        print("   ✅ Classe importada OK")
        
        print("3. Inicializando pipeline...")
        pipeline = HybridPipelineSQL(force_csv=True, show_results=True)
        print("   ✅ Pipeline inicializado OK")
        
        print("4. Verificando método run...")
        if hasattr(pipeline, 'run'):
            print("   ✅ Método run existe")
        else:
            print("   ❌ Método run não existe")
            
        print("5. Testando execução...")
        results = pipeline.run()
        print(f"   ✅ Execução concluída: {type(results)}")
        
        if results:
            print(f"   📊 Resultados: {list(results.keys())}")
        else:
            print("   ⚠️ Nenhum resultado retornado")
            
    except ImportError as e:
        print(f"   ❌ Erro de importação: {e}")
    except Exception as e:
        print(f"   ❌ Erro na execução: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_main()