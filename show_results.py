"""
🚀 Script para apresentar todos os resultados dos algoritmos
DBSCAN, APRIORI, FP-GROWTH, ECLAT + Comparações
"""

import sys
from pathlib import Path

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent / "src"))

def main():
    """Executar apresentação completa dos resultados"""
    print("🚀 Iniciando apresentação completa de resultados...")
    
    try:
        # Importar o apresentador
        from analysis.results_presenter import ResultsPresenter
        
        # Verificar se diretório de análise existe
        analysis_dir = Path("output/analysis")
        if not analysis_dir.exists():
            print(f"❌ Diretório {analysis_dir} não encontrado!")
            print("💡 Execute primeiro o pipeline principal: python main.py")
            return False
        
        # Criar apresentador
        presenter = ResultsPresenter(analysis_dir)
        
        # Gerar relatório completo
        presenter.generate_final_report()
        
        print("\n🎉 Apresentação completa finalizada!")
        print("📊 Todos os gráficos e comparações foram gerados!")
        return True
        
    except ImportError as e:
        print(f"❌ Erro de importação: {e}")
        print("💡 Verifique se o arquivo results_presenter.py está em src/analysis/")
        return False
    except Exception as e:
        print(f"❌ Erro na apresentação: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("✅ Apresentação concluída com sucesso!")
    else:
        print("❌ Falha na apresentação dos resultados")