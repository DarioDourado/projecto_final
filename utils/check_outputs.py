from pathlib import Path
import pandas as pd

def check_analysis_outputs():
    """Verificar se os outputs dos algoritmos estão sendo salvos"""
    
    analysis_dir = Path("output/analysis")
    
    expected_files = {
        'DBSCAN': ['dbscan_results.csv', 'dbscan_summary.csv'],
        'APRIORI': ['apriori_rules.csv'],
        'FP-GROWTH': ['fp_growth_rules.csv'], 
        'ECLAT': ['eclat_rules.csv'],
        'CLUSTERING': ['clustering_results.csv', 'clustering_comparison.csv']
    }
    
    print("🔍 VERIFICAÇÃO DE OUTPUTS - ANÁLISES")
    print("=" * 50)
    
    for algorithm, files in expected_files.items():
        print(f"\n📊 {algorithm}:")
        for file in files:
            file_path = analysis_dir / file
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    print(f"  ✅ {file} - {len(df)} registros")
                except:
                    print(f"  ✅ {file} - Existe mas erro ao ler")
            else:
                print(f"  ❌ {file} - Não encontrado")
    
    # Verificar diretório geral
    if analysis_dir.exists():
        all_files = list(analysis_dir.glob("*.csv"))
        print(f"\n📁 Total de arquivos CSV em output/analysis: {len(all_files)}")
        for file in all_files:
            print(f"  📄 {file.name}")
    else:
        print(f"\n❌ Diretório output/analysis não existe")

if __name__ == "__main__":
    check_analysis_outputs()