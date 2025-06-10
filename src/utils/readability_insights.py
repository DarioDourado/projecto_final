"""
Utilitários para mostrar melhorias de legibilidade no processamento de dados
Preparado para futuro suporte multilingual
Mensagens em Português de Portugal
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging

class ReadabilityInsights:
    """Classe para gerar insights sobre as melhorias de legibilidade"""
    
    def __init__(self):
        self.enhancement_report = None
        self.load_enhancement_report()
    
    def load_enhancement_report(self):
        """Carregar relatório de melhorias"""
        report_path = Path("data/processed/enhancement_report.json")
        
        if report_path.exists():
            try:
                with open(report_path, 'r', encoding='utf-8') as f:
                    self.enhancement_report = json.load(f)
                logging.info("✅ Relatório de melhorias carregado")
            except Exception as e:
                logging.warning(f"⚠️ Erro ao carregar relatório: {e}")
                self.enhancement_report = None
    
    def show_enhancement_summary(self):
        """Mostrar resumo das melhorias aplicadas"""
        if not self.enhancement_report:
            print("⚠️ Relatório de melhorias não encontrado. Execute o processamento primeiro.")
            return
        
        print("="*60)
        print("📊 RESUMO DAS MELHORIAS DE LEGIBILIDADE")
        print("="*60)
        
        # Verificar estrutura do relatório
        summary = self.enhancement_report.get('summary', {})
        column_enhancements = self.enhancement_report.get('column_enhancements', {})
        
        # Estatísticas gerais
        total_columns = summary.get('total_columns_enhanced', len(column_enhancements))
        avg_mapping_rate = summary.get('average_mapping_rate', 0.0)
        
        print(f"🔄 Colunas melhoradas: {total_columns}")
        print(f"📈 Taxa média de mapeamento: {avg_mapping_rate:.1%}")
        
        if column_enhancements:
            print("\n📝 Colunas Processadas:")
            for col_name, col_data in column_enhancements.items():
                mapped_values = col_data.get('mapped_values', 0)
                total_values = col_data.get('total_values', 0)
                mapping_rate = col_data.get('mapping_rate', 0.0)
                
                print(f"  • {col_name}: {mapped_values}/{total_values} valores mapeados ({mapping_rate:.1%})")
        
        # Mostrar exemplos de melhorias se disponíveis
        print("\n📝 Exemplos de Melhorias:")
        examples_shown = 0
        for column, col_data in column_enhancements.items():
            if examples_shown >= 3:  # Limitar a 3 colunas
                break
            
            mappings_applied = col_data.get('mappings_applied', {})
            if mappings_applied:
                print(f"\n  📂 {column.upper()}:")
                for original, enhanced in list(mappings_applied.items())[:3]:  # Top 3 por coluna
                    print(f"    {original} → {enhanced}")
                examples_shown += 1
        
        if not column_enhancements:
            print("  📝 Nenhuma melhoria específica registrada")
    
    def compare_readability(self, original_df, processed_df):
        """Comparar legibilidade antes e depois do processamento"""
        print("\n" + "="*60)
        print("📊 COMPARAÇÃO DE LEGIBILIDADE")
        print("="*60)
        
        # Comparar número de colunas
        print(f"📊 Colunas originais: {len(original_df.columns)}")
        print(f"📊 Colunas melhoradas: {len(processed_df.columns)}")
        
        new_columns = len(processed_df.columns) - len(original_df.columns)
        if new_columns > 0:
            print(f"➕ Novas colunas interpretáveis: {new_columns}")
        
        # Mostrar exemplos de colunas melhoradas
        enhanced_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship']
        
        for col in enhanced_columns:
            if col in original_df.columns and col in processed_df.columns:
                print(f"\n🔄 COLUNA: {col.upper()}")
                
                try:
                    # Valores únicos originais vs melhorados
                    original_unique = set(original_df[col].dropna().astype(str).unique())
                    enhanced_unique = set(processed_df[col].dropna().astype(str).unique())
                    
                    print(f"  📈 Valores únicos originais: {len(original_unique)}")
                    print(f"  📈 Valores únicos melhorados: {len(enhanced_unique)}")
                    
                    # Mostrar algumas melhorias
                    print(f"  📝 Exemplos de melhorias:")
                    sample_original = list(original_unique)[:3]
                    for orig_val in sample_original:
                        # Encontrar valor melhorado correspondente
                        mask = original_df[col].astype(str) == orig_val
                        if mask.any():
                            try:
                                enhanced_val = processed_df.loc[mask, col].iloc[0]
                                if str(orig_val) != str(enhanced_val):
                                    print(f"    {orig_val} → {enhanced_val}")
                            except:
                                continue
                except Exception as e:
                    print(f"  ⚠️ Erro ao processar coluna {col}: {e}")
                    continue
    
    def show_new_features_distribution(self, df):
        """Mostrar distribuição das novas features interpretáveis"""
        print("\n" + "="*60)
        print("📊 DISTRIBUIÇÃO DAS NOVAS FEATURES INTERPRETÁVEIS")
        print("="*60)
        
        # Procurar colunas que parecem ser novas features
        new_feature_patterns = ['age_group', 'work_schedule', 'education_level', 'investment_profile']
        
        found_features = [col for col in df.columns if any(pattern in col for pattern in new_feature_patterns)]
        
        if not found_features:
            print("📝 Nenhuma nova feature interpretável encontrada")
            return
        
        for col in found_features:
            print(f"\n📈 {col.upper().replace('_', ' ')}:")
            try:
                value_counts = df[col].value_counts()
                total = len(df)
                
                for value, count in value_counts.head(5).items():
                    percentage = (count / total) * 100
                    print(f"  • {value}: {count:,} ({percentage:.1f}%)")
                
                if len(value_counts) > 5:
                    others = len(value_counts) - 5
                    print(f"  • ... e mais {others} categorias")
            except Exception as e:
                print(f"  ⚠️ Erro ao processar {col}: {e}")
    
    def show_multilingual_readiness(self):
        """Mostrar como os dados estão preparados para suporte multilingual"""
        print("\n" + "="*60)
        print("🌍 PREPARAÇÃO PARA MULTILINGUAL")
        print("="*60)
        
        print("✅ Estrutura de dados otimizada para tradução:")
        print("  • Categorias claras e descritivas em inglês")
        print("  • Convenções de nomenclatura consistentes")
        print("  • Níveis educacionais hierárquicos")
        print("  • Grupos etários padronizados")
        print("  • Categorias de horas de trabalho interpretáveis")
        print("  • Classificações de perfis de investimento")
        
        print("\n🔄 Pronto para implementação futura:")
        print("  • Tabelas de mapeamento de traduções")
        print("  • Seleção de idioma no dashboard")
        print("  • Apresentação de categorias localizadas")
        print("  • Relatórios multi-idioma")
        
        print("\n💡 Benefícios atuais:")
        print("  • Nomes de categorias mais intuitivos")
        print("  • Melhor compreensão do utilizador")
        print("  • Insights de dados mais claros")
        print("  • Apresentação profissional")
    
    def generate_readability_report(self, original_df, processed_df):
        """Gerar relatório completo de melhorias de legibilidade"""
        print("\n" + "="*80)
        print("🎓 RELATÓRIO COMPLETO DE MELHORIAS DE LEGIBILIDADE")
        print("="*80)
        
        # Resumo das melhorias
        self.show_enhancement_summary()
        
        # Comparação antes/depois
        self.compare_readability(original_df, processed_df)
        
        # Distribuição das novas features
        self.show_new_features_distribution(processed_df)
        
        # Preparação multilingual
        self.show_multilingual_readiness()
        
        # Insights adicionais
        print("\n" + "="*60)
        print("💡 INSIGHTS ADICIONAIS")
        print("="*60)
        
        # Análise de completude
        try:
            original_missing = original_df.isnull().sum().sum()
            processed_missing = processed_df.isnull().sum().sum()
            
            print(f"📊 Valores em falta originais: {original_missing:,}")
            print(f"📊 Valores em falta processados: {processed_missing:,}")
            
            if original_missing >= processed_missing:
                improvement = original_missing - processed_missing
                print(f"✅ Melhoria nos valores em falta: {improvement:,}")
            else:
                print("📝 Estrutura de dados mantida")
        except Exception as e:
            print(f"⚠️ Erro ao calcular completude: {e}")
        
        # Análise da variável target
        if 'salary' in processed_df.columns:
            try:
                target_dist = processed_df['salary'].value_counts()
                print(f"\n🎯 DISTRIBUIÇÃO DA VARIÁVEL ALVO:")
                for value, count in target_dist.items():
                    percentage = (count / len(processed_df)) * 100
                    print(f"  • {value}: {count:,} ({percentage:.1f}%)")
            except Exception as e:
                print(f"⚠️ Erro ao analisar variável alvo: {e}")
        
        print("\n🎉 Os dados estão agora mais legíveis e amigáveis ao utilizador!")
        print("💡 As categorias são descritivas e profissionais.")
        print("📊 Novas features interpretáveis melhoram as capacidades de análise.")
        print("🌍 A estrutura está pronta para futuro suporte multilingual.")

def show_readability_insights():
    """Função principal para mostrar melhorias de legibilidade"""
    try:
        # Carregar dados originais e processados
        original_path = Path("data/raw/4-Carateristicas_salario.csv")
        processed_path = Path("data/processed/data_processed.csv")
        
        if not original_path.exists():
            print("❌ Dados originais não encontrados!")
            return
        
        if not processed_path.exists():
            print("❌ Dados processados não encontrados! Execute o pipeline primeiro.")
            return
        
        # Carregar dados de forma segura
        try:
            original_df = pd.read_csv(original_path)
            processed_df = pd.read_csv(processed_path)
            
            print(f"✅ Dados carregados: {len(original_df)} → {len(processed_df)} registros")
            
        except Exception as e:
            print(f"❌ Erro ao carregar dados: {e}")
            return
        
        # Gerar insights
        insights = ReadabilityInsights()
        insights.generate_readability_report(original_df, processed_df)
        
    except Exception as e:
        print(f"❌ Erro ao gerar insights: {e}")
        logging.error(f"Erro em readability insights: {e}")

if __name__ == "__main__":
    show_readability_insights()