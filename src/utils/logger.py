#!/usr/bin/env python3
"""
🔧 Sistema de Logging Avançado para Pipeline de Análise Salarial
Suporte completo para DBSCAN, APRIORI, FP-GROWTH, ECLAT
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import json

class ColoredFormatter(logging.Formatter):
    """Formatter colorido para melhor visualização no terminal"""
    
    # Códigos de cores ANSI
    COLORS = {
        'DEBUG': '\033[36m',    
        'INFO': '\033[32m',     
        'WARNING': '\033[33m',  
        'ERROR': '\033[31m',   
        'CRITICAL': '\033[35m',
        'RESET': '\033[0m'
    }
    
    def format(self, record):
        # Aplicar cor baseada no nível
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Formatação colorida
        record.levelname = f"{color}{record.levelname:8s}{reset}"
        
        return super().format(record)

class PipelineLogger:
    """Logger especializado para pipeline de análise salarial"""
    
    def __init__(self, name: str = "SalaryAnalysis"):
        self.name = name
        self.logger = None
        self.log_file = None
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def setup(self, 
              log_level: str = "INFO",
              log_file: Optional[str] = None,
              enable_colors: bool = True,
              enable_file_rotation: bool = True) -> logging.Logger:
        """
        Configurar sistema de logging avançado
        
        Args:
            log_level: Nível de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Caminho do arquivo de log (opcional)
            enable_colors: Habilitar cores no console
            enable_file_rotation: Habilitar rotação de arquivos de log
        
        Returns:
            Logger configurado
        """
        # Criar diretório de logs
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Configurar logger principal
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        
        # Limpar handlers existentes
        self._clear_handlers()
        
        # Configurar formatters
        console_formatter = ColoredFormatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ) if enable_colors else logging.Formatter(
            '%(asctime)s | %(levelname)8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Handler para console
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)
        self.logger.addHandler(console_handler)
        
        # Handler para arquivo
        if log_file:
            self.log_file = log_file
        else:
            self.log_file = log_dir / f"pipeline_{self.session_id}.log"
        
        if enable_file_rotation:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                self.log_file,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )
        else:
            file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        
        # Log inicial
        self.logger.info(f"🚀 Logger inicializado para {self.name}")
        self.logger.info(f"📁 Logs salvos em: {self.log_file}")
        self.logger.info(f"🔍 Nível de log: {log_level}")
        self.logger.info(f"🆔 Session ID: {self.session_id}")
        
        return self.logger
    
    def _clear_handlers(self):
        """Limpar handlers existentes para evitar duplicação"""
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
    
    def log_algorithm_start(self, algorithm: str, params: Dict[str, Any] = None):
        """Log específico para início de algoritmo"""
        self.logger.info(f"🎯 INICIANDO {algorithm.upper()}")
        self.logger.info("=" * 60)
        if params:
            self.logger.info(f"📋 Parâmetros: {json.dumps(params, indent=2, ensure_ascii=False)}")
    
    def log_algorithm_end(self, algorithm: str, results: Dict[str, Any]):
        """Log específico para fim de algoritmo"""
        self.logger.info(f"✅ {algorithm.upper()} CONCLUÍDO")
        if 'execution_time' in results:
            self.logger.info(f"⏱️ Tempo de execução: {results['execution_time']:.2f}s")
        if 'results_count' in results:
            self.logger.info(f"📊 Resultados gerados: {results['results_count']}")
        self.logger.info("=" * 60)
    
    def log_data_info(self, df, source: str = "Unknown"):
        """Log informações sobre os dados carregados"""
        if df is not None:
            self.logger.info(f"📊 DADOS CARREGADOS ({source})")
            self.logger.info(f"   • Registros: {len(df):,}")
            self.logger.info(f"   • Colunas: {len(df.columns)}")
            self.logger.info(f"   • Fonte: {source}")
            self.logger.info(f"   • Memória: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
            
            # Tipos de dados
            numeric_cols = len(df.select_dtypes(include=['number']).columns)
            categorical_cols = len(df.select_dtypes(include=['object']).columns)
            self.logger.info(f"   • Numéricas: {numeric_cols} | Categóricas: {categorical_cols}")
        else:
            self.logger.error("❌ Dados não carregados")
    
    def log_error_with_context(self, error: Exception, context: str = ""):
        """Log de erro com contexto adicional"""
        self.logger.error(f"❌ ERRO: {context}")
        self.logger.error(f"   • Tipo: {type(error).__name__}")
        self.logger.error(f"   • Mensagem: {str(error)}")
        
        # Stack trace detalhado no arquivo de log
        import traceback
        self.logger.debug("📋 STACK TRACE COMPLETO:")
        self.logger.debug(traceback.format_exc())
    
    def log_performance_metrics(self, metrics: Dict[str, Any]):
        """Log de métricas de performance"""
        self.logger.info("📈 MÉTRICAS DE PERFORMANCE:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if 'time' in key.lower():
                    self.logger.info(f"   • {key}: {value:.2f}s")
                elif 'memory' in key.lower():
                    self.logger.info(f"   • {key}: {value:.1f} MB")
                else:
                    self.logger.info(f"   • {key}: {value}")
            else:
                self.logger.info(f"   • {key}: {value}")
    
    def create_algorithm_logger(self, algorithm: str) -> logging.Logger:
        """Criar logger específico para um algoritmo"""
        algo_logger = logging.getLogger(f"{self.name}.{algorithm}")
        algo_logger.setLevel(self.logger.level)
        
        # Herdar handlers do logger principal
        for handler in self.logger.handlers:
            algo_logger.addHandler(handler)
        
        # Evitar propagação dupla
        algo_logger.propagate = False
        
        return algo_logger

# Função de compatibilidade com código existente
def setup_logging(log_level: str = "INFO", 
                 log_file: Optional[str] = None,
                 enable_colors: bool = True) -> logging.Logger:
    """
    Função de compatibilidade para manter código existente funcionando
    
    Args:
        log_level: Nível de log
        log_file: Arquivo de log (opcional)
        enable_colors: Habilitar cores
    
    Returns:
        Logger configurado
    """
    pipeline_logger = PipelineLogger("SalaryAnalysis")
    return pipeline_logger.setup(
        log_level=log_level,
        log_file=log_file,
        enable_colors=enable_colors
    )

# Logger global para uso rápido
_global_logger = None

def get_logger(name: str = "SalaryAnalysis") -> logging.Logger:
    """Obter logger global configurado"""
    global _global_logger
    if _global_logger is None:
        _global_logger = setup_logging()
    return _global_logger

def log_algorithm_execution(algorithm: str):
    """Decorator para log automático de execução de algoritmos"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger()
            start_time = datetime.now()
            
            logger.info(f"🎯 Executando {algorithm}...")
            
            try:
                result = func(*args, **kwargs)
                
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()
                
                logger.info(f"✅ {algorithm} concluído em {execution_time:.2f}s")
                return result
                
            except Exception as e:
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()
                
                logger.error(f"❌ {algorithm} falhou após {execution_time:.2f}s: {e}")
                raise
                
        return wrapper
    return decorator

# Configuração padrão para o projeto
def setup_project_logging():
    """Configurar logging padrão para todo o projeto"""
    pipeline_logger = PipelineLogger("SalaryAnalysisPipeline")
    logger = pipeline_logger.setup(
        log_level="INFO",
        enable_colors=True,
        enable_file_rotation=True
    )
    
    # Configurar loggers específicos para cada módulo
    modules = ['clustering', 'association_rules', 'ml_pipeline', 'database']
    
    for module in modules:
        module_logger = pipeline_logger.create_algorithm_logger(module)
        module_logger.info(f"🔧 Logger configurado para módulo {module}")
    
    return logger

if __name__ == "__main__":
    logger = setup_project_logging()
    
    logger.info("🧪 Testando sistema de logging...")
    logger.debug("Debug message")
    logger.warning("Warning message")
    logger.error("Error message")
    
    # Teste de decorator
    @log_algorithm_execution("TESTE_ALGORITHM")
    def test_function():
        import time
        time.sleep(1)
        return "Resultado teste"
    
    result = test_function()
    logger.info(f"Resultado: {result}")