"""
Arquivo principal do Rxivonauta.
Coordena o pipeline de pesquisa acadêmica automatizada.
"""

import asyncio
import logging
import argparse
import os
import sys
from pathlib import Path

# Adicionar o diretório raiz ao PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass
from tqdm import tqdm

try:
    from config import settings
    from config.settings import Directories, LanguageConfig, LogConfig
    from src.agents.query_generator import QueryGenerator
    from src.agents.arxiv_searcher import ArxivSearcher
    from src.agents.content_analyzer import ContentAnalyzer
    from src.agents.content_reviewer import ContentReviewer
    from src.utils.data_processor import DataProcessor
except ImportError as e:
    print(f"Erro ao importar módulos: {e}")
    print(f"PYTHONPATH atual: {sys.path}")
    sys.exit(1)

@dataclass
class PipelineResult:
    """Estrutura para resultados do pipeline."""
    total_articles: int
    selected_articles: int
    processing_time: float
    queries_generated: int
    raw_file: str
    processed_file: str
    literature_review_file: str
    topic: str
    language: str

def setup_logging():
    """Configura o sistema de logging."""
    # Criar diretório de logs se não existir
    LogConfig.DIR.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, LogConfig.LEVEL),
        format=LogConfig.FORMAT,
        handlers=[
            logging.FileHandler(LogConfig.FILE),
            logging.StreamHandler()
        ]
    )

def setup_directories():
    """Cria estrutura de diretórios necessária."""
    directories = [
        Directories.DATA_DIR,
        Directories.RAW_DATA_DIR,
        Directories.PROCESSED_DATA_DIR,
        LogConfig.DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

class RxivonautaPipeline:
    """Pipeline principal do Rxivonauta."""
    
    def __init__(self):
        """Inicializa o pipeline com os agentes necessários."""
        self.logger = logging.getLogger(__name__)
        try:
            # Garantir que diretórios existam
            setup_directories()
            
            # Instanciar agentes
            self.query_generator = QueryGenerator()
            self.arxiv_searcher = ArxivSearcher()
            self.content_analyzer = ContentAnalyzer()
            self.content_reviewer = ContentReviewer()
            self.data_processor = DataProcessor()
            
            self.logger.info("Pipeline initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing pipeline: {str(e)}")
            raise

    async def run(
        self,
        research_topic: str,
        output_lang: str,
        output_dir: Optional[Path] = None,
        model: Optional[str] = None,
        args: argparse.Namespace = None
    ) -> PipelineResult:
        """
        Executa o pipeline completo.

        Args:
            research_topic: Tema de pesquisa
            output_lang: Idioma de saída
            output_dir: Diretório opcional para saída

        Returns:
            PipelineResult com resultados e estatísticas
            
        Raises:
            ValueError: Se os parâmetros forem inválidos
            Exception: Para outros erros durante a execução
        """
        try:
            # Validar entrada
            if not research_topic or not research_topic.strip():
                raise ValueError("Research topic cannot be empty")
                
            if not LanguageConfig.is_supported(output_lang):
                raise ValueError(f"Unsupported language: {output_lang}")
            
            output_dir = output_dir or Directories.PROCESSED_DATA_DIR
            output_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Starting pipeline for topic: {research_topic}")
            start_time = datetime.now()
            
            # 1. Set the model
            if args and args.model:
                settings.OpenRouterConfig.MODEL = args.model
            
            # Criar progress bar
            pbar = tqdm(total=6, desc="Pipeline Progress")

            # 2. Gerar queries
            self.logger.info("Generating search queries...")
            query_data = self.query_generator.generate_queries(research_topic)
            queries_df = self.data_processor.prepare_queries_data(query_data)
            pbar.update(1)

            # 2. Buscar artigos
            self.logger.info("Searching Arxiv...")
            articles = self.arxiv_searcher.search_articles(query_data.queries)
            articles_df = self.data_processor.process_arxiv_results(articles)
            pbar.update(1)

            # 3. Salvar resultados brutos
            raw_filename = f"raw_articles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            self.arxiv_searcher.save_to_raw(articles, raw_filename, output_dir)
            pbar.update(1)

            # 4. Analisar artigos
            self.logger.info("Analyzing articles...")
            results_df = self.content_analyzer.analyze_articles(
                articles_df=articles_df,
                original_topic=query_data.original_topic,
                english_topic=query_data.english_topic,
                queries=query_data.queries
            )
            
            results_df = self.data_processor.process_analysis_results(results_df)
            pbar.update(1)

            # 5. Validar e salvar resultados processados
            if not self.data_processor.validate_output_data(results_df):
                raise ValueError("Output data validation failed")

            output_filename = f"processed_articles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            self.content_analyzer.save_results(
                results_df=results_df,
                filename=output_filename,
                output_dir=output_dir
            )
            
            self.data_processor.save_processing_stats(output_dir)
            pbar.update(1)

            # 6. Gerar revisão de literatura
            self.logger.info("Generating literature review...")
            review_text = self.content_reviewer.generate_review(
                input_file=output_dir / output_filename,
                research_topic=research_topic,
                output_lang=output_lang,
                output_dir=output_dir
            )
            pbar.update(1)

            # Calcular tempo total
            total_time = (datetime.now() - start_time).total_seconds()
            
            # Fechar progress bar
            pbar.close()

            # Criar resultado
            result = PipelineResult(
                total_articles=len(articles_df),
                selected_articles=len(results_df),
                processing_time=total_time,
                queries_generated=len(query_data.queries),
                raw_file=raw_filename,
                processed_file=output_filename,
                literature_review_file=review_text,
                topic=research_topic,
                language=output_lang
            )

            self.logger.info(f"Pipeline completed in {total_time:.2f} seconds")
            return result

        except Exception as e:
            self.logger.error(f"Pipeline error: {str(e)}")
            raise

def print_summary(result: PipelineResult):
    """Imprime resumo da execução."""
    print("\nRxivonauta - Resumo da Execução")
    print("=" * 50)
    print(f"Tema: {result.topic}")
    print(f"Idioma: {LanguageConfig.SUPPORTED_LANGUAGES[result.language]}")
    print(f"Total de artigos encontrados: {result.total_articles}")
    print(f"Artigos selecionados: {result.selected_articles}")
    print(f"Queries geradas: {result.queries_generated}")
    print(f"Tempo de processamento: {result.processing_time:.2f} segundos")
    print(f"Arquivos gerados:")
    print(f"  - Raw: {result.raw_file}")
    print(f"  - Processado: {result.processed_file}")
    print(f"  - Revisão: {result.literature_review_file}")
    print("=" * 50)

async def main():
    """Função principal do script."""
    parser = argparse.ArgumentParser(
        description='Rxivonauta - Pesquisador Acadêmico Automatizado',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        'topic',
        type=str,
        help='Tema de pesquisa'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Diretório para saída',
        default=None
    )

    parser.add_argument(
        '--model',
        type=str,
        help='Modelo a ser utilizado',
        choices=[
            "google/gemini-2.0-pro-exp-02-05:free",
            "cognitivecomputations/dolphin3.0-r1-mistral-24b:free",
            "cognitivecomputations/dolphin3.0-mistral-24b:free",
            "openai/o3-mini-high",
            "openai/o3-mini",
            "openai/chatgpt-4o-latest",
            "openai/gpt-4o-mini",
            "google/gemini-2.0-flash-001",
            "google/gemini-2.0-flash-thinking-exp:free",
            "google/gemini-2.0-flash-lite-preview-02-05:free",
            "deepseek/deepseek-r1-distill-llama-70b:free",
            "deepseek/deepseek-r1-distill-qwen-32b",
            "deepseek/deepseek-r1:free",
            "qwen/qwen-plus",
            "qwen/qwen-max",
            "qwen/qwen-turbo",
            "qwen/qwen2.5-vl-72b-instruct:free",
            "mistralai/codestral-2501",
            "mistralai/mistral-7b-instruct:free",
            "mistralai/mistral-small-24b-instruct-2501:free",
            "anthropic/claude-3.5-haiku-20241022:beta",
            "anthropic/claude-3.5-sonnet",
            "perplexity/sonar-reasoning",
            "perplexity/sonar",
            "perplexity/llama-3.1-sonar-large-128k-online",
            "perplexity/llama-3.1-sonar-small-128k-chat",
            "nvidia/llama-3.1-nemotron-70b-instruct:free",
            "microsoft/phi-3-medium-128k-instruct:free",
            "meta-llama/llama-3.3-70b-instruct:free"
        ],
        default=None
    )

    parser.add_argument(
        '--output-lang',
        type=str,
        help='Idioma de saída',
        choices=list(LanguageConfig.SUPPORTED_LANGUAGES.keys()),
        default=LanguageConfig.DEFAULT_LANGUAGE
    )
    
    args = parser.parse_args()
    
    # Configurar logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Executar pipeline
        pipeline = RxivonautaPipeline()
        result = await pipeline.run(
            research_topic=args.topic,
            output_lang=args.output_lang,
            output_dir=args.output_dir,
            model=args.model,
            args=args
        )
        
        # Imprimir resumo
        print_summary(result)
        
    except KeyboardInterrupt:
        logger.info("Execution interrupted by user")
        print("\nExecução interrompida pelo usuário")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        print(f"\nErro durante a execução: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nPrograma finalizado pelo usuário")
    except Exception as e:
        print(f"\nErro fatal: {str(e)}")
        raise