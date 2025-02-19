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
    from config.settings import OpenRouterConfig, ArxivConfig, SystemConfig
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
        temperature: float = 0.7,
        min_score_threshold: float = 0.6,
        model: Optional[str] = None
    ) -> PipelineResult:
        """
        Executa o pipeline completo.

        Args:
            research_topic: Tema de pesquisa
            output_lang: Idioma de saída
            output_dir: Diretório opcional para saída
            temperature: Temperatura para geração de texto LLM
            min_score_threshold: Pontuação mínima de relevância para seleção de artigos

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

def setup_argparse():
    """Configure command line argument parser with comprehensive defaults."""
    parser = argparse.ArgumentParser(
        description='Rxivonauta - Automated Academic Researcher',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        'topic',
        type=str,
        help='Research topic'
    )
    
    # Optional arguments with defaults
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Directories.PROCESSED_DATA_DIR,
        help='Output directory'
    )
    parser.add_argument(
        '--output-lang',
        type=str,
        choices=list(LanguageConfig.SUPPORTED_LANGUAGES.keys()),
        default=LanguageConfig.DEFAULT_LANGUAGE,
        help='Output language for the review'
    )
    
    # Model selection with default
    parser.add_argument(
        '--model',
        type=str,
        choices=[
            "cognitivecomputations/dolphin3.0-r1-mistral-24b:free",
            "cognitivecomputations/dolphin3.0-mistral-24b:free",
            "openai/o3-mini-high",
            "openai/o3-mini",
            "openai/chatgpt-4o-latest",
            "openai/gpt-4o-mini",
            "google/gemini-2.0-flash-001",
            "google/gemini-2.0-flash-thinking-exp:free",
            "google/gemini-2.0-flash-lite-preview-02-05:free",
            "google/gemini-2.0-pro-exp-02-05:free",
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
        default="google/gemini-2.0-pro-exp-02-05:free",
        help='LLM model to use for text generation'
    )
    
    # Search configuration with defaults
    parser.add_argument(
        '--max-queries',
        type=int,
        default=SystemConfig.MAX_QUERIES,
        help='Maximum number of search queries to generate (default: 5)'
    )
    
    parser.add_argument(
        '--max-results-per-query',
        type=int,
        default=ArxivConfig.MAX_RESULTS,
        help='Maximum number of results per query (default: 5)'
    )
    
    parser.add_argument(
        '--max-age-days',
        type=int,
        default=365,  # Set default to 1 year
        help='Maximum age of articles in days (default: 365, use 0 for no limit)'
    )
    
    parser.add_argument(
        '--categories',
        type=str,
        nargs='+',
        default=ArxivConfig.CATEGORIES,
        help='Arxiv categories to search in (default: cs.AI cs.CL cs.LG stat.ML)'
    )
    
    # Analysis configuration with defaults
    parser.add_argument(
        '--min-score',
        type=float,
        default=0.6,
        help='Minimum relevance score for article selection (0-1)'
    )
    
    parser.add_argument(
        '--max-selected',
        type=int,
        default=SystemConfig.MAX_SELECTED_PER_QUERY,
        help='Maximum number of selected articles per query (default: 2)'
    )
    
    # Advanced options with defaults
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Temperature for LLM text generation (0-1)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=SystemConfig.CHUNK_SIZE,
        help='Batch size for processing articles (default: 1000)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        default=False,
        help='Enable debug logging'
    )

    parser.add_argument(
        '--retry-attempts',
        type=int,
        default=OpenRouterConfig.MAX_RETRIES,
        help='Number of retry attempts for API calls (default: 3)'
    )

    parser.add_argument(
        '--retry-delay',
        type=int,
        default=OpenRouterConfig.RETRY_DELAY,
        help='Delay between retry attempts in seconds (default: 5)'
    )

    parser.add_argument(
        '--timeout',
        type=int,
        default=OpenRouterConfig.TIMEOUT,
        help='API call timeout in seconds (default: 30)'
    )
    
    parser.add_argument(
        '--rate-limit',
        type=float,
        default=ArxivConfig.RATE_LIMIT,
        help='Rate limit for Arxiv API requests in seconds (default: 3)'
    )
    
    return parser

def validate_args(args):
    """Validate command line arguments."""
    # Validate numerical ranges
    if not 0 <= args.temperature <= 1:
        raise ValueError("Temperature must be between 0 and 1")
    if not 0 <= args.min_score <= 1:
        raise ValueError("Min score must be between 0 and 1")
    if args.max_queries < 1:
        raise ValueError("Max queries must be at least 1")
    if args.max_results_per_query < 1:
        raise ValueError("Max results per query must be at least 1")
    if args.max_age_days < 0:
        raise ValueError("Max age days cannot be negative")
    
    # Validate categories
    valid_categories = set(ArxivConfig.CATEGORIES)
    invalid_cats = [cat for cat in args.categories if cat not in valid_categories]
    if invalid_cats:
        raise ValueError(f"Invalid Arxiv categories: {invalid_cats}")

async def main():
    """Main function with enhanced argument handling."""
    parser = setup_argparse()
    args = parser.parse_args()
    
    try:
        # Validate arguments
        validate_args(args)
        
        # Update logging level if debug is enabled
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Setup logging
        setup_logging()
        logger = logging.getLogger(__name__)
        
        # Update configurations based on arguments
        config_updates = {
            'max_queries': args.max_queries,
            'max_selected': args.max_selected,
            'batch_size': args.batch_size,
            'temperature': args.temperature,
            'model': args.model,
            'max_retries': args.retry_attempts,
            'retry_delay': args.retry_delay,
            'timeout': args.timeout
        }
        
        # Initialize pipeline with updated configurations
        pipeline = RxivonautaPipeline()
        pipeline.query_generator = QueryGenerator(
            api_key=OpenRouterConfig.API_KEY,
            temperature=args.temperature
        )
        pipeline.arxiv_searcher = ArxivSearcher(
            max_results=args.max_results_per_query,
            rate_limit=args.rate_limit,
            max_age_days=args.max_age_days
        )
        pipeline.content_analyzer = ContentAnalyzer(
            min_score_threshold=args.min_score,
            batch_size=args.batch_size
        )
        pipeline.content_reviewer = ContentReviewer(
            api_key=OpenRouterConfig.API_KEY,
            temperature=args.temperature
        )
        
        # Execute pipeline
        result = await pipeline.run(
            research_topic=args.topic,
            output_lang=args.output_lang,
            output_dir=args.output_dir,
            model=args.model
        )
        
        # Print summary
        print_summary(result)
        
    except ValueError as e:
        logger.error(f"Configuration error: {str(e)}")
        print(f"\nConfiguration error: {str(e)}")
        sys.exit(1)
        
    except KeyboardInterrupt:
        logger.info("Execution interrupted by user")
        print("\nExecution interrupted by user")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        print(f"\nError during execution: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nPrograma finalizado pelo usuário")
    except Exception as e:
        print(f"\nErro fatal: {str(e)}")
        
        raise