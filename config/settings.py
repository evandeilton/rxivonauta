# Configurações do projeto
"""
Módulo de configurações centralizadas do Rxivonauta.

Este módulo gerencia todas as configurações do projeto, incluindo:
- Caminhos de diretórios
- Configurações de APIs
- Parâmetros do sistema
- Configurações de logging
- Campos de saída
- Suporte a idiomas
"""

from pathlib import Path
import os
from typing import Dict, List
from dotenv import load_dotenv

# Verifica e carrega variáveis de ambiente do arquivo .env
def load_env_file():
    """Carrega e valida variáveis de ambiente."""
    env_file = Path(__file__).resolve().parent.parent / '.env'
    if not env_file.exists():
        raise FileNotFoundError(
            "Arquivo .env não encontrado. Por favor, crie o arquivo .env "
            "baseado no .env.example"
        )
    load_dotenv(env_file)

load_env_file()

# Validação da API key do OpenRouter
def validate_api_key():
    """Valida a presença da API key do OpenRouter."""
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY não encontrada no arquivo .env. "
            "Esta chave é obrigatória para o funcionamento do sistema."
        )
    return api_key

# Estrutura de diretórios
class Directories:
    """Configuração de diretórios do projeto."""
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / 'data'
    RAW_DATA_DIR = DATA_DIR / 'raw'
    PROCESSED_DATA_DIR = DATA_DIR / 'processed'
    LOG_DIR = BASE_DIR / 'logs'

    @classmethod
    def create_dirs(cls):
        """Cria os diretórios necessários se não existirem."""
        for dir_path in [cls.DATA_DIR, cls.RAW_DATA_DIR, 
                        cls.PROCESSED_DATA_DIR, cls.LOG_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)

# Configurações da API OpenRouter
class OpenRouterConfig:
    """Configurações da API do OpenRouter."""
    API_KEY = validate_api_key()
    BASE_URL = 'https://openrouter.ai/api/v1'
    DEFAULT_MODEL = os.getenv('OPENROUTER_MODEL', 'perplexity/llama-3.1-sonar-small-128k-chat') #google/gemini-2.0-flash-001
    MODEL = DEFAULT_MODEL
    SITE_URL = 'https://github.com/evandeilton/rxivonauta'
    SITE_NAME = 'rxivonauta'
    
    # Configurações de taxa de requisição
    MAX_RETRIES = 3
    RETRY_DELAY = 5  # segundos
    TIMEOUT = 30     # segundos

    # Lista de modelos disponíveis
    MODELS = [
        "google/gemini-2.0-flash-001",
        "google/gemini-2.0-flash-thinking-exp:free",
        "google/gemini-2.0-flash-lite-preview-02-05:free",
        "google/gemini-2.0-pro-exp-02-05:free",
        "mistralai/mistral-7b-instruct:free",
        "mistralai/mistral-small-24b-instruct-2501:free",
        "anthropic/claude-3.5-haiku-20241022:beta",
        "openai/o3-mini",
        "openai/o3-mini-high",
        "perplexity/llama-3.1-sonar-small-128k-chat",
        "microsoft/phi-3-medium-128k-instruct:free",
        "meta-llama/llama-3.3-70b-instruct:free"
    ]

    @classmethod
    def validate_model(cls, model: str) -> bool:
        """Valida se o modelo fornecido está na lista de modelos disponíveis."""
        return model in cls.MODELS

# Configurações do sistema
class SystemConfig:
    """Configurações gerais do sistema."""
    MAX_QUERIES = 5              # número de queries a serem geradas
    MAX_SELECTED_PER_QUERY = 2   # número de artigos selecionados por query
    CHUNK_SIZE = 1000            # tamanho do chunk para processamento de dados
    CACHE_ENABLED = True         # habilita cache de requisições
    CACHE_TTL = 3600             # tempo de vida do cache em segundos

# Configurações do Arxiv
class ArxivConfig:
    """Configurações da API do Arxiv."""
    MAX_RESULTS = 5     # número de artigos por query
    RATE_LIMIT = 3      # segundos entre requisições
    SORT_BY = 'SubmittedDate'     # critério de ordenação
    SORT_ORDER = 'descending'     # ordem de classificação
    MAX_AGE_DAYS = None  # idade máxima dos artigos em dias (None = sem limite)
    
    # Categorias padrão para busca
    CATEGORIES = [
        'cs.AI',        # Inteligência Artificial
        'cs.CL',        # Computação Linguística
        'cs.LG',        # Aprendizado de Máquina
        'stat.ML',      # Aprendizado de Máquina (Estatística)
        'stat.ME',      # Metodologia
        'stat.TH',      # Estatística Teórica
        'math.ST',      # Estatística
        'q-bio.QM'      # Métodos Quantitativos
    ]

# Configurações de logging
class LogConfig:
    """Configurações do sistema de logging."""
    DIR = Directories.LOG_DIR
    FILE = DIR / 'rxivonauta.log'
    FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    MAX_SIZE = 10 * 1024 * 1024  # 10MB
    BACKUP_COUNT = 5             # número de arquivos de backup

# Campos de saída
class OutputConfig:
    """Configuração dos campos de saída."""
    # Campos completos incluindo análise
    OUTPUT_FIELDS = [
        'title',
        'authors',
        'published',
        'updated',
        'summary',
        'pdf_url',
        'arxiv_id',
        'categories',
        'doi',
        'journal_ref',
        'comment',
        'query_id',
        'relevance_score',
        'selection_rationale',
        'key_aspects',
        'methodology_score',
        'impact_score',
        'language_notes',
        'final_score',
        'aspect_bonus'
    ]

    # Campos esperados do Arxiv
    ARXIV_OUTPUT_FIELDS = [
        'title',
        'authors',
        'published',
        'updated',
        'summary',
        'pdf_url',
        'arxiv_id',
        'categories',
        'doi',
        'journal_ref',
        'comment',
        'query_id',
    ]

# Configurações de idiomas
class LanguageConfig:
    """Configuração de suporte a idiomas."""
    SUPPORTED_LANGUAGES = {
        "pt-BR": "🇧🇷 Português (Brasil)",
        "en-US": "🇺🇸 Inglês (EUA)",
        "es-ES": "🇪🇸 Espanhol (Espanha)",
        "fr-FR": "🇫🇷 Francês (França)",
        "it-IT": "🇮🇹 Italiano (Itália)",
        "ru-RU": "🇷🇺 Russo (Rússia)",
        "ar-AE": "🇦🇪 Árabe (Emirados)",
        "zh-HK": "🇭🇰 Chinês (Hong Kong)"
    }
    DEFAULT_LANGUAGE = "pt-BR"

    @classmethod
    def is_supported(cls, lang_code: str) -> bool:
        """Verifica se um código de idioma é suportado."""
        return lang_code in cls.SUPPORTED_LANGUAGES

# Criar diretórios necessários
Directories.create_dirs()

# Exportar configurações para uso no projeto
__all__ = [
    'Directories',
    'OpenRouterConfig',
    'SystemConfig',
    'ArxivConfig',
    'LogConfig',
    'OutputConfig',
    'LanguageConfig'
]
