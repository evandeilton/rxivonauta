# Processamento de dados

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass

from config import settings
from config.settings import OutputConfig
from ..agents.query_generator import QueryResult

logger = logging.getLogger(__name__)

@dataclass
class ProcessingStats:
    """Estatísticas do processamento de dados."""
    total_articles: int
    articles_per_query: Dict[str, int]
    missing_fields: Dict[str, int]
    selected_articles: int
    avg_relevance_score: float
    processing_time: float
    timestamp: datetime = datetime.now()

class DataProcessor:
    """
    Responsável pelo processamento, validação e transformação dos dados.
    """
    
    def __init__(self):
        """Inicializa o processador de dados."""
        self.stats = None
        self._processing_start = None
        
    def prepare_queries_data(self, query_data: Union[Dict, QueryResult]) -> pd.DataFrame:
        """
        Prepara os dados das queries para processamento.
        
        Args:
            query_data: Dicionário ou QueryResult contendo dados das queries
            
        Returns:
            DataFrame com as queries processadas
        Raises:
            ValueError: Se faltar campos obrigatórios ou dados inválidos
        """
        try:
            # Handle QueryResult type
            if isinstance(query_data, QueryResult):
                queries = query_data.queries
            else:
                if not query_data or 'queries' not in query_data:
                    raise ValueError("Query data is empty or missing 'queries' key")
                queries = query_data['queries']
            
            if not queries:
                raise ValueError("Empty queries list")
                
            # Criar DataFrame
            df = pd.DataFrame(queries)
            
            # Validar estrutura
            required_columns = {'query_id', 'query_text', 'rationale', 'aspect'}
            missing_columns = required_columns - set(df.columns)
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            # Validar tipos de dados
            if not df['query_id'].apply(lambda x: isinstance(x, str)).all():
                raise ValueError("All query_ids must be strings")
            # Remover duplicatas
            df = df.drop_duplicates(subset=['query_id'])
                
            return df
            
        except Exception as e:
            logger.error(f"Error preparing queries data: {str(e)}")
            raise
            
    def process_arxiv_results(self, articles: List[Dict]) -> pd.DataFrame:
        """
        Processa os resultados do Arxiv.
        
        Args:
            articles: Lista de artigos do Arxiv
            
        Returns:
            DataFrame com artigos processados

        Raises:
            ValueError: Se a lista de artigos estiver vazia ou inválida
        """
        try:
            if not articles:
                raise ValueError("Empty articles list")

            self._processing_start = datetime.now()
            
            # Converter para DataFrame
            df = pd.DataFrame(articles)
            
            if df.empty:
                raise ValueError("No valid articles to process")

            # Verificar campos obrigatórios
            required_fields = set(OutputConfig.ARXIV_OUTPUT_FIELDS)
            missing_fields = required_fields - set(df.columns)
            if missing_fields:
                raise ValueError(f"Missing required fields in articles: {missing_fields}")
            
            # Iniciar contagem de campos ausentes
            missing_fields = {field: df[field].isna().sum()
                            for field in OutputConfig.ARXIV_OUTPUT_FIELDS}

            # Processar campos específicos
            df = self._process_text_fields(df)
            df = self._process_date_fields(df)
            df = self._process_list_fields(df)
            
            # Calcular estatísticas
            articles_per_query = df['query_id'].value_counts().to_dict()
            
            self.stats = ProcessingStats(
                total_articles=len(df),
                articles_per_query=articles_per_query,
                missing_fields=missing_fields,
                selected_articles=0,  # Será atualizado após seleção
                avg_relevance_score=0.0,  # Será atualizado após análise
                processing_time=0.0  # Será atualizado ao finalizar
            )
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing Arxiv results: {str(e)}")
            raise
            
    def process_analysis_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processa os resultados da análise de conteúdo.
        
        Args:
            df: DataFrame com análises. Deve conter as colunas:
               - relevance_score, methodology_score, impact_score
               - selection_rationale, language_notes
            
        Returns:
            DataFrame processado

        Raises:
            ValueError: Se o DataFrame estiver vazio ou campos obrigatórios ausentes
        """
        try:
            if df.empty:
                raise ValueError("Empty DataFrame provided")

            required_cols = {
                'relevance_score', 'methodology_score', 'impact_score',
                'selection_rationale', 'language_notes'
            }
            missing_cols = required_cols - set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Processar scores
            df = self._process_scores(df)
            
            # Processar campos de texto da análise
            df = self._process_analysis_text(df)
            
            # Atualizar estatísticas
            if self.stats:
                self.stats.selected_articles = len(df)
                self.stats.avg_relevance_score = df['relevance_score'].mean()
                self.stats.processing_time = (
                    datetime.now() - self._processing_start
                ).total_seconds()
                
            return df
            
        except Exception as e:
            logger.error(f"Error processing analysis results: {str(e)}")
            raise
            
    def _process_text_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processa campos de texto do DataFrame.
        
        Args:
            df: DataFrame com campos de texto
            
        Returns:
            DataFrame com campos de texto processados
        """
        text_fields = ['title', 'summary', 'comment']
        for field in text_fields:
            if field in df.columns:
                # Tratar valores nulos
                df[field] = df[field].fillna('')
                # Remover quebras de linha extras
                df[field] = df[field].str.replace(r'\s+', ' ', regex=True)
                # Normalizar espaços
                df[field] = df[field].str.strip()
                
        return df
        
    def _process_date_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processa campos de data do DataFrame.
        
        Args:
            df: DataFrame com campos de data
            
        Returns:
            DataFrame com campos de data processados
        """
        date_fields = ['published', 'updated']
        for field in date_fields:
            if field in df.columns:
                # Tratar valores nulos
                df[field] = pd.to_datetime(df[field], utc=True, errors='coerce')
                # Converter para formato string padrão
                df[field] = df[field].dt.strftime('%Y-%m-%d %H:%M:%S')
                
        return df
        
    def _process_list_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processa campos que contêm listas.
        
        Args:
            df: DataFrame com campos de lista
            
        Returns:
            DataFrame com campos de lista processados
        """
        list_fields = ['authors', 'categories', 'key_aspects']
        for field in list_fields:
            if field in df.columns:
                # Tratar valores nulos
                df[field] = df[field].fillna(value=np.nan)
                # Converter listas para strings separadas por ponto e vírgula
                df[field] = df[field].apply(
                    lambda x: '; '.join(x) if isinstance(x, list) else x
                )
                
        return df
        
    def _process_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processa campos de score do DataFrame.
        
        Args:
            df: DataFrame com campos de score
            
        Returns:
            DataFrame com campos de score processados
        """
        score_fields = ['relevance_score', 'methodology_score', 'impact_score']
        for field in score_fields:
            if field in df.columns:
                # Converter para numérico
                df[field] = pd.to_numeric(df[field], errors='coerce')
                # Garantir que scores estão entre 0 e 1
                df[field] = df[field].clip(0, 1)
                # Arredondar para 3 casas decimais
                df[field] = df[field].round(3)
                
        return df
        
    def _process_analysis_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processa campos de texto da análise.
        
        Args:
            df: DataFrame com campos de texto da análise
            
        Returns:
            DataFrame com campos de texto processados
        """
        text_fields = ['selection_rationale', 'language_notes']
        for field in text_fields:
            if field in df.columns:
                # Tratar valores nulos
                df[field] = df[field].fillna('')
                # Limpar formatação
                df[field] = df[field].str.strip()
                df[field] = df[field].str.replace(r'\s+', ' ', regex=True)
                
        return df
        
    def save_processing_stats(self, output_dir: Optional[Path] = None):
        """
        Salva as estatísticas de processamento.
        
        Args:
            output_dir: Diretório de saída opcional. Se não fornecido,
                       usa o diretório padrão das configurações.
        """
        if not self.stats:
            logger.warning("No processing stats available")
            return
            
        try:
            # Preparar dados para salvar
            stats_dict = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_articles': self.stats.total_articles,
                'articles_per_query': self.stats.articles_per_query,
                'missing_fields': self.stats.missing_fields,
                'selected_articles': self.stats.selected_articles,
                'avg_relevance_score': round(self.stats.avg_relevance_score, 3),
                'processing_time_seconds': round(self.stats.processing_time, 2)
            }
            
            # Definir diretório de saída
            output_dir = output_dir or settings.PROCESSED_DATA_DIR
            output_dir.mkdir(parents=True, exist_ok=True)
            stats_file = output_dir / 'processing_stats.json'
            
            # Salvar estatísticas
            pd.Series(stats_dict).to_json(stats_file)
            logger.info(f"Processing stats saved to {stats_file}")
            
        except Exception as e:
            logger.error(f"Error saving processing stats: {str(e)}")
            raise
            
    def validate_output_data(self, df: pd.DataFrame) -> bool:
        """
        Valida o DataFrame final antes da saída.
        
        Args:
            df: DataFrame a ser validado
            
        Returns:
            bool indicando se os dados são válidos
        """
        try:
            if df.empty:
                logger.error("Empty DataFrame")
                return False

            # Verificar campos obrigatórios
            required_fields = set(OutputConfig.OUTPUT_FIELDS)
            missing_fields = required_fields - set(df.columns)
            
            if missing_fields:
                logger.error(f"Missing required fields: {missing_fields}")
                return False
                
            # Verificar valores nulos
            null_counts = df[list(required_fields)].isnull().sum()
            if null_counts.any():
                logger.warning(f"Null values found: {null_counts[null_counts > 0]}")
                
            # Verificar tipos de dados
            expected_types = {
                'relevance_score': 'float64',
                'arxiv_id': 'object',
                'query_id': 'object'
            }
            
            for field, expected_type in expected_types.items():
                if field in df.columns and df[field].dtype != expected_type:
                    logger.error(f"Invalid type for {field}: expected {expected_type}, got {df[field].dtype}")
                    return False
                
            # Verificar valores fora do intervalo
            for score_field in ['relevance_score', 'methodology_score', 'impact_score']:
                if score_field in df.columns:
                    if not (df[score_field].between(0, 1, inclusive='both').all()):
                        logger.error(f"Values in {score_field} outside valid range [0,1]")
                        return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Error validating output data: {str(e)}")
            return False