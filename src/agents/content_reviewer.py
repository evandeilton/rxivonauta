"""
Agente para geração de revisões de literatura.
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass

from config import settings, prompts
from ..utils.api_client import OpenRouterClient

logger = logging.getLogger(__name__)

@dataclass
class ReviewSection:
    """Estrutura para seções da revisão de literatura."""
    title: str
    content: str
    subsections: List[Dict[str, str]]

class ContentReviewer:
    """
    Agente responsável por gerar revisões de literatura abrangentes.
    """
    
    def __init__(
        self,
        api_key: str = settings.OpenRouterConfig.API_KEY,
        temperature: float = 0.7
    ):
        """
        Inicializa o revisor de conteúdo.
        
        Args:
            api_key: Chave da API do OpenRouter
            temperature: Temperatura para geração de texto (0-1)
            
        Raises:
            ValueError: Se a chave da API não for fornecida
        """
        if not api_key:
            raise ValueError("OpenRouter API key is required")
            
        self.client = OpenRouterClient(api_key)
        self.temperature = min(max(temperature, 0), 1)
        
    def generate_review(
        self, 
        input_file: Path,
        research_topic: str,
        output_lang: str,
        output_dir: Optional[Path] = None
    ) -> str:
        """
        Gera uma revisão de literatura completa.
        
        Args:
            input_file: Caminho para o arquivo CSV de artigos
            research_topic: Tema da pesquisa
            output_lang: Idioma de saída
            output_dir: Diretório opcional para salvar a revisão
            
        Returns:
            Revisão de literatura em formato Markdown
            
        Raises:
            ValueError: Se os dados de entrada forem inválidos
            FileNotFoundError: Se o arquivo não existir
        """
        try:
            # Validar entrada
            if not input_file.exists():
                raise FileNotFoundError(f"Input file not found: {input_file}")
                
            if not research_topic or not research_topic.strip():
                raise ValueError("Research topic cannot be empty")
                
            # Validar idioma
            if not self._validate_language(output_lang):
                raise ValueError(f"Unsupported language: {output_lang}")
            
            # Ler e validar artigos
            articles_df = self._read_articles(input_file)
            
            logger.info(f"Generating review for {len(articles_df)} articles")
            
            # Preparar texto dos artigos
            articles_text = self._prepare_articles_text(articles_df)
            
            # Gerar revisão
            review_text = self._generate_review_text(
                articles_text=articles_text,
                research_topic=research_topic,
                output_lang=output_lang
            )
            
            # Processar e formatar revisão
            formatted_review = self._format_review(review_text)
            
            # Salvar se diretório fornecido
            if output_dir:
                self._save_review(formatted_review, output_dir)
                
            return formatted_review
            
        except Exception as e:
            logger.error(f"Error generating literature review: {str(e)}")
            raise
            
    def _read_articles(self, input_file: Path) -> pd.DataFrame:
        """
        Lê e valida os artigos do arquivo CSV.
        
        Args:
            input_file: Caminho do arquivo
            
        Returns:
            DataFrame com artigos processados
            
        Raises:
            ValueError: Se os dados forem inválidos
        """
        try:
            df = pd.read_csv(input_file)
            
            if df.empty:
                raise ValueError("Empty articles file")
            
            required_columns = {
                'title', 'authors', 'summary', 'key_aspects',
                'selection_rationale'
            }
            
            missing_columns = required_columns - set(df.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
                
            return df
            
        except pd.errors.EmptyDataError:
            raise ValueError("Empty or invalid CSV file")
            
    def _validate_language(self, lang_code: str) -> bool:
        """
        Verifica se o idioma é suportado.
        
        Args:
            lang_code: Código do idioma
            
        Returns:
            bool indicando se o idioma é suportado
        """
        return lang_code in settings.LanguageConfig.SUPPORTED_LANGUAGES
            
    def _prepare_articles_text(self, articles_df: pd.DataFrame) -> str:
        """
        Prepara o texto dos artigos para geração da revisão.
        
        Args:
            articles_df: DataFrame com artigos
            
        Returns:
            Texto formatado em Markdown
        """
        articles_text = ""
        
        for idx, article in articles_df.iterrows():
            # Limpar e validar campos
            title = article['title'].strip()
            authors = article['authors'].strip()
            summary = article['summary'].strip()
            key_aspects = article.get('key_aspects', '').strip()
            rationale = article.get('selection_rationale', '').strip()
            
            if not all([title, authors, summary]):
                logger.warning(f"Skipping article {idx + 1} due to missing data")
                continue
            
            # Formatar em Markdown
            article_text = f"""
### Article {idx + 1}

**Title**: {title}
**Authors**: {authors}
**Summary**: {summary}
**Key Aspects**: {key_aspects}
**Selection Rationale**: {rationale}

---
"""
            articles_text += article_text
            
        if not articles_text.strip():
            raise ValueError("No valid articles to process")
            
        return articles_text
        
    def _generate_review_text(
        self,
        articles_text: str,
        research_topic: str,
        output_lang: str
    ) -> str:
        """
        Gera o texto da revisão usando LLM.
        
        Args:
            articles_text: Texto dos artigos formatado
            research_topic: Tema da pesquisa
            output_lang: Idioma de saída
            
        Returns:
            Texto da revisão gerado
            
        Raises:
            ValueError: Se a geração falhar
        """
        try:
            # Preparar prompt
            prompt = prompts.format_review_prompt(
                articles_text=articles_text,
                research_topic=research_topic,
                target_language=output_lang
            )
            
            # Gerar revisão
            review_text = self.client.generate_text(
                prompt=prompt,
                system_message=prompts.SystemMessages.CONTENT_REVIEWER,
                temperature=self.temperature
            )
            
            if not review_text or not review_text.strip():
                raise ValueError("Empty review generated")
                
            return review_text
            
        except Exception as e:
            logger.error(f"Error in review generation: {str(e)}")
            raise
            
    def _format_review(self, review_text: str) -> str:
        """
        Formata e valida a revisão gerada.
        
        Args:
            review_text: Texto bruto da revisão
            
        Returns:
            Texto formatado em Markdown
            
        Raises:
            ValueError: Se a formatação falhar
        """
        try:
            # Verificar seções obrigatórias
            required_sections = [
                "## Introduction",
                "## Theoretical Framework",
                "## Methodology Analysis",
                "## Results Synthesis",
                "## Future Directions",
                "## Conclusion"
            ]
            
            for section in required_sections:
                if section not in review_text:
                    logger.warning(f"Missing section: {section}")
            
            # Adicionar cabeçalho
            header = f"""# Literature Review
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""
            formatted_text = header + review_text
            
            # Garantir quebras de linha consistentes
            formatted_text = formatted_text.replace('\r\n', '\n')
            
            return formatted_text
            
        except Exception as e:
            logger.error(f"Error formatting review: {str(e)}")
            raise
        
    def _save_review(self, review_text: str, output_dir: Path):
        """
        Salva a revisão em arquivo.
        
        Args:
            review_text: Texto da revisão
            output_dir: Diretório de saída
            
        Raises:
            ValueError: Se o texto estiver vazio
            OSError: Se houver erro ao salvar
        """
        try:
            if not review_text or not review_text.strip():
                raise ValueError("Empty review text")
                
            # Criar diretório se não existir
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Gerar nome do arquivo
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = output_dir / f"literature_review_{timestamp}.md"
            
            # Salvar arquivo
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(review_text)
                
            logger.info(f"Review saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving review: {str(e)}")
            raise


# """
# Content Reviewer Agent - Literature Review Generation
# """

# import logging
# import pandas as pd
# from pathlib import Path
# from typing import Dict, List, Optional
# from datetime import datetime

# from config import settings, prompts
# from ..utils.api_client import OpenRouterClient

# logger = logging.getLogger(__name__)

# class ContentReviewer:
#     """
#     Agent responsible for generating comprehensive literature reviews from selected articles.
#     """
    
#     def __init__(self, api_key: str = settings.OPENROUTER_API_KEY):
#         """
#         Initialize the content reviewer.
        
#         Args:
#             api_key: OpenRouter API key
#         """
#         self.client = OpenRouterClient(api_key)
        
#     def generate_review(self, 
#                        input_file: Path,
#                        research_topic: str,
#                        output_lang: str,
#                        output_dir: Optional[Path] = None) -> str:
#         """
#         Generate a comprehensive literature review from processed articles.
        
#         Args:
#             input_file: Path to the processed articles CSV file
#             research_topic: Original research topic
#             output_lang: Target language for the review
#             output_dir: Optional output directory for saving the review
            
#         Returns:
#             Generated literature review in Markdown format
#         """
#         try:
#             # Read processed articles
#             articles_df = pd.read_csv(input_file)
            
#             # Prepare articles summary text
#             articles_text = self._prepare_articles_text(articles_df)
            
#             # Generate review using LLM
#             review_text = self._generate_review_text(
#                 articles_text=articles_text,
#                 research_topic=research_topic,
#                 output_lang=output_lang  # Pass output language
#             )
            
#             # Save review if output directory is provided
#             if output_dir:
#                 self._save_review(review_text, output_dir)
                
#             return review_text
            
#         except Exception as e:
#             logger.error(f"Error generating literature review: {str(e)}")
#             raise
            
#     def _prepare_articles_text(self, articles_df: pd.DataFrame) -> str:
#         """
#         Prepare articles text for review generation.
        
#         Args:
#             articles_df: DataFrame with processed articles
            
#         Returns:
#             Formatted text combining all articles information
#         """
#         articles_text = ""
        
#         for idx, article in articles_df.iterrows():
#             # Format article information in Markdown
#             article_text = f"""
# ### Article {idx + 1}

# **Title**: {article['title']}
# **Authors**: {article['authors']}
# **Summary**: {article['summary']}
# **Key Aspects**: {article.get('key_aspects', '')}
# **Selection Rationale**: {article.get('selection_rationale', '')}

# ---
# """
#             articles_text += article_text
            
#         return articles_text
        
#     def _generate_review_text(self, articles_text: str, research_topic: str, output_lang: str) -> str:
#         """
#         Generate literature review using LLM.
        
#         Args:
#             articles_text: Formatted articles text
#             research_topic: Research topic
#             output_lang: Target language for the review
            
#         Returns:
#             Generated literature review
#         """
#         # Prepare prompt for review generation
#         prompt = prompts.format_review_prompt(
#             articles_text=articles_text,
#             research_topic=research_topic,
#             target_language=output_lang  # Include target language
#         )
        
#         # Get review from LLM
#         review_text = self.client.generate_text(
#             prompt=prompt,
#             system_message=prompts.SYSTEM_MESSAGES['content_reviewer'],
#             temperature=0.7  # Slightly higher for more creative writing
#         )
        
#         return review_text
        
#     def _save_review(self, review_text: str, output_dir: Path):
#         """
#         Save generated review to file.
        
#         Args:
#             review_text: Generated review text
#             output_dir: Output directory
#         """
#         try:
#             timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#             output_file = output_dir / f"literature_review_{timestamp}.md"
            
#             with open(output_file, 'w', encoding='utf-8') as f:
#                 f.write(review_text)
                
#             logger.info(f"Review saved to {output_file}")
            
#         except Exception as e:
#             logger.error(f"Error saving review: {str(e)}")
#             raise