"""
Prompt templates for Large Language Models (LLMs).
All prompts are in English for better model performance.
"""

from typing import Dict, Optional, List, Any
from dataclasses import dataclass

@dataclass
class ArticleAnalysis:
    """Data class for article analysis results."""
    arxiv_id: str
    query_id: str
    relevance_score: float
    selection_rationale: str
    key_aspects: List[str]
    methodology_score: float
    impact_score: float
    language_notes: str

@dataclass
class PromptTemplates:
    """Collection of prompt templates for different tasks."""
    
    # Translation prompt
    TRANSLATION = """
    You are an expert in scientific translation. Analyze and translate the following research topic into English, preserving technical terminology and context.

    RESEARCH TOPIC: {research_topic}

    Provide your response in the following JSON format only:
    {{
        "translated_topic": "your translation here",
        "key_terms": ["list", "of", "key", "technical", "terms"],
        "explanation": "brief explanation of translation choices (optional)"
    }}

    Guidelines:
    1. Keep technical terms, acronyms (SQL, API, LLM), and proper names unchanged
    2. Preserve mathematical notation and equations
    3. Maintain domain-specific terminology
    4. Explain any ambiguous or context-dependent translations
    """

    # Query generation prompt
    QUERY_GENERATION = """
    You are an expert academic researcher. Generate precise search queries for research on the following topic:

    ORIGINAL TOPIC: {original_topic}
    ENGLISH TOPIC: {english_topic}

    QUERY FORMAT:
    {{
        "queries": [
            {{
                "query_id": "q1",
                "query_text": "your search query",
                "rationale": "explanation of query construction",
                "aspect": "specific aspect or subtopic targeted"
            }}
        ]
    }}

    QUERY CONSTRUCTION RULES:

    1. FIELD PREFIXES (in query_text):
    - ti: title search 
    - abs: abstract search 
    - au: author search 
    - all: all fields search 
    - cat: subject category 

    2. OPERATORS & SYNTAX:
    - Don't use numbers or bullet points in query_text
    - Use proper boolean operators: AND, OR, ANDNOT
    - Use quotes ("") only for multi-word technical terms
    - Break down complex terms into component parts with OR
    - Use parentheses for complex boolean logic
    - Avoid over-restrictive exact phrase matching
    - Consider alternative names and abbreviations

    EXAMPLE QUERIES:
    {{
        "queries": [
            {{
                "query_id": "q1",
                "query_text": "ti:\"deep learning\" AND abs:\"computer vision\"",
                "rationale": "Finds papers with 'deep learning' in title and 'computer vision' in abstract",
                "aspect": "deep learning applications in computer vision"
            }},
            {{
                "query_id": "q2",
                "query_text": "(ti:neural OR ti:\"deep learning\") AND abs:optimization",
                "rationale": "Searches for neural networks or deep learning papers focusing on optimization",
                "aspect": "optimization methods in neural networks"
            }}
        ]
    }}

    QUERY GUIDELINES:
    - Include relevant technical terms and synonyms
    - Consider different methodological approaches
    - Target both broad and specific aspects
    - Focus on recent developments
    - Combine multiple fields when appropriate
    - Include subject categories when relevant
    - Consider cross-disciplinary aspects

    BASE URL (for testing):
    http://export.arxiv.org/api/query?search_query=YOUR_QUERY
    """

    # Content analysis prompt
    CONTENT_ANALYSIS = """
    You are an expert in academic content analysis. Evaluate this article's relevance to the research topic.

    ORIGINAL TOPIC: {original_topic}
    ENGLISH TOPIC: {english_topic}
    QUERY USED: {query_text}

    ARTICLE:
    Title: {title}
    Abstract: {summary}
    Categories: {categories}

    EVALUATION CRITERIA:
    1. Topic Relevance (40%)
       - Direct alignment with research topic
       - Coverage of key concepts
       - Depth of relevant discussion

    2. Methodology (30%)
       - Research design quality
       - Data collection and analysis
       - Reproducibility and rigor

    3. Impact Potential (30%)
       - Citation metrics
       - Practical applications
       - Innovation level
       - Future research potential

    RESPONSE FORMAT:
    Return a JSON object with the following structure EXACTLY as shown:
    {{
        "relevance_score": 0.85,
        "methodology_score": 0.75,
        "impact_score": 0.8,
        "key_aspects": ["aspect1", "aspect2", "aspect3"],
        "selection_rationale": "Detailed explanation of why this article is relevant",
        "language_notes": "Notes about terminology and translation considerations"
    }}

    IMPORTANT:
    - All scores must be decimal numbers between 0 and 1
    - key_aspects must be a list of strings
    - Do not include any text before or after the JSON object
    - Ensure the JSON is properly formatted with double quotes
    """

    # Literature review prompt
    REVIEW_GENERATION = """
    You are an expert academic researcher writing a comprehensive literature review.

    RESEARCH TOPIC: {research_topic}
    TARGET LANGUAGE: {target_language}
    
    ARTICLES TO REVIEW:
    {articles_text}

    PROCESS:
    1. Analysis Phase (English):
       - Extract key findings and methodologies
       - Identify patterns and themes
       - Evaluate research quality
       - Map knowledge gaps

    2. Synthesis Phase (English):
       - Connect related findings
       - Compare methodologies
       - Highlight contradictions
       - Identify research trends

    3. Translation Phase ({target_language}):
       - Translate headers and subheaders
       - Maintain academic rigor
       - Preserve technical terminology
       - Ensure cultural appropriateness
       - Keep citations consistent

    REQUIRED SECTIONS:
    ## Introduction
    - Research context
    - Scope and objectives
    - Review methodology

    ## Theoretical Framework
    - Key concepts
    - Current debates
    - Theoretical evolution

    ## Methodology Analysis
    - Research approaches
    - Data collection methods
    - Analytical frameworks

    ## Results Synthesis
    - Major findings
    - Emerging patterns
    - Contradictions
    - Knowledge gaps

    ## Future Directions
    - Research opportunities
    - Methodological gaps
    - Practical implications

    ## Conclusion
    - Key insights
    - Limitations
    - Recommendations

    FORMAT REQUIREMENTS:
    1. Use Markdown formatting
    2. LaTeX for equations ($$...$$)
    3. [Author] citation format
    4. Headers with ## prefix
    5. Bullet points for key lists
    6. Tables for comparisons
    """

@dataclass
class SystemMessages:
    """System messages for controlling LLM behavior."""
    
    TRANSLATOR = """You are a scientific translation expert specializing in academic 
    and technical translations. You must preserve model names, version numbers, and technical terms EXACTLY as written.
    Never attempt to 'correct' or change technical names, even if they appear to be misspelled or similar to other terms.
    Focus on accuracy, terminology preservation, and cultural appropriateness."""

    QUERY_GENERATOR = """You are an academic research expert with extensive 
    knowledge across multiple fields. Create comprehensive and precise search 
    queries that capture all relevant aspects of the research topic."""

    CONTENT_ANALYZER = """You are an expert academic content analyst specialized 
    in quantitative research evaluation. Focus on methodological rigor, impact 
    assessment, and technical quality evaluation."""

    CONTENT_REVIEWER = """You are an expert academic researcher and translator 
    specializing in literature reviews. Combine analytical skills with precise 
    translation capabilities to create comprehensive, well-structured reviews."""

@dataclass
class ErrorMessages:
    """Error messages in multiple languages."""
    
    INVALID_QUERY = {
        'en': "LLM response is not in the expected JSON format for queries",
        'pt': "A resposta do LLM não está no formato JSON esperado para queries"
    }
    
    INVALID_ANALYSIS = {
        'en': "LLM response is not in the expected JSON format for analysis",
        'pt': "A resposta do LLM não está no formato JSON esperado para análise"
    }
    
    EMPTY_SUMMARY = {
        'en': "Article has no abstract available for analysis",
        'pt': "O artigo não possui resumo disponível para análise"
    }
    
    API_ERROR = {
        'en': "Error in LLM API communication",
        'pt': "Erro na comunicação com a API do LLM"
    }
    
    TRANSLATION_ERROR = {
        'en': "Error translating research topic",
        'pt': "Erro na tradução do tema de pesquisa"
    }

def validate_prompt_args(**kwargs) -> None:
    """
    Validates prompt formatting arguments.
    
    Raises:
        ValueError: If required arguments are missing or invalid
    """
    required_fields = set(kwargs.keys())
    for field, value in kwargs.items():
        if value is None or (isinstance(value, str) and not value.strip()):
            raise ValueError(f"Missing or empty required field: {field}")

def format_translation_prompt(research_topic: str) -> str:
    """
    Formats the translation prompt.
    
    Args:
        research_topic: Original research topic
        
    Returns:
        Formatted prompt string
    
    Raises:
        ValueError: If research_topic is empty or invalid
    """
    validate_prompt_args(research_topic=research_topic)
    return PromptTemplates.TRANSLATION.format(research_topic=research_topic)

def format_query_prompt(original_topic: str, english_topic: str) -> str:
    """
    Formats the query generation prompt.
    
    Args:
        original_topic: Original research topic
        english_topic: English translation of the topic
        
    Returns:
        Formatted prompt string
    """
    validate_prompt_args(original_topic=original_topic, english_topic=english_topic)
    return PromptTemplates.QUERY_GENERATION.format(
        original_topic=original_topic,
        english_topic=english_topic
    )

def format_analysis_prompt(
    article_data: Dict[str, Any],
    original_topic: str,
    english_topic: str,
    query_text: str
) -> str:
    """
    Formats the content analysis prompt.
    
    Args:
        article_data: Dictionary with article information
        original_topic: Original research topic
        english_topic: English translation of the topic
        query_text: Query used to find the article
        
    Returns:
        Formatted prompt string
    """
    validate_prompt_args(
        original_topic=original_topic,
        english_topic=english_topic,
        query_text=query_text
    )
    
    return PromptTemplates.CONTENT_ANALYSIS.format(
        original_topic=original_topic,
        english_topic=english_topic,
        query_text=query_text,
        title=article_data.get('title', ''),
        summary=article_data.get('summary', ''),
        categories=', '.join(article_data.get('categories', []))
    )

def format_review_prompt(
    articles_text: str,
    research_topic: str,
    target_language: str
) -> str:
    """
    Formats the literature review generation prompt.
    
    Args:
        articles_text: Text containing article information
        research_topic: Research topic
        target_language: Target language for the review
        
    Returns:
        Formatted prompt string
    """
    validate_prompt_args(
        articles_text=articles_text,
        research_topic=research_topic,
        target_language=target_language
    )
    
    return PromptTemplates.REVIEW_GENERATION.format(
        articles_text=articles_text,
        research_topic=research_topic,
        target_language=target_language
    )

# Expose classes and constants
__all__ = [
    'PromptTemplates',
    'SystemMessages',
    'ErrorMessages',
    'ArticleAnalysis',
    'format_translation_prompt',
    'format_query_prompt',
    'format_analysis_prompt',
    'format_review_prompt'
]
