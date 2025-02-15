"""Basic test script for Rxivonauta."""
import asyncio
from pathlib import Path
from src.main import main

async def test_basic_search():
    """Test basic search functionality."""
    args = type('Args', (), {
        'topic': "GLM Regression Models",
        'output_dir': Path('test_output'),
        'output_lang': 'en-US',
        'model': "google/gemini-2.0-pro-exp-02-05:free",
        'max_queries': 2,
        'max_results_per_query': 2,
        'max_age_days': 365,
        'categories': ['stat.ML', 'stat.ME'],
        'min_score': 0.6,
        'max_selected': 2,
        'temperature': 0.7,
        'batch_size': 1000,
        'debug': True,
        'retry_attempts': 3,
        'retry_delay': 5,
        'timeout': 30,
        'rate_limit': 3.0
    })()
    
    result = await main(args)
    print(f"Test completed with {result.total_articles} articles found")

if __name__ == "__main__":
    asyncio.run(test_basic_search())
