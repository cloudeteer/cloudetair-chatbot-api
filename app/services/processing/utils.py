"""Backward-compatible text cleaning utilities."""

import html
import re
from typing import Optional


def clean_text(html_or_markup: str) -> str:
    """
    Lightweight text cleaning fallback for when Docling is not available.
    
    This is a fallback function for basic text cleaning when a full
    DoclingDocument is not available. The primary path should use Docling
    preprocessing for better structure preservation.
    
    Args:
        html_or_markup: HTML or markup text to clean
        
    Returns:
        str: Cleaned plain text
    """
    if not html_or_markup:
        return ""
    
    text = html_or_markup
    
    # Decode HTML entities
    text = html.unescape(text)
    
    # Remove HTML tags but preserve some structure
    text = re.sub(r'<br[^>]*>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</?(p|div|h[1-6])[^>]*>', '\n\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</li>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', '', text)
    
    # Clean up whitespace
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Multiple newlines to double
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces to single
    text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE)  # Trim lines
    
    return text.strip()