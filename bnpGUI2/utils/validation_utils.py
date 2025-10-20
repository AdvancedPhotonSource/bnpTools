"""
Input validation utilities
Extracted from original setupFrame.py
"""

import tkinter as tk
from typing import Any


def checkEntryDigit(value: str) -> bool:
    """
    Validate that entry contains only digits and decimal point
    Extracted from original misc.checkEntryDigit
    
    Args:
        value: String value to validate
        
    Returns:
        True if valid, False otherwise
    """
    if value == "":
        return True
    
    try:
        float(value)
        return True
    except ValueError:
        return False


def limit_stringvar_length(string_var: tk.StringVar, max_length: int) -> None:
    """
    Limit the length of a StringVar to prevent excessive display
    Extracted from original misc.limit_stringvar_length
    
    Args:
        string_var: Tkinter StringVar to limit
        max_length: Maximum allowed length
    """
    current_value = string_var.get()
    if len(current_value) > max_length:
        string_var.set(current_value[:max_length] + "...")
