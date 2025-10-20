"""
String utilities for GUI components
"""

import tkinter as tk
from typing import Callable, Optional


class StringVarWithLength(tk.StringVar):
    """
    StringVar with length limitation and callback support
    """
    
    def __init__(self, max_length: Optional[int] = None, callback: Optional[Callable] = None):
        super().__init__()
        self.max_length = max_length
        self.callback = callback
        
        if callback:
            self.trace('w', lambda *args: callback())
    
    def set(self, value: str) -> None:
        """Set value with length limitation"""
        if self.max_length and len(value) > self.max_length:
            value = value[:self.max_length]
        super().set(value)