"""
BNP GUI v2 - Modular Refactored Version
=======================================

A modular refactoring of the original bnpGUI setupFrame.py
Maintains exact GUI layout while improving code organization.

Author: BNP Development Team
Version: 2.0.0
"""

__version__ = "2.0.0"
__author__ = "BNP Development Team"

from .gui_components.setup_frame import SetupFrame

__all__ = ['SetupFrame']
