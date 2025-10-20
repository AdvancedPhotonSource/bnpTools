"""
Main entry point for BNP GUI v2
"""

import sys
import tkinter as tk
from tkinter import ttk
from pathlib import Path

# Add parent directory to path for imports
current_dir = Path(__file__).parent.absolute()
parent_dir = current_dir.parent.parent
sys.path.insert(0, str(parent_dir))

from bnpTools.bnpGUI2.gui_components.setup_frame import SetupFrame
from pvComm import pvComm


def main():
    """Main application entry point"""
    # Create main window
    root = tk.Tk()
    root.title("BNP GUI v2 - Modular Setup Frame")
    root.geometry("1200x800")
    
    # Create tab control
    tab_control = ttk.Notebook(root)
    tab_control.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Initialize PV communication
    pv_comm = pvComm()
    
    # Create setup frame
    setup_frame = SetupFrame(tab_control, pv_comm)
    tab_control.add(setup_frame.setupfrm, text="Setup")
    
    # Start GUI
    root.mainloop()


if __name__ == "__main__":
    main()
