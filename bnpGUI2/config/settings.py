"""
Settings management for BNP GUI v2
"""

from typing import Dict, Any, Optional
import json
import os
from pathlib import Path


class SettingsManager:
    """Manages application settings and configuration"""
    
    def __init__(self, config_file: Optional[str] = None):
        if config_file is None:
            config_file = os.path.join(os.path.expanduser("~"), ".bnpgui2", "config.json")
        
        self.config_file = Path(config_file)
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        self.settings = self._load_settings()
    
    def _load_settings(self) -> Dict[str, Any]:
        """Load settings from file"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        
        # Return default settings
        return self._get_default_settings()
    
    def _get_default_settings(self) -> Dict[str, Any]:
        """Get default settings"""
        return {
            "default_xrf_folder": "/mnt/micdata1/2idd",
            "default_ptycho_folder": "/mnt/micdata1/bnp/2021-3/",
            "canvas_width": 650,
            "canvas_height": 650,
            "log_scale_linthresh": 0.5,
            "max_file_message_length": 100,
            "scan_time_efficiency": 0.8,
            "max_scan_width": 80,
            "default_sample_name": "",
            "default_dwell_time": 1.0,
            "default_width": 20.0,
            "default_height": 20.0,
            "default_w_step": 1.0,
            "default_h_step": 1.0,
        }
    
    def save_settings(self):
        """Save settings to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.settings, f, indent=2)
        except IOError as e:
            print(f"Error saving settings: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get setting value"""
        return self.settings.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set setting value"""
        self.settings[key] = value
    
    def update(self, settings: Dict[str, Any]):
        """Update multiple settings"""
        self.settings.update(settings)