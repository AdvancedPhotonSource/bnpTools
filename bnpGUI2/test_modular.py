"""
Test script for the modular bnpGUI2 structure
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
current_dir = Path(__file__).parent.absolute()
parent_dir = current_dir.parent.parent
sys.path.insert(0, str(parent_dir))

def test_imports():
    """Test that all modules can be imported"""
    try:
        from bnpTools.bnpGUI2.utils import coordinate_transform, checkEntryDigit, limit_stringvar_length
        print("✓ Utils imported successfully")
        
        from bnpTools.bnpGUI2.data_handlers import XRFDataHandler, PtychoDataHandler
        print("✓ Data handlers imported successfully")
        
        from bnpTools.bnpGUI2.gui_components import (
            FileSelectionWidget, ParameterInputWidget, 
            ImageCanvasWidget, ScanControlsWidget, SetupFrame
        )
        print("✓ GUI components imported successfully")
        
        from bnpTools.bnpGUI2.config import SettingsManager
        print("✓ Config imported successfully")
        
        from bnpTools.bnpGUI2 import SetupFrame as MainSetupFrame
        print("✓ Main package imported successfully")
        
        print("\n🎉 All imports successful! The modular structure is working correctly.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_functionality():
    """Test basic functionality of components"""
    try:
        # Test settings manager
        from bnpTools.bnpGUI2.config import SettingsManager
        settings = SettingsManager()
        settings.set("test_key", "test_value")
        assert settings.get("test_key") == "test_value"
        print("✓ Settings manager working")
        
        # Test validation utilities
        from bnpTools.bnpGUI2.utils import checkEntryDigit
        assert checkEntryDigit("123.45") == True
        assert checkEntryDigit("abc") == False
        assert checkEntryDigit("") == True
        print("✓ Validation utilities working")
        
        # Test coordinate utilities
        from bnpTools.bnpGUI2.utils import coordinate_transform
        result = coordinate_transform(0.0, 1.0, 2.0, 3.0)
        assert isinstance(result, dict)
        assert "x" in result and "y" in result and "z" in result
        print("✓ Coordinate utilities working")
        
        print("\n🎉 All functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Functionality test error: {e}")
        return False

if __name__ == "__main__":
    print("Testing BNP GUI v2 Modular Structure")
    print("=" * 40)
    
    print("\n1. Testing imports...")
    import_success = test_imports()
    
    print("\n2. Testing functionality...")
    func_success = test_functionality()
    
    print("\n" + "=" * 40)
    if import_success and func_success:
        print("🎉 All tests passed! The modular structure is ready to use.")
        print("\nTo run the GUI:")
        print("  python bnpTools/bnpGUI2/main.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
