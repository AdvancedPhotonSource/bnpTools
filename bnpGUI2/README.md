# BNP GUI v2 - Modular Refactored Version

A modular refactoring of the original `bnpGUI/setupFrame.py` that maintains the exact same GUI layout while improving code organization and maintainability.

## Structure

```
bnpGUI2/
├── __init__.py                 # Main package initialization
├── main.py                     # Application entry point
├── test_modular.py            # Test script for modular structure
├── README.md                  # This file
├── config/                    # Configuration management
│   ├── __init__.py
│   └── settings.py           # Settings manager
├── data_handlers/             # Data loading and processing
│   ├── __init__.py
│   ├── xrf_handler.py        # XRF data handling
│   └── ptycho_handler.py     # Ptycho data handling
├── gui_components/            # Modular GUI widgets
│   ├── __init__.py
│   ├── file_selection.py     # File/folder selection widgets
│   ├── parameter_inputs.py   # Scan parameter input widgets
│   ├── image_canvas.py       # Matplotlib canvas widget
│   ├── scan_controls.py      # Scan control buttons
│   └── setup_frame.py        # Main orchestrating frame
└── utils/                     # Utility functions
    ├── __init__.py
    ├── coordinate_utils.py   # Coordinate transformations
    ├── validation_utils.py   # Input validation
    └── string_utils.py       # String handling utilities
```

## Key Features

### 🔧 **Modular Design**
- **Separation of Concerns**: Each component has a single responsibility
- **Reusable Components**: GUI widgets can be used independently
- **Clean Interfaces**: Well-defined APIs between components

### 🎨 **Preserved Layout**
- **Exact GUI Layout**: Maintains the original `setupFrame.py` layout pixel-perfect
- **Same Functionality**: All original features preserved
- **Compatible API**: Drop-in replacement for original setupFrame

### 📊 **Data Handling**
- **XRF Data**: Modular XRF data loading and processing
- **Ptycho Data**: Separate Ptycho data handling with reconstruction support
- **Error Handling**: Robust error handling and user feedback

### 🖼️ **Image Display**
- **Matplotlib Integration**: Interactive 2D image display
- **Canvas Interactions**: Click-to-select coordinates and scan boxes
- **Log Scale**: Toggle between linear and log scale display

### ⚙️ **Configuration**
- **Settings Management**: Persistent configuration storage
- **Default Values**: Sensible defaults for all parameters
- **Customizable**: Easy to modify settings and behavior

## Usage

### Running the Application

```bash
# From the guidev directory
python bnpTools/bnpGUI2/main.py
```

### Testing the Modular Structure

```bash
# Run the test script
python bnpTools/bnpGUI2/test_modular.py
```

### Using as a Module

```python
from bnpTools.bnpGUI2 import SetupFrame
from pvComm import pvComm

# Create setup frame
pv_comm = pvComm()
setup_frame = SetupFrame(tab_control, pv_comm)
```

## Component Details

### File Selection Widget
- **XRF Folder Selection**: Choose XRF data folders
- **Ptycho Folder Selection**: Choose Ptycho reconstruction folders
- **File Browsing**: Automatic file listing and selection
- **Detector Selection**: Choose XRF detector elements

### Parameter Input Widget
- **Scan Type Selection**: XRF, Coarse-Fine, Angle Sweep options
- **Insert Method**: Manual, XY Center, ScanBox selection
- **Parameter Fields**: All original scan parameters
- **Time Calculation**: Automatic scan time estimation
- **Coordinate Updates**: XYZ position updates from PV

### Image Canvas Widget
- **2D Image Display**: Matplotlib-based image visualization
- **Interactive Selection**: Click for coordinates, drag for scan boxes
- **Log Scale Toggle**: Switch between linear and log scale
- **Real-time Updates**: Dynamic image updates

### Scan Controls Widget
- **Add to Scan**: Button to add current parameters to scan queue
- **Status Display**: File loading and error status messages
- **Directory Display**: Current data save directory
- **Time Display**: Total estimated scan time

## Dependencies

- `tkinter` - GUI framework
- `matplotlib` - Image display
- `numpy` - Numerical operations
- `h5py` - HDF5 file handling
- `tifffile` - TIFF file handling
- `pvComm` - EPICS PV communication
- `mic_vis.bnp.mda` - MDA file handling

## Migration from Original

The modular version is designed as a drop-in replacement for the original `setupFrame.py`:

1. **Same API**: `addToScanBtn()` method preserved
2. **Same Layout**: Identical widget positioning and appearance
3. **Same Functionality**: All original features maintained
4. **Better Organization**: Code split into logical modules

## Benefits

### For Developers
- **Easier Maintenance**: Clear separation of concerns
- **Better Testing**: Individual components can be tested
- **Code Reuse**: Components can be used in other applications
- **Documentation**: Well-documented interfaces

### For Users
- **Same Experience**: Identical GUI and functionality
- **Better Performance**: Optimized data handling
- **More Reliable**: Better error handling and validation
- **Future-Proof**: Easier to add new features

## Future Enhancements

- **Unit Tests**: Comprehensive test coverage
- **Configuration GUI**: Settings management interface
- **Plugin System**: Extensible component architecture
- **Documentation**: Detailed API documentation
- **Performance**: Further optimization opportunities