"""
Image canvas GUI component
Extracted from original setupFrame.py
"""

import tkinter as tk
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib import colors, patches
from typing import Callable, Optional, Tuple, Dict, Any


class ImageCanvasWidget:
    """Widget for displaying 2D images with interaction"""
    
    def __init__(self, parent_frame):
        self.parent_frame = parent_frame
        
        # Canvas state
        self.rectangle_drawing = False
        self.dot_drawing = False
        self.non_animated_background = None
        self.show_roi_rectangle = None
        self.rectangle = None
        
        # Data
        self.image2d = None
        self.x = None
        self.y = None
        self.file_z = None
        self.file_theta = None
        
        # Callbacks
        self.xy_center_callback = None
        self.scan_box_callback = None
        self.coordinate_callback = None
        
        self._create_widgets()
    
    def _create_widgets(self):
        """Create image canvas widgets"""
        # Create matplotlib figure and canvas
        self.figure2d = Figure()
        self.axe2d = self.figure2d.add_axes([0, 0, 1, 1])
        self.axe2d.set_axis_off()
        
        self.canvas2d = FigureCanvasTkAgg(self.figure2d, master=self.parent_frame)
        self.canvas2d.draw()
        self.canvas2d.get_tk_widget().config(width=650, height=650, cursor="cross")
        self.canvas2d.get_tk_widget().grid(
            column=0, columnspan=10, row=7, rowspan=50,
            padx=(10, 20), pady=(5, 5), sticky="W"
        )
        
        # Canvas event bindings
        self.canvas2d_mouse_hover_event = self.canvas2d.mpl_connect(
            "motion_notify_event", self._canvas2d_mouse_hover
        )
        self.canvas2d_button_press_event = self.canvas2d.mpl_connect(
            "button_press_event", self._canvas2d_button_pressed
        )
        
        # Canvas controls
        self._create_canvas_controls()
    
    def _create_canvas_controls(self):
        """Create canvas control widgets"""
        # Log scale button
        self.log_button = tk.Button(
            self.parent_frame, text="Log", command=self.logscale_changed
        )
        self.log_button.grid(column=0, row=58, padx=(5, 5), pady=(0, 0))
        
        # Coordinate display
        self.xycorr = tk.StringVar()
        self.xycorr.set("x, y: (0.00, 0.00)")
        xycorr_label = tk.Label(self.parent_frame, textvariable=self.xycorr)
        xycorr_label.grid(row=58, column=1)
    
    def display2data(self, axe, x, y):
        """Convert display coordinates to data coordinates"""
        return axe.transData.inverted().transform(np.array([(x, y)]))[0]
    
    def draw_rectangle(self, xmin, xmax, ymin, ymax, color, lw=2, animated=False):
        """Draw rectangle on canvas"""
        rectangle = patches.Rectangle(
            (xmin, ymin),
            width=xmax - xmin,
            height=ymax - ymin,
            alpha=1,
            edgecolor="w",
            fill=False,
            linewidth=lw,
            animated=animated,
        )
        self.axe2d.add_patch(rectangle)
        return rectangle
    
    def display_image(self, data, x=None, y=None, cmap="gray", log_scale=False):
        """Display image data on canvas"""
        if x is not None:
            self.x = x
        if y is not None:
            self.y = y
        
        # Set normalization
        if log_scale:
            norm = colors.SymLogNorm(linthresh=0.5)
        else:
            norm = colors.Normalize()
        
        # Display image
        self.image2d = self.axe2d.imshow(
            data,
            aspect="equal",
            interpolation="nearest",
            cmap=cmap,
            origin="lower",
            norm=norm
        )
        
        self.canvas2d.draw()
    
    def update_image_data(self, data, cmap=None, log_scale=None):
        """Update existing image data"""
        if self.image2d is None:
            return
        
        # Check if data shape matches
        img_data = self.image2d.get_array()
        if img_data.shape != data.shape:
            # Need to replot
            self.display_image(data, cmap=cmap or "inferno", log_scale=log_scale)
            return
        
        # Update existing image
        self.image2d.set_array(data)
        if cmap:
            self.image2d.set_cmap(cmap)
        
        if log_scale is not None:
            if log_scale:
                self.image2d.set_norm(colors.SymLogNorm(linthresh=0.5))
            else:
                self.image2d.set_norm(colors.Normalize())
        
        self.canvas2d.draw()
    
    def logscale_changed(self):
        """Handle log scale toggle"""
        if self.image2d is None:
            return
            
        if self.log_button.config("relief")[-1] == "sunken":
            self.log_button.config(relief="raised")
            self.image2d.set_norm(colors.Normalize())
        else:
            self.log_button.config(relief="sunken")
            self.image2d.set_norm(colors.SymLogNorm(linthresh=0.5))
        self.canvas2d.draw()
    
    def _canvas2d_button_pressed(self, event):
        """Handle canvas button press"""
        if not event.inaxes:
            return
            
        self.xstart, self.ystart = list(
            map(
                lambda x: int(round(x, 0)),
                (self.display2data(self.axe2d, event.x, event.y)),
            )
        )
        
        if event.button == 1:
            # Left click - show coordinates
            if self.x is not None and self.y is not None:
                self.xycorr.set(
                    "x, y: (%.2f, %.2f)" % (self.x[self.xstart], self.y[self.ystart])
                )
            
            # Handle XY Center mode
            if hasattr(self, 'insert_type') and self.insert_type.get() == "XY Center":
                try:
                    self.show_roi_rectangle.remove()
                except (AttributeError, ValueError):
                    pass
                if self.xy_center_callback:
                    self.xy_center_callback(self.xstart, self.ystart)
        
        elif event.button == 3:
            # Right click - handle ScanBox mode
            if hasattr(self, 'insert_type') and self.insert_type.get() == "ScanBox":
                self.rectangle_drawing = True
                try:
                    self.show_roi_rectangle.remove()
                except (AttributeError, ValueError):
                    pass
                self.canvas2d.mpl_disconnect(self.canvas2d_button_press_event)
                self.canvas2d_button_release_event = self.canvas2d.mpl_connect(
                    "button_release_event", self._canvas2d_button_released
                )
    
    def _canvas2d_button_released(self, event):
        """Handle canvas button release"""
        if not event.inaxes:
            return
            
        self.xend, self.yend = list(
            map(
                lambda x: int(round(x, 0)),
                (self.display2data(self.axe2d, event.x, event.y)),
            )
        )

        self.canvas2d.mpl_disconnect(self.canvas2d_button_release_event)
        self.canvas2d_button_press_event = self.canvas2d.mpl_connect(
            "button_press_event", self._canvas2d_button_pressed
        )

        if self.rectangle_drawing:
            self.rectangle_drawing = False
            self.non_animated_background = None
            try:
                self.show_roi_rectangle.remove()
            except (AttributeError, ValueError):
                pass
            xmin = int(min(self.xend, self.xstart))
            xmax = int(max(self.xend, self.xstart))
            ymin = int(min(self.yend, self.ystart))
            ymax = int(max(self.yend, self.ystart))
            self.show_roi_rectangle = self.draw_rectangle(xmin, xmax, ymin, ymax, "w")
            
            if self.scan_box_callback:
                self.scan_box_callback(xmin, xmax, ymin, ymax)
    
    def _canvas2d_mouse_hover(self, event):
        """Handle mouse hover on canvas"""
        if not event.inaxes or self.image2d is None:
            return
            
        self.xend, self.yend = list(
            map(
                lambda x: int(round(x, 0)),
                (self.display2data(self.axe2d, event.x, event.y)),
            )
        )
        
        try:
            dim_y, dim_x = self.image2d.get_array().shape
        except:
            return
            
        if (
            (self.yend < dim_y - 0.5)
            and (self.yend > -0.5)
            and (self.xend > -0.5)
            and (self.xend < dim_x - 0.5)
        ):
            if self.rectangle_drawing:
                if self.non_animated_background is not None:
                    # Restore the clean slate background
                    self.canvas2d.restore_region(self.non_animated_background)
                    if self.xstart > self.xend:  # Modify the starting point
                        self.rectangle.set_x(self.xend)
                    self.rectangle.set_width(abs(self.xend - self.xstart))
                    if self.ystart > self.yend:  # Modify the starting point
                        self.rectangle.set_y(self.yend)
                    self.rectangle.set_height(abs(self.yend - self.ystart))
                    self.axe2d.draw_artist(self.rectangle)
                    self.canvas2d.blit(self.axe2d.bbox)
                else:
                    xmax = int(max(self.xend, self.xstart))
                    xmin = int(min(self.xend, self.xstart))
                    ymax = int(max(self.yend, self.ystart))
                    ymin = int(min(self.yend, self.ystart))
                    self.rectangle = self.draw_rectangle(
                        xmin, xmax, ymin, ymax, "w", animated=True
                    )
                    self.canvas2d.draw()
                    self.non_animated_background = self.canvas2d.copy_from_bbox(
                        self.axe2d.bbox
                    )
    
    def set_xy_center_callback(self, callback: Callable):
        """Set callback for XY center selection"""
        self.xy_center_callback = callback
    
    def set_scan_box_callback(self, callback: Callable):
        """Set callback for scan box selection"""
        self.scan_box_callback = callback
    
    def set_coordinate_callback(self, callback: Callable):
        """Set callback for coordinate display"""
        self.coordinate_callback = callback
