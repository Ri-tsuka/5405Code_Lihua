# Repaint: Interactive Image Repainting and Decoration Demo

## Project Overview

Repaint is an interactive image editing application based on PyQt5, specifically designed for interior design and wall decoration scenarios. The program allows users to implement the following three functions through simple click operations:

1. **Painting Replacement**: Perspective mapping of a specified image (such as artwork) onto a selected area on the wall
2. **Wall Recoloring**: Select wall sections and apply new colors while preserving original lighting effects and texture details
3. **Wallpaper Replacement**: Apply wallpaper textures to selected areas on walls

## Key Features

- **Intuitive User Interface**: Clean graphical interface with mouse click interaction
- **Custom Image Processing Algorithms**: Manually implemented Gaussian blur and perspective transformation algorithms
- **Intelligent Color Adjustment**: Preserves original lighting structure when changing colors
- **Smooth Blending Effects**: All transformations use gradient edges to ensure natural results
- **RGB Color Presets**: Built-in wall color preset options

## Technical Highlights

1. **Manually Implemented Gaussian Blur Algorithm**: Does not rely on OpenCV's advanced functions, with custom-written convolution operations
2. **Manually Implemented Perspective Transformation**: Uses inverse mapping and bilinear interpolation for professional-grade perspective effects
3. **Grayscale Structure-Preserving Coloring**: Preserves the light-dark structure of the original image during recoloring
4. **Region and Flood Fill Combination**: Utilizes a combination of polygonal regions and seed filling for precise selection

## Installation Dependencies

The program requires the following Python libraries:

```bash
pip install numpy opencv-python PyQt5
```

## Usage Instructions

1. Prepare three image files and place them in the same directory as the program:
   - `test2.jpg`: Original interior scene image
   - painting.jpg: Artwork image for replacement
   - wallpaper.jpg: Wallpaper texture image for application

2. Run the program:
   ```bash
   python repaint_tem.py
   ```

3. Select operation mode:
   - **Replace Painting**: Replace artwork
   - **Recolour Wall**: Recolor wall surface
   - **Change Wallpaper**: Change wallpaper

4. Perform corresponding operations based on the selected mode:
   - **Painting Replacement Mode**: Click the 4 corner points of the target area in clockwise order, then click the "APPLY" button
   - **Wall Coloring Mode**: Click a point on the wall as a seed point, select a color preset, then click the "APPLY" button
   - **Wallpaper Replacement Mode**: First click 4 corner points to define the area, then click a point inside the area as a seed point, then click the "APPLY" button

5. Use the "RESET" button to restore the original image

## Working Principles

### Painting Replacement

1. User clicks to define the 4 corner points of the target quadrilateral
2. System calculates the perspective transformation matrix
3. Applies perspective transformation to map the artwork to the target area
4. Creates a polygon mask and replaces the corresponding area in the original image with the transformed artwork

### Wall Recoloring

1. User clicks a point on the wall as a seed point
2. System uses FloodFill algorithm to identify areas of similar color
3. Applies Gaussian blur to the mask to create smooth transitions
4. Converts the original image to grayscale and preserves light-dark structure
5. Multiplies the selected new color with the grayscale structure to achieve texture-preserving coloring
6. Blends the new color area with the original image

### Wallpaper Application

1. User clicks to define the 4 corner points of the target area
2. User clicks a point inside the area as a seed point
3. System combines polygonal region and seed point filling to precisely identify the target area
4. Applies Gaussian blur to the mask to create smooth transitions
5. Performs perspective transformation on the wallpaper and blends it with the original image

## Custom Implemented Algorithms

### 1. Gaussian Blur

The program manually implements the Gaussian blur algorithm, including:
- Generation of two-dimensional Gaussian kernel function
- Application of the kernel to the image using two-dimensional convolution operations
- Edge handling using reflection padding

### 2. Perspective Transformation

The program manually implements the perspective transformation algorithm, including:
- Calculation of inverse transformation matrix
- Use of inverse mapping method
- Implementation of bilinear interpolation to ensure smooth pixel values

### 3. Structure-Preserving Coloring Technique

When recoloring, the program preserves the structural information of the original image:
- Converts RGB image to grayscale
- Normalizes grayscale values as brightness structure
- Multiplies new colors by this structure, preserving light-dark variations

## Notes

- Ensure the three preset image files are in the same directory as the program
- Best results require clear boundaries and appropriate lighting conditions
- Wall recoloring effect depends on the seed point selection location
- Wallpaper application requires selection of clear area boundaries

## Possible Extensions

- Add more color selection options
- Support loading custom images and textures
- Implement undo/redo functionality
- Add more image processing filters
- Support saving processed images

## Author

This demonstration program showcases core concepts of computer vision and image processing through implementation of multiple manual image processing algorithms. The program is suitable as a teaching tool, introducing the implementation principles of perspective transformation, color adjustment, and region selection techniques.