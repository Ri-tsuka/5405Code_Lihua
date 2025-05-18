# Repaint: Interactive Image Repainting and Decoration Demo

## Installation Dependencies

The program requires the following Python libraries:

```bash
pip install numpy opencv-python PyQt5
```

## Usage Instructions

1. Prepare three image files and place them in the same directory as the program:
   - `test2.jpg`: Original interior scene image
   - painting.jpg: Artwork image for replacement
   - wallpaper.jpg: Wallpaper texture image for replacement

2. Run the program:
   ```bash
   python repaint_tem.py
   ```

3. Select operation mode:
   - **Replace Painting**: Replace artwork
   - **Recolour Wall**: Recolor wall surface
   - **Change Wallpaper**: Change wallpaper

4. Perform corresponding operations based on the selected mode:
   - **Painting Replacement Mode**: Click on 'Replace Painting', click the 4 corner points of the target area in clockwise order, then click the "APPLY" button
   - **Wall Coloring Mode**: Click on 'Recolour Wall', click a point on the wall as a seed point, select a color preset, then click the "APPLY" button
   - **Wallpaper Replacement Mode**: Click on 'Change Wallpaper', click 4 corner points to define the area, then click a point inside the area as a seed point, then click the "APPLY" button

5. Use the "RESET" button to restore the original image

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
