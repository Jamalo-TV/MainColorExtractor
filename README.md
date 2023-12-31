# MainColorExtractor

**MainColorExtractor** is a Python program that utilizes OpenCV and scikit-learn's KMeans clustering to determine the dominant color in an image.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.x
- OpenCV (cv2) library
- scikit-learn library
- numpy library

You can install these using pip:

```
pip install opencv-python-headless
pip install scikit-learn
pip install numpy
```

## How to use MainColorExtractor

1. Clone this repository:
```
git clone <https://github.com/Jamalo-TV/MainColorExtractor>
```

2. Run the program:
```
python MainColorExtractor.py
```

By default, the program uses the image `test_image1.jpg`. You can modify the `image_path` variable in the code to process a different image.

## Functionality

The core functionality is the `dominant_color_extraction` function. This function takes in:

- `image_path`: Path to the image file.
- `k`: Number of clusters for KMeans (default is 3).

It returns the RGB tuple of the dominant color.

Example:

```python
image_path = 'path_to_your_image.jpg'
color = dominant_color_extraction(image_path)
print(f"Dominant Color: {color}")
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
