# People Counter Project

## Overview

This project implements a people counter system using computer vision techniques. The main goal is to detect and count people entering and exiting a specific area, such as a hotel entrance.

## Key Components

1. **Person Detection**: Utilizes the YOLOv8 model for detecting people in video frames.
2. **Region of Interest (ROI)**: Defines specific areas to focus the detection and counting process.
3. **Tracking**: Implements a tracking system to assign unique IDs to detected individuals.
4. **Counting Logic**: Determines the direction of movement (entering or exiting) based on the person's position relative to defined areas.

## Implementation Details

### Region of Interest (ROI)

Two areas of interest are defined:
- Area 1: Represents the inner boundary
- Area 2: Represents the outer boundary

<img width="237" alt="Picture 1" src="https://github.com/user-attachments/assets/0cf89476-8d6b-4536-b7ac-029fde2f771f">


The ROI is drawn using the `cv2.polylines` function. Coordinates for these areas are obtained using a mouse function:

```python
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)
```
<img width="482" alt="Picture 2" src="https://github.com/user-attachments/assets/0e108c8a-7190-401d-a8e6-794ad5f9c5f8">

### Person Detection

- The system detects people using the YOLOv8 model.
- Detection is limited to the defined ROI to focus on relevant areas.

### Tracking

- A tracking system assigns unique IDs to each detected person.
- This allows for consistent identification across frames.

### Counting Logic

- For a person to be counted as entering:
  1. They must first be detected in Area 2 (outer boundary)
  2. Then move to Area 1 (inner boundary)
- For a person to be counted as exiting:
  1. They must first be detected in Area 1 (inner boundary)
  2. Then move to Area 2 (outer boundary)

## Visual Indicators

- Blue polygons: Represent the areas of interest
- Green rectangles: Bounding boxes around detected persons
- Pink circles: Marked at the right-most corner of each bounding box
- White text: Displays the unique ID for each detected person

<img width="482" alt="Picture 3" src="https://github.com/user-attachments/assets/3fd33fe2-dec7-412c-bb3e-11b7d41786ca">

## Key Functions

1. `cv2.polylines()`: Draws the regions of interest
2. `cv2.pointPolygonTest()`: Determines if a point (person) is within a defined area
3. `tracker.update()`: Updates the tracking information for detected persons

## Notes

- The system focuses on detecting persons only when they are within the defined outer boundary (Area 2).
- The tracking system helps in maintaining consistency and avoiding double-counting.

## Future Improvements

- Implement a more robust tracking algorithm to handle occlusions and crowded scenes.
- Add a user interface for real-time display of entry and exit counts.
- Integrate with a database system for long-term data storage and analysis.
