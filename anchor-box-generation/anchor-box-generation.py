import numpy as np

def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    """
    Generate anchor boxes for object detection.
    """
    # Write code here
    stride = image_size / feature_size        
    anchor_boxes = []
    for i in range(feature_size): # row
        for j in range(feature_size): # column
            for s in scales:
                for r in aspect_ratios:                            
                    cx = (j +  0.5) * stride
                    cy = (i +  0.5) * stride
                    w = s * np.sqrt(r)
                    h = s / np.sqrt(r)
                    anchor_box = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
                    anchor_boxes.append(anchor_box)
    return anchor_boxes