"""Validate dataset output resolution and resize before drawing objects."""

import cv2


def validate_image_size(size):
    """Accept a positive integer pixel size, excluding booleans and floats."""
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("data_generation.size must be a positive integer")
    return size


def resize_background(image, size):
    """Produce a square canvas; objects and labels are created afterward."""
    validate_image_size(size)
    height, width = image.shape[:2]
    if (height, width) == (size, size):
        return image
    interpolation = cv2.INTER_AREA if size <= min(height, width) else cv2.INTER_LINEAR
    return cv2.resize(image, (size, size), interpolation=interpolation)
