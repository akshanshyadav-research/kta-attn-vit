import numpy as np
import torch
import torchvision.transforms as transforms
def bgr_to_rgb(img):
    """
    Convert an image from BGR to RGB color space.

    Args:
    - img: Input image in BGR format (numpy array)

    Returns:
    - img_rgb: Image converted to RGB color space
    """
    img_rgb = img[:, :, ::-1]
    return img_rgb

def resize_nearest_neighbor(image, size):
    original_height = len(image)
    original_width = len(image[0])
    (new_width, new_height) = size
    if isinstance(image[0][0], list) or isinstance(image[0][0], np.ndarray):
        resized_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)
        for i in range(new_height):
            for j in range(new_width):
                x = int(j * original_width / new_width)
                y = int(i * original_height / new_height)
                resized_image[i, j] = image[y][x]
    else:
        resized_image = np.zeros((new_height, new_width), dtype=np.uint8)
        for i in range(new_height):
            for j in range(new_width):
                x = int(j * original_width / new_width)
                y = int(i * original_height / new_height)
                resized_image[i, j] = image[y][x]
    return np.array(resized_image)

def transform_image(image, device):
    image = transforms.ToTensor()(image)
    image = image.unsqueeze(0)
    image = image.to(device)
    return image
