## functions to help debug / test things

import matplotlib.pyplot as plt

def mark_point_on_image(image, coordinate_list:list[tuple[int]], color='red'):
    """  """
    # dont want to modify origional
    image_copy = image.copy()

    # Display the image
    plt.imshow(image_copy, cmap='gray' if len(image.shape) == 2 else None)
    for coordinates in coordinate_list:
        plt.scatter(coordinates[0], coordinates[1], color=color, marker='x', s=100)  # 'x' marker for the point
    plt.show()