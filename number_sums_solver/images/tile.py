import numpy as np
import random
import math
from PIL import Image, ImageDraw, ImageFont, ImageColor
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt

from number_sums_solver.components.square import Square
from number_sums_solver.components.group import Group
from number_sums_solver.components.matrix import Matrix

def _get_len(list_:list) -> int:
    length = len(list_)
    if math.isqrt(length)**2 != length:
        raise ValueError("length of the list must be a perfect square. got {length}")
    return int(length**(1/2)+1)

def _get_size(size:tuple|int) -> tuple[int]:
    if isinstance(size, int):
        size = (size, size, 3)
    elif isinstance(size, tuple):
        if len(size) != 3:
            raise ValueError(f"size must be RGB compatible tuple len 3. got {len(size)}")
    return size

def _get_random_color(seed=None) -> tuple[str, tuple[int]]:
    random.seed(seed)
    color_name = random.choice(list(mcolors.CSS4_COLORS.keys())) 
    rgb_tuple = tuple(int(v * 255) for v in mcolors.to_rgba(color_name)[:3])
    return color_name, rgb_tuple

def _get_color(color:tuple|str):
    if isinstance(color, tuple):
        if len(color) != 3:
            raise ValueError(f"size must be RGB compatible tuple len 3. got {len(color)}")
    elif isinstance(color, str):
        try:
            color = ImageColor.getrgb(color)
        except ValueError as e: # NOTE: this part not working correctly yet. lower priority. after pushing changes, create issue
            color_name, rgb_tuple = _get_random_color()
            print(f'color {color} could not be identified. random color assignment {color_name}')
            color = rgb_tuple
    return color 

def _get_blank_tile(size:tuple|int=200, fill_color:tuple|str=(255,255,255)):
    return np.ones(size, dtype=np.uint8) * np.array(fill_color, dtype=np.uint8)

def get_tile_image(
        value:int,
        size:tuple|int=200,
        fill_color:tuple|str=(255,255,255),
        font_input:tuple=("arial.ttf", 120)
    ) -> np.ndarray:

    #TODO: change the file name to 'tiles?'. call this value to tile. will make a new one 'selected_tile'

    size = _get_size(size)
    fill_color = _get_color(fill_color)

    # create background
    np_img = _get_blank_tile(size, fill_color)

    # convert to img and draw objects
    img = Image.fromarray(np_img)
    draw = ImageDraw.Draw(img)
    
    # get text and font
    text = str(value)
    font = ImageFont.truetype(*font_input)

    # bounds for text
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    # center* text and draw
    text_x = (size[0] - text_width) // 2
    text_y = (size[1] - text_height) // 3 # *start 1/3 from top
    draw.text((text_x, text_y), text, fill="black", font=font)

    return np.array(img)

def show_matrix(matrix:Matrix, size:int|tuple[int]=5):

    size = _get_size(size)

    len_ = _get_len(matrix.squares)

    tiles = [np.ones((200,200,3), dtype=np.uint8) * np.array((255,255,255), dtype=np.uint8)]

    for row in range(len_):
        for col in range(len_):
            if row == 0 and col == 0:
                continue
            tile = matrix.get_tile((row, col))
            if isinstance(tile, Square):
                color = 'blue' if tile.color == 'FFE06666' else 'red' # 'FFE06666', 'FF6FA8DC'
                tiles.append(get_tile_image(tile.value, fill_color=color))
            else:
                tiles.append(get_tile_image(tile.nominal_target))

    fig, axes = plt.subplots(len_, len_, figsize=size)
    axes = axes.flatten()

    for ax, square in zip(axes, tiles):
        ax.imshow(square)
        ax.axis('off')

    plt.tight_layout()
    plt.show()
