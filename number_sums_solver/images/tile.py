import numpy as np
import random
import math
from PIL import Image, ImageDraw, ImageFont, ImageColor, ImageOps
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
import matplotlib.patches as patches

from number_sums_solver.components.square import Square
from number_sums_solver.components.group import Group
from number_sums_solver.components.matrix import Matrix
from number_sums_solver.components.utils import len_is_sqrt

def _get_len(list_:list) -> int:
    length = len(list_)
    if math.isqrt(length)**2 != length:
        raise ValueError(f"length of the list must be a perfect square. got {length}")
    return int(length**(1/2)+1)

def _get_size(size:tuple|int) -> tuple[int]:
    if isinstance(size, int):
        size = (size, size, 3)
    elif isinstance(size, tuple):
        if len(size) != 3:
            raise ValueError(f"size must be RGB compatible tuple len 3. got {len(size)}")
    return size

def _get_color(color:tuple|str): # very not great color
    if isinstance(color, tuple):
        if len(color) != 3:
            raise ValueError(f"size must be RGB compatible tuple len 3. got {len(color)}")
        return color # assume len 3 tuples are RGB
    elif isinstance(color, str):
        if color == '000000': # dont love that this is hardcoded but ok
           color = ImageColor.getrgb('white')
        else:
           color = ImageColor.getrgb(color) # TODO: will fail if color cannot be converted to RBG should create acceptable color list
    return color 

def _get_blank_tile(size:tuple|int=200, fill_color:tuple|str=(255,255,255)):
    size = _get_size(size)
    return np.ones(size, dtype=np.uint8) * np.array(fill_color, dtype=np.uint8)

def get_tile_image(
        value:int,
        size:tuple|int=200,
        fill_color:tuple|str=(255,255,255),
        font_input:tuple=("arial.ttf", 120)
    ) -> np.ndarray:

    size = _get_size(size)
    fill_color = _get_color(fill_color)

    # create background
    np_img = _get_blank_tile(size, fill_color)

    # convert to img and draw objects
    img = Image.fromarray(np_img)
    img = ImageOps.expand(img, border=3, fill="black")
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

def show_matrix(matrix:Matrix, plot_coordinates:bool=False, show:bool=True):

    len_is_sqrt(matrix.squares+[0])
    len_ = int(1+len(matrix.squares)**(1/2))

    tiles = [_get_blank_tile()]
    coordinates = [(0,0)] # big sloppy but ok 
    for row in range(len_):
        for col in range(len_):
            if row == 0 and col == 0:
                continue
            tile = matrix.get_tile((row, col))
            if isinstance(tile, Square):
                if not tile.active:
                    tiles.append(_get_blank_tile())
                else:
                    tiles.append(get_tile_image(tile.value, fill_color=tile.color))
            else:
                tiles.append(get_tile_image(tile.nominal_target))
            coordinates.append((row, col))

    fig, axes = plt.subplots(len_, len_, figsize=None) # NOTE: not sure why this needs to be None

    axes = axes.flatten()

    for i, (ax, square) in enumerate(zip(axes, tiles)):
        ax.imshow(square)
        if plot_coordinates and i>0:
            ax.text(0.5, -.1, coordinates[i], fontsize=8, ha='center', transform=ax.transAxes)
        ax.axis('off')

    # add legend with color groups target values
    if matrix.colors:
        legend_patches = [patches.Patch(color=color, label=f"{color}: {count}") for color, count in matrix.colors.target_dict.items()]

        fig.legend(handles=legend_patches, loc='upper left', title="Legend")

    plt.tight_layout()
    if show:
        plt.show()
    else:
        return fig
