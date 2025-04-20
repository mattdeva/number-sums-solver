import numpy as np
import pandas as pd
import math
import cv2
import matplotlib.pyplot as plt
import itertools
import pytesseract

from typing import Sequence, Optional

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


class MatrixImage:
    def __init__(
            self, 
            image:np.ndarray, 
            gray_code=cv2.COLOR_BGR2GRAY, 
            edge_upper_threshold=1000, 
            edge_lower_threshold=350, 
            retr_external=cv2.RETR_EXTERNAL, 
            chain_approx_simple=cv2.CHAIN_APPROX_SIMPLE
        ):
        self.image = image
        self._gray_code = gray_code
        self._edge_upper_threshold = edge_upper_threshold
        self._edge_lower_threshold = edge_lower_threshold
        self._retr_external = retr_external
        self._chain_approx_simple = chain_approx_simple

    @classmethod
    def from_path(cls, path:str):
        return cls(cv2.imread(path))

    @property
    def grey(self):
        return cv2.cvtColor(self.image, self._gray_code)
    
    @property
    def edge(self):
        return cv2.Canny(self.grey, self._edge_upper_threshold, self._edge_lower_threshold)
    
    @property
    def contours(self):
        return cv2.findContours(self.edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    
    def show_image_with_axis(self):
        plt.imshow(self.image)
        
    def show_image(self):
        plt.imshow(self.image)
        plt.axis(False)
        plt.show()

    def show_grey(self):
        plt.imshow(self.grey)
        plt.axis(False)
        plt.show()

    def show_edge(self):
        plt.imshow(self.edge)
        plt.axis(False)
        plt.show()

def int_from_img(img) -> Optional[int]:
    str_ = pytesseract.image_to_string(img, config='--psm 8 -c tessedit_char_whitelist=0123456789').replace('\n','')    
    return None if str_ == '' else int(str_)

def parse_values_from_image(
        matrix_image:MatrixImage=None,
        x_y_buffer:range=range(5,25,5), 
        w_h_buffer:range=range(10,30,5), 
        verbose=False
    ):
    value_dict = {}
    missing_value_regions = {}
    for i, contour in enumerate(matrix_image.contours):
        for buffers in itertools.product(x_y_buffer, w_h_buffer):
            x,y,w,h = cv2.boundingRect(contour)
            x-=buffers[0]
            y-=buffers[0]
            w+=buffers[1]
            h+=buffers[1]
            region = matrix_image.grey[y:y + h, x:x + w]

            img_decode = int_from_img(region)

            if isinstance(img_decode, int):
                value_dict[(x,y)] = img_decode
                break
        if img_decode is None:
            # if/when build model can include here
            if verbose:
                print(f'cannot identify value. {i}')
            x,y,w,h = cv2.boundingRect(contour)
            missing_value_regions[(x,y)] = matrix_image.grey[y:y + h, x:x + w]

    return value_dict, missing_value_regions

def find_pairs(coordinates:Sequence[tuple[int]], x_buffer:int=75, y_buffer:int=10) -> list[tuple[int]]:
    ''' return coordinates that within specified buffer '''
    # NOTE: limiting to pairs instead of an unspecified number of values is a potential bug (feature improvement opportunity)
    pairs = []
    for tup in coordinates:
        x,y = tup
        possible_pairs = [t for t in coordinates if 0 < t[0]-x < x_buffer] # return coordinates to the right of x position within buffer
        possible_pairs = [t for t in possible_pairs if -1*y_buffer < t[1]-y < y_buffer] # return coordinates within buffer of y position

        if len(possible_pairs) > 1:
            raise ValueError(f'more than 1 possible pair identified. {tup}, {possible_pairs}')
        if len(possible_pairs) == 1:
            pairs.append((tup, possible_pairs[0]))

    return pairs

def concat_digits(*args):
    return int("".join([str(arg) for arg in args]))

def create_multi_digit_dict(coordinates:Sequence[tuple[tuple[int]]], coord_value_dict:dict[tuple[int], int]) -> dict[tuple[int],int]: # great name
    ''' take sequence of individual digits that should be considered 1, create dictionary with starting coordinate (key) and value (value)'''
    # NOTE: only works if pairs
    multi_digit_dict = {}
    for tup in coordinates:
        c1, c2 = coord_value_dict[tup[0]], coord_value_dict[tup[1]]
        multi_digit_dict[tup[0]] = concat_digits(c1,c2)

    return multi_digit_dict

def update_value_dict(
        value_dict:dict[tuple[int], int],
        pairs:Sequence[tuple[tuple[int]]],
        multi_digit_dict:dict[tuple[int], int]
    ):
    return {**{(0,0):0}, **{k:v for k,v in value_dict.items() if k not in [p[1] for p in pairs]}, **multi_digit_dict}

def floor_round(number:int):
    return math.floor(number/100)*100

def create_value_tuple(d:dict[tuple[int], int]) -> tuple[int]:
    ''' create sorted list of tuples'''
    list_ = [(k[0], k[1], v) for k,v in d.items()]
    return sorted(list_, key=lambda x: (floor_round(x[1]), x[0]))

def df_from_value_list(list_:list[int]) -> pd.DataFrame:
    length = len(list_)
    if math.isqrt(length)**2 != length:
        raise ValueError("length of the list must be a perfect square. got {length}")
    
    n = math.isqrt(length) 
    
    # Reshape the list row-wise into a matrix
    return pd.DataFrame([list_[i * n:(i + 1) * n] for i in range(n)]) # create records of length n from list

def df_from_matrix_iamge(
        matrix_image:MatrixImage, 
        x_y_buffer:range=range(5,25,5), 
        w_h_buffer:range=range(10,30,5), 
        x_buffer:int=75, 
        y_buffer:int=10,
        verbose=False
    ):
    ''' wrapper function to parse values from image and create dataframe '''

    # create dictionary of coordinates of number start (top left of contour), and the value {(x_coord, y_coord):cell_value}
    value_dict = parse_values_from_image(matrix_image, x_y_buffer, w_h_buffer, verbose)[0]

    # from the dictionary, identify digits that are apart of the same number (2 digit numbers)
    pairs = find_pairs(list(value_dict), x_buffer, y_buffer)

    # create a dictionary from the double digit numbers. number start (top left of first digit contour) and value {(x_coord, y_coord):cell_value}
    multi_digit_dict = create_multi_digit_dict(pairs, value_dict)

    # update the dictionary to include double digit numbers (making new variable for debugging)
    updated_value_dict = update_value_dict(value_dict, pairs, multi_digit_dict)

    # create ordered list ordering by the y-values, the x-values. additionally, add a 0 to start
    value_list = [t[2] for t in create_value_tuple(updated_value_dict)]

    # convert the ordered list to dataframe. create rowise. length of columns (and rows) will be sqrt of value_list length
    return df_from_value_list(value_list)
