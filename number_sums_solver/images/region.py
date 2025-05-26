from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
from shapely import Polygon

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

from typing import Optional


def _get_rand_scale(start:float, end:float, seed=None):
    random.seed(seed)
    return round(random.uniform(start,end),2)

def _int_from_img(img) -> Optional[int]:
    str_ = pytesseract.image_to_string(img, config='--psm 8 -c tessedit_char_whitelist=0123456789').replace('\n','')    
    return None if str_ == '' else int(str_)

class Region:
    def __init__(
            self, 
            img:np.ndarray, 
            x:int, 
            y:int, 
            w:int, 
            h:int, 
            value:int|None=None, 
            x_buffer:int=10, 
            y_buffer:int=10, 
            w_buffer:int=15, 
            h_buffer:int=15
        ):

        self.img = img
        self._x = x
        self._y = y
        self._w = w
        self._h = h
        self._value = value
        self.x_buffer = x_buffer
        self.y_buffer = y_buffer
        self.w_buffer = w_buffer
        self.h_buffer = h_buffer

        self._value_img = None

    def __str__(self):
        return f'Region({self._x}, {self._y}, {self._w}, {self._h})'
    
    def __repr__(self):
        return f'Region({self._x}, {self._y}, {self._w}, {self._h})'

    def __hash__(self):
        return hash((self._x, self._y, self._w, self._h))
    
    def __eq__(self, other):
        if isinstance(other, Region):
            return (self.img, self._x, self._y, self._w, self._h) == (other.img, other._x, other._y, other._w, other._h)
        return False

    @classmethod
    def from_contour(cls, img, contour:np.ndarray, **kwargs):
        x,y,w,h = cv2.boundingRect(contour)
        return cls(img,x,y,w,h,**kwargs)

    @property
    def value(self):
        # provides way to get value without calling `find_value`, but still giving ability to do so with extra options
        if self._value is None:
            self.find_value()
        return self._value if isinstance(self._value, int) else 0

    @property
    def x(self) -> int:
        return self._x - self.x_buffer
    
    @property
    def y(self) -> int:
        return self._y - self.y_buffer
    
    @property
    def w(self) -> int:
        return self._w + self.h_buffer
    
    @property
    def h(self) -> int:
        return self._h + self.h_buffer
    
    @property
    def area(self):
        return self._h*self._w
    
    @property
    def coordinates(self) -> tuple[int]:
        return self._x, self._y

    @property
    def _value_found(self):
        return isinstance(self._value, int)
    
    @property
    def polygon(self):
        return Polygon([(self.x, self.y), (self.x + self.w, self.y), (self.x + self.w, self.y + self.h), (self.x, self.y + self.h)])

    def _clip(self) -> np.ndarray:
        return self.img[self.y:self.y + self.h, self.x:self.x + self.w]

    def _rescale(self, fx:float=1.2, fy:float=.8) -> np.ndarray:
        return cv2.resize(self._clip(), None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA) # not specifying new image size (2nd arg) instead denote scale x,y 
    
    def _get_squish(self, seed=None) -> np.ndarray:
        fx_,fy_ = _get_rand_scale(.8,1,seed), _get_rand_scale(.9,1.2,seed) # most cases want to squish horizontally, but ok with sometimes squish otherway
        return self._rescale(fx_, fy_)
    
    def show(self) -> None:
        plt.imshow(self._clip())

    def merge(self, region:Region) -> Region:
        if not isinstance(region, Region):
            raise ValueError(f'Expected Region merge to Region. Got {type(region)}')
        
        if not (self.img == region.img).all():
            raise ValueError(f'To merge, Regions must be from same image.')
        
        x = min(self._x, region._x)
        y = min(self._y, region._y)
        w = max((self._x + self._w), (region._x + region._w)) - x
        h = max((self._y + self._h), (region._y + region._h)) - y

        return Region(self.img, x, y, w, h)

    def find_value(self, i_max=10, seed:int|None=None) -> None: # ok being called externally even if doesnt need to be. 

        self._value = _int_from_img(self._clip())
        if isinstance(self._value, int):
            self._value_img = self._clip()

        i = 0
        while i < i_max and not self._value_found:
            img = self._get_squish(seed)
            int_ = _int_from_img(img)
            if isinstance(int_, int):
                self._value = int_
                self._value_img = img
            i+=1