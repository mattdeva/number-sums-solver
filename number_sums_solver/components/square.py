class Square:
    def __init__(self, r:int, c:int, value:int, color:str|None=None, active:bool=True, selected:bool=False):
        self.r = r
        self.c = c
        self.value = value
        self.color = 'white' if color is None else color
        self.active = active
        self.selected = selected

    def __str__(self):
        return f'Square({self.r}, {self.c})'
    
    def __repr__(self):
        return f'Square({self.r}, {self.c}, {self.value}, {self.active}, {self.selected})'
    
    def __hash__(self):
        return hash((self.r, self.c, self.value))
    
    def __eq__(self, other):
        if isinstance(other, Square):
            return (self.r, self.c, self.value) == (other.r, other.c, other.value)
        return False

    @property
    def coordinates(self) -> tuple[int]:
        # changed to row, col... much easier for me while indexing df. keeping coordinates tho :)
        return self.r, self.c
    
    def select(self) -> None:
        self.selected = True

    def deactivate(self) -> None:
        self.active = False