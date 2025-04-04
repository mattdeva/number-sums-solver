from number_sums_solver.components.solver import Solver, _flatten_and_unique
from number_sums_solver.components.square import Square

def _check_unique_coordinates(squares) -> None:
    ''' ensure unique coordinates in a list of coordinates '''
    coordinates = [square.coordinates for square in squares]

    if len(coordinates) != len(set(coordinates)):
        raise ValueError(f'non unique coordinates in Squares')

class Group:

    def __init__(self, squares:list[Square], nominal_target:int, name:str|None=None):
        self.squares = squares
        _check_unique_coordinates(self.squares) # not sure if putting here is best practice...

        self.nominal_target = nominal_target
        self.name = f'{self.nominal_target}' if name is None else name # name might be helpful later to debug, but not required

    def __str__(self):
        return self.name
    
    def __getitem__(self, index):
        return self.squares[index]

    def __len__(self):
        return len(self.squares)

    @staticmethod
    def _get_key(d:dict):
        ''' returns the key of a dictionary length 1 '''
        if not len(d) == 1:
            raise ValueError(f'Expecting dictionary length 1. got {len(d)}')
        return next(iter(d))

    @property
    def running_target(self):
        target = self.nominal_target
        selected_squares = [square for square in self.squares if square.selected] # only want to loop through squares selected
        for selected_square in selected_squares: 
            target -= selected_square.value # subtract value from total
        return target
    
    @property
    def available_squares(self) -> list[Square]:
        ''' return a list of squares that are not disactivated, and not selected '''
        return [square for square in self.squares if square.active and not square.selected]
    
    @property
    def selected_squares(self) -> list[Square]:
        ''' return a list of squares that are not disactivated, and not selected '''
        return [square for square in self.squares if square.selected]
    
    @property
    def solved(self):
        return sum([s.value for s in self.selected_squares]) == self.nominal_target

    def _get_square_int_dict(self):
        return dict(zip(self.available_squares, [square.value for square in self.available_squares]))
    
    def squeeze(self, verbose:bool=False):
        ''' disactivate squares that cannot be correct '''

        square_int_dict = self._get_square_int_dict()
        list_ = Solver(square_int_dict.values()).squeeze(self.running_target)
        if len(list_) == 1:
            for square, int_ in square_int_dict.items():
                if int_ in list_[0]:
                    square.select()
                    print(f'select {square}') if verbose else None
                else:
                    square.deactivate()
                    print(f'deactivate {square}') if verbose else None
        
        elif len(list_) > 1:
            int_list = _flatten_and_unique(list_) # {(1, 3): 4, (2, 2): 4} -> [1,2,3]
            for square, int_ in square_int_dict.items():
                if int_ not in int_list:
                    square.deactivate()
                    print(f'deactivate {square}') if verbose else None

        else: # deactivate all non-selected squares
            for square, int_ in square_int_dict.items():
                if square not in self.selected_squares:
                    square.deactivate()