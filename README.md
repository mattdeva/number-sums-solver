# Number-Sums-Solver
Number_Sums_Solver, as the name would suggest, is a program designed to solve for number sums puzzles. Below is an example of the game.

# Number Sums Game 

## Description
The number sums puzzle is a game in which the player will attempt to select the numbers in the matrix to equal the values on the outside for each row and column. While there may be many ways to sum the values in an individual row or column, the challenge involves selecting the numbers that solve all rows and columns. The game can also increase the difficulty by adding color sections to the matrix. Similar to the rows and column, the player must ensure the values within the color section equal the value for the color. 

## Example Puzzle

### Start
|   | 5 | 3 | 4 |
|---|---|---|---|
| 4 | 3 | 1 | 4 |
| 6 | 2 | 3 | 4 |
| 2 | 2 | 2 | 1 |

### Solved
|   | 5 | 3 | 4 |
|---|---|---|---|
| 4 | 3 | 1 |   |
| 6 | 2 |   | 4 |
| 2 |   | 2 |   |

# Creating a Puzzle for Number-Sums-Solver
There are several ways to load a puzzle into the program:
1. Programatically create the puzzle using solver functions
2. Create the puzzle in an excel file
3. Upload an image of the puzzle (currently not able to handle colors)

See demos for more detail.

# Future development
The main functions that need to be completed include finishing the image processing to handle puzzles with colors and creating a GUI where users are able to create puzzles more easily.
