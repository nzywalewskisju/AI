#######################################################
#### 
#### Author: Nick Zywalewski
#### Homework 3, Problem 1
#### Purpose: Comparing the behaviors of the Greedy Best-First and A* search algorithms, using Manhattan distance as the heuristic function.
#### Note:
#### --- Change line 45 as needed to switch between the two algorithms.
#### 
#######################################################
import tkinter as tk
#from PIL import ImageTk, Image, ImageOps 
from queue import PriorityQueue



######################################################

#### A cell stores f(), g() and h() values
#### A cell is either open or part of a wall
######################################################

class Cell:
    #### Initially, arre maze cells have g() = inf and h() = 0
    def __init__(self, x, y, is_wall=False):
        self.x = x
        self.y = y
        self.is_wall = is_wall
        self.g = float("inf")
        self.h = 0
        self.f = float("inf")
        self.parent = None

    #### Compare two cells based on their evaluation functions
    def __lt__(self, other):
        return self.f < other.f


######################################################
# A maze is a grid of size rows X cols
######################################################
class MazeGame:
    def __init__(self, root, maze):

        ### Set the search algorithm here: "A_star" or "Greedy"
        self.algorithm = "Greedy" # Change as needed

        self.root = root
        self.maze = maze
        
        self.rows = len(maze)
        self.cols = len(maze[0])

        #### Start state: Row 2, Column 1
        self.agent_pos = (1, 0)

        #### Goal state:  (rows-1, cols-1) or bottom right
        self.goal_pos = (self.rows - 1, self.cols - 1)
        
        self.cells = [[Cell(x, y, maze[x][y] == 1) for y in range(self.cols)] for x in range(self.rows)]
        
        #### Start state's initial values for f(n) = g(n) + h(n) 
        self.cells[self.agent_pos[0]][self.agent_pos[1]].g = 0
        self.cells[self.agent_pos[0]][self.agent_pos[1]].h = self.heuristic(self.agent_pos)
        self.cells[self.agent_pos[0]][self.agent_pos[1]].f = self.heuristic(self.agent_pos)

        #### The maze cell size in pixels
        self.cell_size = 75
        self.canvas = tk.Canvas(root, width=self.cols * self.cell_size, height=self.rows * self.cell_size, bg='white')
        self.canvas.pack()

        self.draw_maze()
        
        #### Display the optimum path in the maze
        self.find_path()



    ############################################################
    #### This is for the GUI part. No need to modify this unless
    #### GUI changes are needed.
    ############################################################
    def draw_maze(self):
        for x in range(self.rows):
            for y in range(self.cols):
                color = 'maroon' if self.maze[x][y] == 1 else 'white'
                self.canvas.create_rectangle(y * self.cell_size, x * self.cell_size, (y + 1) * self.cell_size, (x + 1) * self.cell_size, fill=color)
                if not self.cells[x][y].is_wall:
                    text = f'g={self.cells[x][y].g}\nh={self.cells[x][y].h}'
                    self.canvas.create_text((y + 0.5) * self.cell_size, (x + 0.5) * self.cell_size, font=("Purisa", 12), text=text)



    ############################################################
    #### Manhattan distance
    ############################################################
    def heuristic(self, pos):
        return (abs(pos[0] - self.goal_pos[0]) + abs(pos[1] - self.goal_pos[1]))



    ############################################################
    #### Search Algorithm
    ############################################################
    def find_path(self):
        open_set = PriorityQueue()
        
        #### Add the start state to the queue
        open_set.put((0, self.agent_pos))

        #### Continue exploring until the queue is exhausted
        while not open_set.empty():
            current_cost, current_pos = open_set.get()
            current_cell = self.cells[current_pos[0]][current_pos[1]]

            #### Stop if goal is reached
            if current_pos == self.goal_pos:
                self.reconstruct_path()
                break

            
            #### Agent goes E, W, N, and S, whenever possible
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_pos = (current_pos[0] + dx, current_pos[1] + dy)

                if 0 <= new_pos[0] < self.rows and 0 <= new_pos[1] < self.cols and not self.cells[new_pos[0]][new_pos[1]].is_wall:
                
                    #### The cost of moving to a new position is 1 unit
                    new_g = current_cell.g + 1
                    
                    # Calculate the new heuristic and evaluation function values
                    new_h = self.heuristic(new_pos)
                    if self.algorithm == "A_star":
                        new_f = new_g + new_h # if A* search -- f(n) = g(n) + h(n) -- since h(n) is fixed for each cell, this is the same as comparing just g(n) values
                    else:
                        new_f = new_h # if Greedy Best-First search -- f(n) = h(n); no g(n) component

                    # Update only if this path lowers the evaluation f:
                    # --- A*:  f = g + h (since h is position-only, this is the same as only checking g)
                    # --- Greedy: f = h  (first discovery typically wins; later visits won't lower f)   
                    if new_f < self.cells[new_pos[0]][new_pos[1]].f:
                        # Update the path cost g()
                        self.cells[new_pos[0]][new_pos[1]].g = new_g
                        # Update the heuristic h()
                        self.cells[new_pos[0]][new_pos[1]].h = new_h
                        # Update the evaluation function f()
                        self.cells[new_pos[0]][new_pos[1]].f = new_f
                        self.cells[new_pos[0]][new_pos[1]].parent = current_cell
                        # Add the new cell to the priority queue
                        open_set.put((new_f, new_pos))
                        
                        

    ############################################################
    #### This is for the GUI part. No need to modify this unless
    #### screen changes are needed.
    ############################################################
    def reconstruct_path(self):
        current_cell = self.cells[self.goal_pos[0]][self.goal_pos[1]]
        while current_cell.parent:
            x, y = current_cell.x, current_cell.y
            self.canvas.create_rectangle(y * self.cell_size, x * self.cell_size, (y + 1) * self.cell_size, (x + 1) * self.cell_size, fill='green')
            current_cell = current_cell.parent

            # Redraw cell with updated g() and h() values
            self.canvas.create_rectangle(y * self.cell_size, x * self.cell_size, (y + 1) * self.cell_size, (x + 1) * self.cell_size, fill='skyblue')
            text = f'g={self.cells[x][y].g}\nh={self.cells[x][y].h}'
            self.canvas.create_text((y + 0.5) * self.cell_size, (x + 0.5) * self.cell_size, font=("Purisa", 12), text=text)


    ############################################################
    #### This is for the GUI part. No need to modify this unless
    #### screen changes are needed.
    ############################################################
    def move_agent(self, event):
    
        #### Move right, if possible
        if event.keysym == 'Right' and self.agent_pos[1] + 1 < self.cols and not self.cells[self.agent_pos[0]][self.agent_pos[1] + 1].is_wall:
            self.agent_pos = (self.agent_pos[0], self.agent_pos[1] + 1)


        #### Move Left, if possible            
        elif event.keysym == 'Left' and self.agent_pos[1] - 1 >= 0 and not self.cells[self.agent_pos[0]][self.agent_pos[1] - 1].is_wall:
            self.agent_pos = (self.agent_pos[0], self.agent_pos[1] - 1)
        
        #### Move Down, if possible
        elif event.keysym == 'Down' and self.agent_pos[0] + 1 < self.rows and not self.cells[self.agent_pos[0] + 1][self.agent_pos[1]].is_wall:
            self.agent_pos = (self.agent_pos[0] + 1, self.agent_pos[1])
   
        #### Move Up, if possible   
        elif event.keysym == 'Up' and self.agent_pos[0] - 1 >= 0 and not self.cells[self.agent_pos[0] - 1][self.agent_pos[1]].is_wall:
            self.agent_pos = (self.agent_pos[0] - 1, self.agent_pos[1])

        #### Erase agent from the previous cell at time t
        self.canvas.delete("agent")

        
        ### Redraw the agent in color navy in the new cell position at time t+1
        self.canvas.create_rectangle(self.agent_pos[1] * self.cell_size, self.agent_pos[0] * self.cell_size, 
                                    (self.agent_pos[1] + 1) * self.cell_size, (self.agent_pos[0] + 1) * self.cell_size, 
                                    fill='navy', tags="agent")

                  

############################################################
#### Modify the wall cells to experiment with different maze
#### configurations.
############################################################
maze = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]



############################################################
#### The mainloop activates the GUI.
############################################################
root = tk.Tk()
root.title("Path Traversed")

game = MazeGame(root, maze)
root.bind("<KeyPress>", game.move_agent)

root.mainloop()