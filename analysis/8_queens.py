import random as rand
import math

def show_board(locations):
  board = [['.' for _ in range(8)] for _ in range(8)]
  for i in range(len(locations)):
    location = locations[i]
    x = location[0]
    y = location[1]
    board[x][y] = str(i)
  
  for row in board:
    print(' '.join(row))

def same_row(point1, point2):
  return point1[0] == point2[0]

def same_column(point1, point2):
  return point1[1] == point2[1]

def same_diagonal(point1, point2):
  slope = (point1[1] - point2[1])/(point1[0] - point2[0])
  return slope == 1 or slope == -1

def calc_cost(locations):
  cost = 0
  for i in range(len(locations)):
    p1 = locations[i]
    for j in range(len(locations)):
      p2 = locations[j]
      if i!= j and (same_row(p1,p2) or same_column(p1,p2) or same_diagonal(p1,p2)):
        cost += 1/2
  return cost

def random_optimizer(n):
  locations_list = [[(math.floor(9*rand.random()), math.floor(9*rand.random())) for _ in range(8)] for _ in range(n)]
  costs = [calc_cost(locations) for locations in locations_list]
  min_cost = costs[0]
  index =  0
  for i in range(len(costs)):
    cost = costs[i]
    if cost < min_cost:
      min_cost = cost
      index = i

  return {'locations':locations_list[index], 'cost':min_cost} 
    

locations = [(0,0), (6,1), (2,2), (5,3), (4,4), (7,5), (1,6), (2,6)]
show_board(locations)
print(calc_cost(locations))
nums = [10,50,100,500,1000]
for n in nums:
  print('n =', n, ':', random_optimizer(n))
