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
print('Random Optimizer:')
nums = [10,50,100,500,1000]
for n in nums:
  print('n =', n, ':', random_optimizer(n))


def steepest_descent_optimizer(n):
  random_configuration = random_optimizer(100)
  current_locations = random_configuration['locations']
  current_cost = random_configuration['cost']
  for _ in range(n):
    locations_list = []
    for location_index in range(len(current_locations)):
      location = current_locations[location_index]
      for i_1 in range(-1,2):
        for i_2 in range(-1,2):
          new_locations = current_locations.copy()
          
          not_same = i_1, i_2 != 0
          on_board = location[0]+i_1 in [i for i in range(8)]and location[1]+i_2 in [i for i in range(8)]
          not_occupied = (location[0]+i_1, location[1]+i_2) not in current_locations
          if not_same and on_board and not_occupied: 
            new_locations[location_index] = (location[0]+i_1, location[1]+i_2)
            locations_list.append(new_locations)
    
    costs = [calc_cost(locations) for locations in locations_list]
    min_cost = costs[0]
    index =  0
    for i in range(len(costs)):
      cost = costs[i]
      if cost < min_cost:
        min_cost = cost
        index = i

    if min_cost < current_cost:
      current_locations = locations_list[index]
      current_cost = min_cost 
  
  return {'locations':current_locations, 'cost': current_cost}


print('Steepest Descent Optimizer:')
nums = [10,50,100,500,1000]
for n in nums:
  print('n =', n, ':', steepest_descent_optimizer(n))

            


          



