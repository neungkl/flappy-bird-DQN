import numpy as np
import collections

GRID_WIDTH = 4
GRID_HEIGHT = 3

ACTION_SIZE = 4

grid_world = [
  ['', '', '', 'G'],
  ['', 'X', '', 'R'],
  ['', '', '', '']
]

Q = np.zeros((GRID_HEIGHT, GRID_WIDTH, ACTION_SIZE))

action_dir = np.array([
  [0,1],  # Left
  [0,-1], # Right
  [1,0],  # Down
  [-1,0]  # Up
])

def is_correct_move(x, y, a):
  new_y = y + action_dir[a][0]
  new_x = x + action_dir[a][1]
  if new_x < 0 or new_x >= GRID_WIDTH or new_y < 0 or new_y >= GRID_HEIGHT:
    return False
  return grid_world[new_y][new_x] != 'X'

def is_end_game(x, y):
  return grid_world[y][x] != ''

def select_move(x, y):
  action_order = np.argsort(Q[y][x])
  for a in action_order:
    if is_correct_move(x, y, a):
      return a, x + action_dir[a][1], y + action_dir[a][0]

def random_move(x, y):
  action_order = np.arange(ACTION_SIZE)
  np.random.shuffle(action_order)
  for a in action_order:
    if is_correct_move(x, y, a):
      return a, x + action_dir[a][1], y + action_dir[a][0]

def calculate_reward(x, y):
  if grid_world[y][x] == '':
    return -0.01
  elif grid_world[y][x] == 'G':
    return 1.0
  return -1.0

explore_rate = 1.0
explore_rate_stop = 0.1
explore_rate_decay = 0.99

learning_rate = 0.1
discount_factor = 0.99

for update_t in range(10):
  state_x = 0
  state_y = GRID_HEIGHT - 1

  while not is_end_game(state_x, state_y):
    if np.random.rand() > explore_rate:
      action, new_x, new_y = select_move(state_x, state_y)
    else:
      action, new_x, new_y = random_move(state_x, state_y)

    reward = calculate_reward(new_x, new_y)

    Q[state_y][state_x][action] = (1 - learning_rate) * Q[state_y][state_x][action] \
      + learning_rate * (reward + discount_factor * np.max(Q[new_y][new_x]))
    
    state_x = new_x
    state_y = new_y

  explore_rate *= explore_rate_decay
  if explore_rate < explore_rate_stop:
    explore_rate = explore_rate_stop

print(Q)