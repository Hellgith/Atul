import numpy as np
import random

maze = np.array([
    [0, 0, 0, 1],
    [1, 0, 0, 1],
    [0, 0, 0, 0],
    [0, 1, 1, 0]
])

start, goal = (0, 0), (3, 3)
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
Q = np.zeros((4, 4, 4))

alpha, gamma, epsilon = 0.1, 0.9, 0.2

for episode in range(2000):
    state = start
    for _ in range(100):
        r, c = state
        a = random.randrange(4) if random.random() < epsilon else np.argmax(Q[r, c])
        dr, dc = actions[a]
        nr, nc = r + dr, c + dc

        if not (0 <= nr < 4 and 0 <= nc < 4) or maze[nr, nc] == 1:
            next_state, reward = state, -5
        elif (nr, nc) == goal:
            next_state, reward = (nr, nc), 100
        else:
            next_state, reward = (nr, nc), -1

        Q[r, c, a] += alpha * (
            reward + gamma * np.max(Q[next_state[0], next_state[1]])
            - Q[r, c, a]
        )
        state = next_state
        if state == goal:
            break

state = start
path = [state]
for _ in range(20):
    r, c = state
    a = np.argmax(Q[r, c])
    dr, dc = actions[a]
    nr, nc = r + dr, c + dc

    if not (0 <= nr < 4 and 0 <= nc < 4) or maze[nr, nc] == 1:
        break
    state = (nr, nc)
    path.append(state)
    if state == goal:
        break

print("Learned Path:", path)
print("Goal Reached:", state == goal)
