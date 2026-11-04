import numpy as np
import random

class TicTacToe:
    def __init__(self):
        self.board = [0] * 9

    def available_moves(self):
        return [i for i in range(9) if self.board[i] == 0]

    def check_winner(self):
        wins = [(0,1,2), (3,4,5), (6,7,8),
                (0,3,6), (1,4,7), (2,5,8),
                (0,4,8), (2,4,6)]

        for a, b, c in wins:
            if self.board[a] == self.board[b] == self.board[c] != 0:
                return self.board[a]

        return 0 if 0 not in self.board else None

    def reset(self):
        self.board = [0] * 9

Q = {}

alpha = 0.5
gamma = 0.9
epsilon = 0.2

def get_q(state, action):
    return Q.get((state, action), 0.0)

def choose_action(state, moves):
    if random.random() < epsilon:
        return random.choice(moves)

    return max(moves, key=lambda a: get_q(state, a))

# Step 3: Train the Agent
for episode in range(20000):
    game = TicTacToe()

    while True:
        state = tuple(game.board)
        moves = game.available_moves()
        action = choose_action(state, moves)

        game.board[action] = 1
        result = game.check_winner()

        if result is not None:
            reward = 1 if result == 1 else 0.5
            Q[(state, action)] = get_q(state, action) + alpha * (
                reward - get_q(state, action))
            break

        # Random opponent
        opponent_move = random.choice(game.available_moves())
        game.board[opponent_move] = -1

        result = game.check_winner()

        if result is not None:
            reward = -1 if result == -1 else 0.5
            Q[(state, action)] = get_q(state, action) + alpha * (
                reward - get_q(state, action))
            break

        next_state = tuple(game.board)
        next_moves = game.available_moves()

        best_next = max(get_q(next_state, a) for a in next_moves)

        old_q = get_q(state, action)
        Q[(state, action)] = old_q + alpha * (
            gamma * best_next - old_q
        )

# Step 4: Test the Trained Agent
epsilon = 0

game = TicTacToe()

print("Tic-Tac-Toe Game")
print("You are O (-1), Computer is X (1)")

while True:
    print(np.array(game.board).reshape(3, 3))

    # Human move
    move = int(input("Enter position (0-8): "))

    if move not in game.available_moves():
        print("Invalid move!")
        continue

    game.board[move] = -1

    result = game.check_winner()

    if result is not None:
        print(np.array(game.board).reshape(3, 3))
        print("You Win!" if result == -1 else "Draw!")
        break

    # Computer move
    state = tuple(game.board)
    moves = game.available_moves()
    action = max(moves, key=lambda a: get_q(state, a))

    game.board[action] = 1

    result = game.check_winner()

    if result is not None:
        print(np.array(game.board).reshape(3, 3))
        print("Computer Wins!" if result == 1 else "Draw!")
        break
