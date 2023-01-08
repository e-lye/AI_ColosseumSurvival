# Student agent: Add your own agent here
import time
import numpy as np
from copy import deepcopy
from agents.agent import Agent
from store import register_agent

import sys

@register_agent("student_agent3")
class StudentAgent(Agent):

    def __init__(self):
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.zoneTiles = []
        self.start_time = 0
        self.board_size = 0
        self.initialized = False
        self.time_limit = 1.992
        self.curr_best = None
        self.time_up = False
        self.autoplay = True
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        # Opposite Directions
        self.opposites = {0: 2, 1: 3, 2: 0, 3: 1}

    def step(self, chess_board, my_pos, adv_pos, max_step):
        # first step
        if not self.initialized:
            self.board_size = chess_board.shape[0]
            self.initialized = True

        # find a cheap panic move in case we lack time (one of the 4 directions will always be a valid move, otherwise the game would be done).
        r, c = my_pos
        for bdir in range(4):
            if not chess_board[r, c, bdir]:
                self.curr_best = tuple(((r,c), bdir))
                break
        
        self.start_time = time.time()
        
        # iterative deepening minimax search w/ alpha beta pruning
        try:
            search_depth = 1
            while(True):
                (r, c, b_dir), stop_deliberate = self.alphabeta(chess_board, my_pos, adv_pos, max_step, search_depth)
                # print("returned on depth: ", search_depth)
                search_depth = search_depth+1
                if stop_deliberate:
                    # found winning move
                    return ((r, c), b_dir)
                else:
                    # record the current best move 
                    self.curr_best = tuple(((r,c), b_dir))
        except TimeUpException as e:
            return self.curr_best

    # [Finished] Minimax search with cut-off depth and alpha beta pruning
    # [Returns] (move:(r, c, dir), stop_deliberating: Boolean) 
    def alphabeta (self, chess_board, my_pos, adv_pos, max_step, max_depth):
        def terminal_test(chess_board, my_pos, adv_pos, max_step, curr_depth):
            rep = self.agents_area(chess_board, my_pos, adv_pos)
            isEnd, score = rep
            if curr_depth == max_depth:
                return self.eval_func(chess_board, my_pos, adv_pos, max_step, score)
            if isEnd:
                if score > 0:
                    return float('inf')
                elif score < 0:
                    return float('-inf')
                else:
                    return self.eval_func(chess_board, my_pos, adv_pos, max_step, score)
            return None
        
        def max (chess_board, my_pos, adv_pos, max_step, alpha, beta, curr_depth):
            self.check_time()
            eval = terminal_test(chess_board, my_pos, adv_pos, max_step, curr_depth)
            if eval is not None:
                # end game leaf node
                return (eval, None)
            else:
                eval = float('-inf')
                new_alpha = alpha
                
                valid_moves = self.get_all_valid_steps(chess_board, my_pos, adv_pos, max_step)
                
                move_to_eval = {}
                for move in valid_moves:
                    r, c, b_dir = move

                    cur_board = deepcopy(chess_board)
                    self.set_barrier(cur_board, r, c, b_dir)
                    
                    min_value, mv = min(cur_board, adv_pos, (r, c), max_step, new_alpha, beta, curr_depth+1)
                    move_to_eval[move] = min_value

                    if min_value > eval:
                        eval = min_value
                    if eval >= beta:
                        # beta cut-off
                        return (eval, move)
                    new_alpha = eval if eval > alpha else new_alpha
                for mv, ev in move_to_eval.items():
                    if ev == eval:
                        return (eval, mv)
        
        def min (chess_board, my_pos, adv_pos, max_step, alpha, beta, curr_depth):
            self.check_time()
            eval = terminal_test(chess_board, adv_pos, my_pos, max_step, curr_depth)
            # flip adv and my pos since we want the utility of the max player
            if eval is not None:
                # end game leaf node
                return (eval, None)
            else:
                eval = float('inf')
                new_beta = beta
                
                valid_moves = self.get_all_valid_steps(chess_board, my_pos, adv_pos, max_step)

                move_to_eval = {}
                for move in valid_moves:
                    r, c, b_dir = move

                    cur_board = deepcopy(chess_board)
                    self.set_barrier(cur_board, r, c, b_dir)
                    
                    max_value, mv = max(cur_board, adv_pos, (r, c), max_step, alpha, new_beta, curr_depth+1)
                    move_to_eval[move] = max_value

                    if max_value < eval:
                        eval = max_value
                    if eval <= alpha:
                        # alpha cut-off
                        return (eval, move)
                    new_beta = eval if eval < new_beta else new_beta
                for mv, ev in move_to_eval.items():
                    if ev == eval:
                        return (eval, mv)
                
        value, move = max(chess_board, my_pos, adv_pos, max_step, float('-inf'), float('inf'), 0)
        
        stop_deliberating = False

        if value == float('inf'):
            # check mate stop thinking
            stop_deliberating = True
        if value == float('-inf'):
            # losing stop thinking
            stop_deliberating = True
            # don't want to give up in case of weak opponent 
            # execute previous best move
            (r,c), bdir = self.curr_best
            return (r,c, bdir), stop_deliberating
        
        return (move, stop_deliberating)

    def eval_func(self, chess_board, my_pos, adv_pos, max_step, area):
        #TODO implement
        # print(1 - self.get_manhattan_dist(my_pos, adv_pos) / (self.board_size * 2))
        # print (self.get_number_of_reachable_tiles(chess_board, my_pos, adv_pos, max_step))
        return self.get_number_of_reachable_tiles(chess_board, my_pos, adv_pos, max_step) \
            #    - self.get_number_of_reachable_tiles(chess_board, adv_pos, my_pos, max_step)
            #    + (self.board_size**2 - area)
            #    + 0.1 * self.get_number_of_adj_barriers(chess_board, adv_pos) \
            #    - 0.1 * self.get_number_of_adj_barriers(chess_board, my_pos)
        # return 1 - self.get_manhattan_dist(my_pos, adv_pos) / (self.board_size ** 2) \

        return 0

    # [Finished] Calculate and return all possible moves and barrier placements
    # [Returns] validMoves := Queue<((r, c, dir))>
    def get_all_valid_steps(self, chess_board, my_pos, adv_pos, max_step):

        # BFS
        all_queue = []
        state_queue = [(my_pos, 0)]
        visited = set()
        visited.add(my_pos)
        
        while state_queue:
            
            cur_pos, cur_step = state_queue.pop(0)
            cur_r, cur_c = cur_pos

            # BFS, so if we popped out a max_step+1, we already looked at the entire max_step layer
            if cur_step == max_step + 1:
                break

            for dir, move in enumerate(self.moves):
                
                if chess_board[cur_r, cur_c, dir]:
                    continue

                # Append the direction to our possible moves
                all_queue.append((cur_r, cur_c, dir))

                # Get the next position
                next_r = cur_pos[0] + move[0]
                next_c = cur_pos[1] + move[1]
                next_pos = next_r, next_c

                if np.array_equal(next_pos, adv_pos) or next_pos in visited:
                    continue

                visited.add(next_pos)
                state_queue.append((next_pos, cur_step + 1))
        
        return all_queue

    # [Finished, taken from world.py check_endgame] Checks the agents area score
    # [Returns] (gameEnded:Boolean, agentScore: int) 
    def agents_area(self, chess_board, my_pos, adv_pos):
        
        N = chess_board.shape[0]

        # Union-Find
        father = dict()
        for r in range(N):
            for c in range(N):
                father[(r, c)] = (r, c)

        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(N):
            # self.check_time()
            for c in range(N):
                for dir, move in enumerate(
                    self.moves[1:3]
                ):  # Only check down and right
                    if chess_board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(N):
            for c in range(N):
                find((r, c))

        p0_r = find(tuple(my_pos))
        p1_r = find(tuple(adv_pos))

        if p0_r == p1_r:
            return False, 0

        my_score = list(father.values()).count(p0_r)
        adv_score = list(father.values()).count(p1_r)

        return True, my_score - adv_score

    # [Finished, taken from world.py] Sets the barrier
    def set_barrier(self, chess_board, r, c, bar_dir):
        # Set the barrier to True
        chess_board[r, c, bar_dir] = True
        # Set the opposite barrier to True
        move = self.moves[bar_dir]
        chess_board[r + move[0], c + move[1], self.opposites[bar_dir]] = True

    def check_time(self):
        # print(time.time() - self.start_time)
        if time.time() - self.start_time > self.time_limit:
            # print(time.time() - self.start_time)
            # print("Time Up")
            raise TimeUpException

    def get_manhattan_dist(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def get_number_of_adj_barriers(self, chess_board, curr_pos):
        r, c = curr_pos
        count = 0
        for bdir in range(4):
            if chess_board[r, c, bdir]:
                count = count + 1
        return count

    def get_number_of_reachable_tiles(self, chess_board, my_pos, adv_pos, max_step):
        # BFS
        all_queue = []
        state_queue = [(my_pos, 0)]
        visited = set()
        visited.add(my_pos)
        num = 0
        
        while state_queue:
            
            cur_pos, cur_step = state_queue.pop(0)
            cur_r, cur_c = cur_pos

            # BFS, so if we popped out a max_step+1, we already looked at the entire max_step layer
            if cur_step == max_step + 1:
                break

            num = num + 1

            for dir, move in enumerate(self.moves):
                
                if chess_board[cur_r, cur_c, dir]:
                    continue

                # Append the direction to our possible moves
                all_queue.append((cur_r, cur_c, dir))

                # Get the next position
                next_r = cur_pos[0] + move[0]
                next_c = cur_pos[1] + move[1]
                next_pos = next_r, next_c

                if np.array_equal(next_pos, adv_pos) or next_pos in visited:
                    continue

                visited.add(next_pos)
                state_queue.append((next_pos, cur_step + 1))
        
        return num
    
class TimeUpException(Exception):
    pass
