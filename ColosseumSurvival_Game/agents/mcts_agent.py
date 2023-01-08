# Student agent: Add your own agent here
from agents.agent import Agent
from agents import minmax_agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy


import time

class MonteCarloAgent:
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.start_time = 0
        self.time_limit = 1.97
        self.num_sims = 5
        self.panic_move = None
        self.autoplay = True
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        # Opposite Directions
        self.opposites = {0: 2, 1: 3, 2: 0, 3: 1}

    def step(self, chess_board, my_pos, adv_pos, max_step):


        minmaxAgent = minmax_agent.MinmaxAgent()
        valid_moves = self.get_all_simple_steps(chess_board, my_pos, adv_pos, max_step)

        N = chess_board.shape[0]
        moves_count = len(valid_moves)     

        # Too large to run MCTS on
        if (N == 9 and moves_count > 120) or (N == 10 and moves_count > 100) or (N == 11 and moves_count > 90) or (N == 12 and moves_count > 80):
            return minmaxAgent.step(chess_board, my_pos, adv_pos, max_step)

        next_pos, next_dir = self.mcts_best_move(chess_board, valid_moves, adv_pos, max_step)

        return next_pos, next_dir

    # [Finished] Calculates the best result from our MCTS
    # [Returns] next_pos: (r, c), next_dir: int
    def mcts_best_move(self, chess_board, moves_list, adv_pos, max_step):
 
        start_time = time.time()
        moves = {}

        while True:
            
            if time.time() - start_time > self.time_limit:
                break

            for move in moves_list:

                if time.time() - start_time > self.time_limit:
                    break

                sim_r, sim_c, sim_bdir = move
                sim_board = deepcopy(chess_board)

                self.set_barrier(sim_board, sim_r, sim_c, sim_bdir)
                mcts_result = self.single_simulation(sim_board, (sim_r, sim_c), adv_pos, 1, max_step)
                move = ((sim_r, sim_c), sim_bdir)
                moves[move] = moves.get(move, 0) + mcts_result
        
        return max(moves, key = moves.get)
    
    # [Finished] Runs random simulations n times
    #  [Returns] the fraction of the number of games won
    def random_simulations(self, chess_board, my_pos, adv_pos, max_step):
        
        turn = 0
        sum = 0
        
        for i in range(self.num_sims):
            sim_board = deepcopy(chess_board)
            sum += self.single_simulation(sim_board, my_pos, adv_pos, turn, max_step)
        return sum


    # [Finished, taken from world.py] Checks if a step is valid
    # [Returns] isValid: Boolean
    def check_valid_step(self, chess_board, start_pos, end_pos, bar_dir, adv_pos, max_step):

        # Endpoint already has barrier, is boarder or is current pos
        r, c = end_pos
        if chess_board[r, c, bar_dir]:
            return False
        if np.array_equal(start_pos, end_pos):
            return True

        # BFS
        state_queue = [(start_pos, 0)]
        visited = {tuple(start_pos)}
        is_reached = False
        while state_queue and not is_reached:
            self.check_time()
            cur_pos, cur_step = state_queue.pop(0)
            r, c = cur_pos
            if cur_step == max_step:
                break
            for dir, move in enumerate(self.moves):
                if chess_board[r, c, dir]:
                    continue
                
                next_pos = cur_pos[0] + move[0],cur_pos[1] + move[1]
                if np.array_equal(next_pos, adv_pos) or tuple(next_pos) in visited:
                    continue
                if np.array_equal(next_pos, end_pos):
                    is_reached = True
                    break

                visited.add(tuple(next_pos))
                state_queue.append((next_pos, cur_step + 1))

        return is_reached

    # [Finished] given (turn : 0 = ours, 1 = adv), run a random simulation
    # [Returns]
    def single_simulation(self, chess_board, my_pos, adv_pos, turn, max_step):

        sim_board = deepcopy(chess_board)
        is_over, my_score = self.agents_area(sim_board, my_pos, adv_pos)

        # Loses in One Move
        if my_score == -10000:
            return -10000000
        # Ties
        elif my_score == -1:
            return -150
        # Wins in One Move
        elif my_score == 100:
            return 10000000

        loses_in_one = True

        #  While our simulation game has not ended
        while not is_over:
                
            if turn == 0:
                my_pos, my_bdir = self.random_move(sim_board, my_pos, adv_pos, max_step)
                self.set_barrier(sim_board, my_pos[0], my_pos[1], my_bdir)
                turn = 1
                loses_in_one = False
            elif turn == 1:
                adv_pos, adv_bdir = self.random_move(sim_board, adv_pos, my_pos, max_step)
                self.set_barrier(sim_board, adv_pos[0], adv_pos[1], adv_bdir)
                turn = 0

            is_over, my_score = self.agents_area(sim_board, my_pos, adv_pos)

        if loses_in_one and my_score == -10000:
            return -10000000

        return my_score

    #[Finished] Sets a barrier on the chess board
    def set_barrier(self, chess_board, r, c, dir):
        # Set the barrier to True
        chess_board[r, c, dir] = True
        # Set the opposite barrier to True
        move = self.moves[dir]
        chess_board[r + move[0], c + move[1], self.opposites[dir]] = True

    # [Finished, taken from random_agent.py] 
    def random_move(self, chess_board, my_pos, adv_pos, max_step):
        
        # Moves (Up, Right, Down, Left)
        ori_pos = deepcopy(my_pos)
        steps = np.random.randint(0, max_step + 1)
       
        # Random Walk
        for _ in range(steps):
            r, c = my_pos
            dir = np.random.randint(0, 4)
            m_r, m_c = self.moves[dir]
            my_pos = (r + m_r, c + m_c)

            # Special Case enclosed by Adversary
            k = 0
            while chess_board[r, c, dir] or my_pos == adv_pos:
                k += 1
                if k > 300:
                    break
                dir = np.random.randint(0, 4)
                m_r, m_c = self.moves[dir]
                my_pos = (r + m_r, c + m_c)

            if k > 300:
                my_pos = ori_pos
                break

        # Put Barrier
        dir = np.random.randint(0, 4)
        r, c = my_pos
        while chess_board[r, c, dir]:
            dir = np.random.randint(0, 4)

        return my_pos, dir

    #[Finished, based off world.py check_endgame] 
    # [Returns] IsEndGame:Boolean, p0score
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

        p0_r = find(my_pos)
        p1_r = find(adv_pos)
       
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)
        if p0_r == p1_r:

            return False, 0
        player_win = None
        win_blocks = -1
        if p0_score > p1_score:
            player_win = 0
            win_blocks = p0_score
        elif p0_score < p1_score:
            player_win = 1
            win_blocks = p1_score
        else:
            player_win = -1  # Tie
            return True, -25
       

        if (p0_score - p1_score) < 0:
             return True, -10000
        else:
            return True, 100

    # [Function] Calculate and return all possible moves and barrier placements
    # [Returns] validMoves := Queue<((r, c, dir))>
    def get_all_simple_steps(self, chess_board, my_pos, adv_pos, max_step):

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
    