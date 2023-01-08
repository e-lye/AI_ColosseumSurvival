# Student agent: Add your own agent here
from random import shuffle
from agents.agent import Agent
from store import register_agent
from copy import deepcopy
import numpy as np
import time
import traceback

class MinmaxAgent:
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
        self.zoneTiles = []
        self.start_time = 0
        self.time_limit = 1.95
        self.panic_move = None
        self.autoplay = True
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        # Opposite Directions
        self.opposites = {0: 2, 1: 3, 2: 0, 3: 1}

    def step(self, chess_board, my_pos, adv_pos, max_step):

        self.start_time = time.time()
        
        # find a cheap panic move in case we lack time (one of the 4 directions will always be a valid move, otherwise the game would be done).
        r, c = my_pos
        for bdir in range(4):
            if not chess_board[r, c, bdir]:
                self.panic_move = tuple(((r,c), bdir))
                break
        
        next_move = self.find_next_step(chess_board, my_pos, adv_pos, max_step)
        return next_move

    # [Finished] Gets the next best step
    # [Returns] NextMove: ((r, c), dir)
    def find_next_step(self, chess_board, my_pos, adv_pos, max_step):
        try:

            sortedMoves = self.sort_by_best(chess_board, my_pos, adv_pos, max_step)
            
            r, c, bdir = sortedMoves[0]
            return tuple(((r, c), bdir))
        except Exception as e:
            traceback.print_exc()
            return self.panic_move

    # [Finished] Sections/sorts the given list, putting our winning moves at the front
    # [Returns] (WinMoves + otherMoves): List<r,c,dir>
    def sort_by_best(self, chess_board, my_pos, adv_pos, max_step):
        
        win_moves = []
        tie_moves = []
        lose_moves = []
        no_end_moves = []

        def eval_move(move):

            r, c, bdir = move
            cur_board = deepcopy(chess_board)
            self.set_barrier(cur_board, r, c, bdir)
           
            rep = None

            if self.could_potentially_finish(chess_board, move):

                rep = self.agents_area(cur_board, (r,c), adv_pos)
                isEnd, score = rep

            if rep == None:
                no_end_moves.append(move)
            else:
                if isEnd:
                    # Take this option since we are winning
                    if score > 0:
                        # just immediately take the winning move
                        return True, [move]
                    elif score == 0:
                        tie_moves.append(move)
                    else: 
                        lose_moves.append(move)
                else:
                    no_end_moves.append(move)
            
            return False, None

        output = self.execute_on_all_valid_steps(chess_board, my_pos, adv_pos, max_step, eval_move)

        if output != None:
            return output

        if win_moves:
            return win_moves
        
        # sort losing moves in ascending order (smallest difference first)
        lose_moves.sort(key=lambda y: -y[1])
        
        no_end_moves, tie_is_better = self.sort_by_safest(chess_board, adv_pos, max_step, no_end_moves)

        if tie_is_better:
            return tie_moves + no_end_moves + lose_moves
        else:
            return no_end_moves + tie_moves + lose_moves

    # [Finished] Sections/sorts the given list into safe moves and then bad moves
    # [Returns] (SafeMoves + BadMoves): List<r,c,dir>, tieIsBetter: Boolean
    def sort_by_safest(self, chess_board, adv_pos, max_step, moves: list):
        
        bad_moves = []
        safe_moves = []
        
        for move in moves:
            r, c, bdir = move
            cur_board = deepcopy(chess_board)
            self.set_barrier(cur_board, r, c, bdir)

            # set our current position to the move
            cur_pos = tuple((r, c))

            # Sees if the adversary can win on this turn
            advCanWin, nb_turn = self.has_winning_moves(chess_board, adv_pos, cur_pos, max_step)

            if (advCanWin and nb_turn == 0):
                bad_moves.append(move)
            else:
                # Change our panic move to a non-dangerous move
                self.panic_move = tuple(((r,c),bdir))
                if nb_turn == 1:
                    safe_moves.append(move)
                else:
                    return [move], False
        
        return safe_moves + bad_moves, True

    # [Finished] Checks if our agent has winning moves (instant or force adv to suicide)
    # [Returns] hasWinningMoves: Boolean, turn: int
    def has_winning_moves(self, chess_board, my_pos, adv_pos, max_step):

        N = chess_board.shape[0]

        # [Returns] should_return: boolean, (could_win: boolean, turn: int)
        def could_win(move):
            r, c, bdir = move
            cur_board = deepcopy(chess_board)
            self.set_barrier(cur_board, r, c, bdir)

            # If theres no moves that can unite 2 barriers, return false
            if not self.could_potentially_finish(chess_board, move):
                return False, (False, 0)
            
            isEnded, agentScore = self.agents_area(cur_board, (r, c), adv_pos)

            if isEnded:
                if agentScore > 0:
                    return True, (True, 0)
                else:
                    # game is Over, and we lose
                    return False, (False, 0)
            
            advWontLose = self.does_not_lose(chess_board, adv_pos, (r, c), max_step)
            # If the adv could lose
            if not advWontLose:
                return True, (True, 0)
            else:
                return False, (False, 0)
        
        output = self.execute_on_all_valid_steps(chess_board, my_pos, adv_pos, max_step, could_win)

        if output == None:
            return False, 0
        else:
            return output

    # [Finished] Checks if the current position for our agent could lose
    # [Returns] doesntLose: Boolean
    def does_not_lose(self, chess_board, my_pos, adv_pos, max_step):

        N = chess_board.shape[0]
        # [Returns] (should_return: Boolean, 
        def wont_lose(move):

            r, c, bdir = move
            potential_board = deepcopy(chess_board)
            self.set_barrier(potential_board, r, c, bdir)

            rep = None

            # Checks if the move finishes and what our scores would be if it does
            if self.could_potentially_finish(chess_board, move):
                rep = self.agents_area(potential_board, (r,c), adv_pos)
                isEnded, agentScore = rep

                if (rep == None or not isEnded or agentScore >= 0):
                    # if we win, won't end, or at least tie, we won't lose.
                    return True, True

            return False, False

        output = self.execute_on_all_valid_steps(chess_board, my_pos, adv_pos, max_step, wont_lose)

        if output == None:
            return False
        else: 
            return output
    
    # [Finished] Applies Function (func) to all valid moves
    # func should return (should_return: boolean, FunctionValue)
    # if the func returns should_return, we return after function has been run on that move (ignores rest of valid)
    def execute_on_all_valid_steps(self, chess_board, my_pos, adv_pos, max_step, func):

        r_value = None
        valid_moves = self.get_all_valid_steps(chess_board, my_pos, adv_pos, max_step)
        for move in valid_moves:
            should_return, r_value = func(move)
            if should_return:
                return r_value

        return r_value

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

    # [Finished] Checks if the move unites 2 barriers together.
    # [Returns] isPotentialFinisher: Boolean
    def could_potentially_finish(self, chess_board, move):
        r, c, bdir = move
        N = chess_board.shape[0]
        # If the barrier is UP or DOWN
        if (bdir % 2 == 0):
            h_l = (c == 0 or chess_board[r, c-1, bdir])
            h_r = (c == N-1 or chess_board[r, c+1, bdir])
            # If the barrier is UP
            if bdir == 0: 
                v_l = chess_board[r, c, 3] or chess_board[r - 1, c, 3]
                v_r = chess_board[r, c, 1] or chess_board[r - 1, c, 1]
            # If the barrier is DOWN
            else: 
                v_l = chess_board[r, c, 3] or chess_board[r + 1, c, 3]
                v_r = chess_board[r, c, 1] or chess_board[r + 1, c, 1]
                
            return (v_l or h_l) and (v_r or h_r)

        # If the move is LEFT OR RIGHT
        else:
            v_u = (r == 0 or chess_board[r - 1, c, bdir])
            v_d = (r == N-1 or chess_board[r + 1, c, bdir])
            if bdir == 1: # right
                h_u = chess_board[r, c, 0] or chess_board[r, c + 1, 0]
                h_d = chess_board[r, c, 2] or chess_board[r, c + 1, 2]
            else: # left
                h_u = chess_board[r, c, 0] or chess_board[r, c - 1, 0]
                h_d = chess_board[r, c, 2] or chess_board[r, c - 1, 2]

            return (h_u or v_u) and (h_d or v_d) 

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
            self.check_time()
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
        if time.time() - self.start_time > self.time_limit:
            #print("Time breach")
            raise TimeUpException

class TimeUpException(Exception):
    pass
