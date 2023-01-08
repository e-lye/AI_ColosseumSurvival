# Student agent: Add your own agent here
import time
import numpy as np
from copy import deepcopy
from agents.agent import Agent
from store import register_agent

import sys

@register_agent("student_agent5")
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
        self.moves_table = {} # transposition table to cache moves to its evaluation value
        self.time_limit = 1.993 # create some buffer time to 2s 
        self.curr_best = None # the current best move based on the search
        self.autoplay = True
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        # Opposite Directions
        self.opposites = {0: 2, 1: 3, 2: 0, 3: 1}

    def step(self, chess_board, my_pos, adv_pos, max_step):
        # first step
        if not self.initialized:
            self.time_limit = 1.993 # can give more time for first step? Not really necessary
            self.board_size = chess_board.shape[0]
            self.initialized = True
        else:
            self.time_limit = 1.993

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
                # if(search_depth == 2):
                #     print(time.time() - self.start_time)
                #     return ((r, c), b_dir)
                search_depth = search_depth+1
                if stop_deliberate:
                    # found winning move
                    return ((r, c), b_dir)
                else:
                    # record the current best move 
                    self.curr_best = tuple(((r,c), b_dir))
        except TimeUpException as e:
            self.moves_table.clear()
            return self.curr_best

    def alphabeta (self, chess_board, my_pos, adv_pos, max_step, max_depth):
        """Heuristic minimax search with cut-off depth and alpha beta pruning.
        to stop the iterative deepening when it already determined a winning move or
        guaranteed to suffered a lost.
        A transposition table (moves_table) is used to optimize the search time.

        Parameters
        ----------
        chess_board, my_pos, adv_pos, max_step: np.array, (int, int), (int, int), int
            Parameters that define the state of the game.
        max_depth
            the cut-off depth limit of the search from the root node (input game state). 

        Returns
        --------
        (move:(r, c, dir), stop_deliberating: Boolean) 
            Move is the optimal move from the search, stop_deliberating is a variable
            to stop the iterative deepening when it already determined a winning move or
            guaranteed to suffered a lost. (Deeper search would return the same move).

        Raises
        -------
        TimeUpException (through check_time())
            Raised if search time reaches the allocated time (1.992s).
        """
        
        def terminal_test(chess_board, my_pos, adv_pos, max_step, curr_depth):
            """A test to determine if the input game state is a terminal state (leaf 
            node of a game tree). A transposition table (moves_table) is used as a 
            hash table save evaluation time for seen game states.

            Parameters
            ----------
            chess_board, my_pos, adv_pos, max_step: np.array, (int, int), (int, int), int
                Parameters that define the state of the game.
            curr_depth
                the current depth of the search from the root node (input game state of 
                alphabeta).

            Returns
            --------
            eval: int 
                Eval is the evaluation of the input game state. It can either be a heuristic
                evaluation when cut-off occurs or an evaluation of an end game state (infinity
                when winning and negative infinity when losing). 
            """
            eval = 0

            if (chess_board.data.tobytes(), my_pos, adv_pos) in self.moves_table:
                # cache hit in transposition table
                eval = self.moves_table.get((chess_board.tostring(), my_pos, adv_pos))
                # return only if the input state is a terminal or cutoff state
                if eval >= 1000 or eval <= -1000:
                    return eval
                elif curr_depth == max_depth:
                    return eval

            # check end game
            rep = self.agents_area(chess_board, my_pos, adv_pos)
            isEnd, score = rep

            if isEnd:
                if score > 0:
                    # winning so utility is infinity
                    eval = score * 1000
                    self.moves_table[(chess_board.data.tobytes(), my_pos, adv_pos)] = eval
                    return eval
                elif score < 0:
                    # losing so utility is -infinity
                    eval = score * 1000
                    self.moves_table[(chess_board.data.tobytes(), my_pos, adv_pos)] = eval
                    return eval
                else:
                    # draw so utility is the heuristic evaluation
                    eval = self.eval_func(chess_board, my_pos, adv_pos, max_step)
                    self.moves_table[(chess_board.data.tobytes(), my_pos, adv_pos)] = eval
                    return eval

            if curr_depth == max_depth:
                # At cut-off state use heuristic evaluation as utility
                eval = self.eval_func(chess_board, my_pos, adv_pos, max_step)
                self.moves_table[(chess_board.data.tobytes(), my_pos, adv_pos)] = eval
                return eval

            # not terminal
            return None
        
        def max (chess_board, my_pos, adv_pos, max_step, alpha, beta, curr_depth):
            """A function to determine the utility value of the terminal / cutoff states 
            from the point of view of the max player (our agent).

            Parameters
            ----------
            chess_board, my_pos, adv_pos, max_step: np.array, (int, int), (int, int), int
                Parameters that define the state of the game.
            alpha
                The value of the optimal choice for the max player so far during the search
                a.k.a. lower bound of the highest value of max 
            beta
                The value of the optimal choice for the min player so far during the search
                a.k.a. upper bound of the lowest value of min
            curr_depth
                the current depth of the search from the root node (input game state of 
                alphabeta).
            
            Returns
            --------
            (eval: int, mv: (r, c, dir)) 
                eval is the utility value of the terminal / cutoff states from the point of 
                view of the max player, it is the maximum of the utilities of its child nodes
                mv is the move that generates the child node with maximum utility
            """
            self.check_time()
            eval = terminal_test(chess_board, my_pos, adv_pos, max_step, curr_depth)
            if eval is not None:
                # terminal leaf node
                return (eval, None)
            else:
                # initialize utility and alpha for update
                eval = float('-inf')
                new_alpha = alpha
                
                valid_moves = self.get_all_valid_steps(chess_board, my_pos, adv_pos, max_step)
                move_to_shallow_eval = {}
                # maps valid moves to the heuristic value of the game state they generate, for
                # reordering moves 
            
                move_to_board = {}  
                # maps valid moves to the game state they generate, saves from computing twice

                for move in valid_moves:
                    r, c, b_dir = move

                    # generate corresponding game state
                    eval_board = deepcopy(chess_board)
                    self.set_barrier(eval_board, r, c, b_dir)

                    move_to_board[move] = eval_board
                    if (eval_board.data.tobytes(), (r, c), adv_pos) in self.moves_table:
                        # game state in cache
                        # print("max cache hit")
                        move_to_shallow_eval[move] = self.moves_table[(eval_board.data.tobytes(), (r, c), adv_pos)]

                # order the valid moves based on decreasing order of its evaluation if they are in the transposition table   
                ordered_moves = sorted([mv for mv in valid_moves if mv in move_to_shallow_eval], key=lambda x: move_to_shallow_eval[x], reverse=True) \
                    + [mv for mv in valid_moves if mv not in move_to_shallow_eval]

                # print_moves = [(move, move_to_shallow_eval[move]) for move in ordered_moves if move in move_to_shallow_eval]
                # print(print_moves)

                move_to_deep_eval = {}
                # maps the move to its end game / cutoff utility

                for move in ordered_moves:
                    r, c, b_dir = move

                    cur_board = move_to_board[move] 
                    # guaranteed to exist in the dictionary 
                    
                    min_value, _ = min(cur_board, adv_pos, (r, c), max_step, new_alpha, beta, curr_depth+1)
                    move_to_deep_eval[move] = min_value

                    if min_value > eval:
                        # take higher utility (optimal for max player)
                        eval = min_value
                    if eval >= beta:
                        # beta cut-off
                        return (eval, move)
                    # update alpha if the current highest utility is greater than alpha
                    new_alpha = eval if eval > alpha else new_alpha

                for mv, ev in move_to_deep_eval.items():
                    # find the first move corresponding to the min utility
                    if ev == eval:
                        return (eval, mv)
        
        def min (chess_board, my_pos, adv_pos, max_step, alpha, beta, curr_depth):
            """A function to determine the utility value of the terminal / cutoff states 
            from the point of view of the min player (our opponent).

            Parameters
            ----------
            chess_board, my_pos, adv_pos, max_step: np.array, (int, int), (int, int), int
                Parameters that define the state of the game.
            alpha
                The value of the optimal choice for the max player so far during the search
                a.k.a. lower bound of the highest value of max 
            beta
                The value of the optimal choice for the min player so far during the search
                a.k.a. upper bound of the lowest value of min
            curr_depth
                the current depth of the search from the root node (input game state of 
                alphabeta).
            
            Returns
            --------
            (eval: int, mv: (r, c, dir)) 
                eval is the utility value of the terminal / cutoff states from the point of 
                view of the min player, it is the minimum of the utilities of its child nodes
                mv is the move that generates the child node with minimum utility
            """
            self.check_time()

            eval = terminal_test(chess_board, adv_pos, my_pos, max_step, curr_depth)
            # note: flip adv and my pos since we want the utility of the max player
            
            if eval is not None:
                # end game leaf node
                return (eval, None)
            else:
                # initialize utility and beta for update
                eval = float('inf')
                new_beta = beta
                
                valid_moves = self.get_all_valid_steps(chess_board, my_pos, adv_pos, max_step)
                move_to_shallow_eval = {} 
                # maps valid moves to the heuristic value of the game state they generate, for
                # reordering moves

                move_to_board = {}
                # maps valid moves to the game state they generate, saves from computing twice

                for move in valid_moves:
                    r, c, b_dir = move

                    # generate corresponding game state
                    eval_board = deepcopy(chess_board)
                    self.set_barrier(eval_board, r, c, b_dir)

                    move_to_board[move] = eval_board
                    
                    if (eval_board.data.tobytes(), adv_pos, (r, c)) in self.moves_table:
                        # game state in cache
                        # print("min cache hit")
                        move_to_shallow_eval[move] = self.moves_table[(eval_board.data.tobytes(), adv_pos, (r, c))]
                        
                # order the valid moves based on increasing order of its evaluation if they are in the transposition table
                ordered_moves = sorted([mv for mv in valid_moves if mv in move_to_shallow_eval], key=lambda x: move_to_shallow_eval[x]) \
                    +[mv for mv in valid_moves if mv not in move_to_shallow_eval]

                for move in ordered_moves:
                    r, c, b_dir = move

                    cur_board = move_to_board[move] 
                    # guaranteed to exist in the dictionary 
                    
                    max_value, mv = max(cur_board, adv_pos, (r, c), max_step, alpha, new_beta, curr_depth+1)

                    if max_value < eval:
                        # take lower utility (optimal for min player)
                        eval = max_value
                    if eval <= alpha:
                        # alpha cut-off
                        return (eval, move)
                    # update beta if the current lowest utility is less than beta
                    new_beta = eval if eval < new_beta else new_beta

                return(eval, None)

        # search from max player's (our) turn with depth 0        
        value, move = max(chess_board, my_pos, adv_pos, max_step, float('-inf'), float('inf'), 0)

        stop_deliberating = False

        if value >= 1000:
            # check mate stop thinking
            print("I win")
            stop_deliberating = True
        if value <= -1000:
            # losing stop thinking
            stop_deliberating = True
            # don't want to give up in case of weak opponent 
            # execute previous best move
            (r,c), bdir = self.curr_best
            return (r,c, bdir), stop_deliberating
        
        return (move, stop_deliberating)

    def eval_func(self, chess_board, my_pos, adv_pos, max_step):
        """Heuristic evaluation function determined through trial and error

        Parameters
        ----------
        chess_board, my_pos, adv_pos, max_step: np.array, (int, int), (int, int), int
            Parameters that define the state of the game.

        Returns
        --------
        eval: float
            A heuristic evaluation of a non-terminal game state, weighted sum of
            our agents reachable tiles minus that of the opponents 
        """
        # print(1 - self.get_manhattan_dist(my_pos, adv_pos) / (self.board_size * 2))
        # print (self.get_number_of_reachable_tiles(chess_board, my_pos, adv_pos, max_step))
        return 0.4 * self.get_number_of_reachable_tiles(chess_board, my_pos, adv_pos, max_step) \
               + 0.6 * (- self.get_number_of_reachable_tiles(chess_board, adv_pos, my_pos, max_step))
            #    + (self.board_size**2 - area)
            #    + 0.1 * self.get_number_of_adj_barriers(chess_board, adv_pos) \
            #    - 0.1 * self.get_number_of_adj_barriers(chess_board, my_pos)
        # return 1 - self.get_manhattan_dist(my_pos, adv_pos) / (self.board_size ** 2) \
        # return 0

    def get_all_valid_steps(self, chess_board, my_pos, adv_pos, max_step):
        """Calculate and return all possible moves and barrier placements

        Parameters
        ----------
        chess_board, my_pos, adv_pos, max_step: np.array, (int, int), (int, int), int
            Parameters that define the state of the game.

        Returns
        --------
        validMoves := Queue<((r, c, dir))>
            All possible moves and barrier placements
        """

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
    
    # def get_random_walk_move(self, chess_board, my_pos, adv_pos, max_step):
    #     # Moves (Up, Right, Down, Left)
    #     ori_pos = deepcopy(my_pos)
    #     moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
    #     steps = np.random.randint(0, max_step + 1)

    #     # Random Walk
    #     for _ in reversed(range(steps)):
    #         r, c = my_pos
    #         dir = np.random.randint(0, 4)
    #         m_r, m_c = moves[dir]
    #         my_pos = (r + m_r, c + m_c)

    #         # Special Case enclosed by Adversary
    #         k = 0
    #         while chess_board[r, c, dir] or my_pos == adv_pos:
    #             k += 1
    #             if k > 300:
    #                 break
    #             dir = np.random.randint(0, 4)
    #             m_r, m_c = moves[dir]
    #             my_pos = (r + m_r, c + m_c)

    #         if k > 300:
    #             my_pos = ori_pos
    #             break

    #     # Put Barrier
    #     dir = np.random.randint(0, 4)
    #     r, c = my_pos
    #     while chess_board[r, c, dir]:
    #         dir = np.random.randint(0, 4)

    #     return r, c, dir

class TimeUpException(Exception):
    pass
