import logic
import random
from AbstractPlayers import *
import time
import math
import copy

# commands to use for move players. dictionary : Move(enum) -> function(board),
# all the functions {up,down,left,right) receive board as parameter and return tuple of (new_board, done, score).
# new_board is according to the step taken, done is true if the step is legal, score is the sum of all numbers that
# combined in this step.
# (you can see GreedyMovePlayer implementation for example)
commands = {Move.UP: logic.up, Move.DOWN: logic.down,
            Move.LEFT: logic.left, Move.RIGHT: logic.right}


# generate value between {2,4} with probability p for 4
def gen_value(p=PROBABILITY):
    return logic.gen_two_or_four(p)


class GreedyMovePlayer(AbstractMovePlayer):
    """Greedy move player provided to you (no need to change),
    the player receives time limit for a single step and the board as parameter and return the next move that gives
    the best score by looking one step ahead.
    """
    def get_move(self, board, time_limit) -> Move:
        optional_moves_score = {}
        for move in Move:
            new_board, done, score = commands[move](board)
            if done:
                optional_moves_score[move] = score
        return max(optional_moves_score, key=optional_moves_score.get)


class RandomIndexPlayer(AbstractIndexPlayer):
    """Random index player provided to you (no need to change),
    the player receives time limit for a single step and the board as parameter and return the next indices to
    put 2 randomly.
    """
    def get_indices(self, board, value, time_limit) -> (int, int):
        a = random.randint(0, len(board) - 1)
        b = random.randint(0, len(board) - 1)
        while board[a][b] != 0:
            a = random.randint(0, len(board) - 1)
            b = random.randint(0, len(board) - 1)
        return a, b


# part A
class ImprovedGreedyMovePlayer(AbstractMovePlayer):
    """Improved greedy Move Player,
    implement get_move function with greedy move that looks only one step ahead with heuristic.
    (you can add helper functions as you want).
    """
    def __init__(self):
        AbstractMovePlayer.__init__(self)

    def log2(self, number):
        counter = 0
        n = number
        while int(n/2) != 0:
            counter = counter+1
            n = n/2
        return counter

    def monotonus(self,board):
        ascending_up_to_down = 0
        descending_up_to_down = 0
        ascending_left_to_right = 0
        descending_left_to_right = 0
        for row in range(4):
            for col in range(4):
                if col + 1 < 4:
                    if board[row][col] > board[row][col + 1]:
                        descending_left_to_right -= board[row][col] - board[row][col + 1]
                    else:
                        ascending_left_to_right += board[row][col] - board[row][col + 1]
                if row + 1 < 4:
                    if board[row][col] > board[row + 1][col]:
                        descending_up_to_down -= board[row][col] - board[row + 1][col]
                    else:
                        ascending_up_to_down += board[row][col] - board[row + 1][col]
        return max(descending_left_to_right, ascending_left_to_right) + max(descending_up_to_down, ascending_up_to_down)


    def smoothness(self,board):
        counter = 0
        for row in range(4):
            for col in range(4):
                if board[row][col] != 0:
                    i = 1
                    while col+i < 4:
                        if board[row][col + i] == 0:
                            i += 1
                            continue
                        if board[row][col+i] == board[row][col]:
                            counter += board[row][col]
                        break

                    i = 1
                    while row+i < 4:
                        if board[row + i][col] == 0:
                            i += 1
                            continue
                        if board[row+i][col] == board[row][col]:
                            counter += board[row][col]
                        break
        return counter


    def calcCellScore(self,value):
        if value <=2:
            return 0
        return 2*self.calcCellScore(value/2) + 2**self.log2(value)

    def score(self, board):
        cells_sum = 0
        for row in range(4):
            for col in range(4):
                 cells_sum += self.calcCellScore(board[row][col])
        return cells_sum

    def maxCellValue(self, board):
        copy_board = board
        max_value = 0
        for row in range(4):
            for col in range(4):
                if copy_board[row][col] > max_value:
                    max_value = copy_board[row][col]
        return max_value

    def hotCorners(self, board):
        max_value = self.maxCellValue(board)
        grade = 0
        for row in range(4):
            for col in range(4):
                if board[row][col] == max_value:
                    if (row == 0 and col == 0) or (row == 3 and col == 3) or (row == 0 and col == 3) or (row == 3 and col == 0):
                        grade += 2*(max_value)
                        continue
                    if (row == 1 and col == 1) or (row == 2 and col == 2) or (row == 1 and col == 2) or (row == 2 and col == 1):
                        grade -= (max_value)
                        continue
                    grade -= (max_value)

        return grade

    def emptyCells(self,board):
        counter = 0
        for i in range(4):
            for j in range(4):
                if board[i][j] == 0:
                    counter += 1
        return counter

    def calculateNewHScore(self, board):
        greedy_score = self.score(board)
        max_value_cell = self.maxCellValue(board)
        empty_value = self.emptyCells(board)
        smoothness_score = self.smoothness(board)
        monoton_score = self.monotonus(board)
        hot_corners = self.hotCorners(board)

        score_fac = 0.2
        max_value_cell_fac = 0.2
        empty_value_fac = 1
        hot_corners_fac = 0.2
        smoothness_score_fac = 0.2
        monoton_score_fac = 0.2

        return float(greedy_score)*score_fac\
               + max_value_cell*max_value_cell_fac\
               + empty_value*empty_value_fac\
               + hot_corners*hot_corners_fac\
               + smoothness_score*smoothness_score_fac\
               + monoton_score*monoton_score_fac

# ---------------- GET MOVE USING NEW HEURISTICS ---------------

    def get_move(self, board, time_limit) -> Move:
        optional_moves_score = {}
        for move in Move:
            new_board, done, score = commands[move](board)
            if done:
                optional_moves_score[move] = self.calculateNewHScore(new_board)
        return max(optional_moves_score, key=optional_moves_score.get)



# part B

class MiniMaxMovePlayer(AbstractMovePlayer):
    """MiniMax Move Player,
    implement get_move function according to MiniMax algorithm
    (you can add helper functions as you want).
    """
    def __init__(self):
        AbstractMovePlayer.__init__(self)
        self.active_player = 'MINIPLAYER'  # get move starts with Max Player
        # TODO: add here if needed

    def change_player(self, player):
        self.active_player = player

# WE CAN CALCULATE TIME FOR 1 BOARD SUCCESSORS INITIALIZATION AND THEN FOR TIME LIMIT
# MULTIPLY BY 2^DEPTH

    def calcCellScore(self,value):
        if value <=2:
            return 0
        return 2*self.calcCellScore(value/2) + value

    def score(self, board):
        cells_sum = 0
        for row in range(4):
            for col in range(4):
                 cells_sum += self.calcCellScore(board[row][col])
        return cells_sum

    # GET THE NEXT SUCCESSORS OF A CERTAIN BOARD
    def generate_optional_boards(self, board, time_limit):
        start_time = time.time()
        boards = list()
        for i in range(4):
            for j in range(4):
                if time_limit -(time.time()-start_time) <= 0.05:
                    raise TimeoutError()
                if board[i][j] == 0:
                    new_board = copy.deepcopy(board)
                    new_board[i][j] = 2
                    boards.append(new_board)
        return boards

# score move by making greedy on all optional moves
    def score_move(self, boards):
        total_score_for_boards = 0
        for board in boards:
            total_score_for_boards = total_score_for_boards + self.score(board)
        return total_score_for_boards

# get key by value for dict
    def get_key(self, dict, val):
        for key, value in dict.items():
            if val == value:
                return key
        return None

    def minimax(self, board, depth, time_limit):
        start_time = time.time()
        if time_limit <= 0.05:
            raise TimeoutError()
        if depth == 0:
            return self.score(board)
        # best_value = 0 # Dont know if to initialize because return value
        if self.active_player == 'MAXIPLAYER':
            best_value = -math.inf
            for move in Move:
                new_board, done, score = commands[move](board)
                if done:
                    self.change_player('MINIPLAYER')
                    val = self.minimax(new_board, depth - 1,time_limit - (time.time()-start_time))
                    best_value = max(best_value, val)
            return best_value

        else:  # self.active_player == 'MINIPLAYER':
            best_value = math.inf
            optional_boards = self.generate_optional_boards(board, time_limit - (time.time()-start_time))
            for optional_board in optional_boards:
                self.change_player('MAXIPLAYER')
                val = self.minimax(optional_board, depth - 1, time_limit - (time.time()-start_time))
                best_value = min(best_value, val)
            return best_value

    def get_move(self, board, time_limit) -> Move:
        depth = 0
        time_left = time_limit
        ret_value = 0
        #  while time_left > (2**(depth-1))*prev_time and time_left > 0.02:
        while True:
            try:
                start_time = time.time()
                optional_moves_score = {}
                for move in Move:
                    new_board, done, score = commands[move](board)
                    if done:
                        optional_moves_score[move] = self.minimax(new_board, depth, time_left - (time.time() - start_time))
            except TimeoutError:
                break
            ret_value = max(optional_moves_score, key=optional_moves_score.get)
            depth += 1
            end_time = time.time()
            prev_time = float(end_time - start_time)
            time_left = float(time_left - prev_time)
        #  print("depth " + str(depth))
        return ret_value

    # the function flow: on maxiplayer -> recursion on max total boards scores per move
    #                    on miniplayer -> recursion on min total boards scores per move


class MiniMaxIndexPlayer(AbstractIndexPlayer):
    """MiniMax Index Player,
    this player is the opponent of the move player and need to return the indices on the board where to put 2.
    the goal of the player is to reduce move player score.
    implement get_indices function according to MiniMax algorithm, the value in minimax player value is only 2.
    (you can add helper functions as you want).
    """

    ## for board we should greedy calculate the moves score and do the same
    def __init__(self):
        AbstractIndexPlayer.__init__(self)
        self.active_player = 'MAXIPLAYER'

    def change_player(self, player):
        self.active_player = player

    def calcCellScore(self, value):
        if value <=2:
            return 0
        return 2*self.calcCellScore(value/2) + value

    def score(self, board):
        cells_sum = 0
        for row in range(4):
            for col in range(4):
                 cells_sum += self.calcCellScore(board[row][col])
        return cells_sum

    def generate_optional_boards(self, board, time_limit):
        start_time = time.time()
        boards = list()
        for i in range(4):
            for j in range(4):
                if time_limit - (time.time() - start_time) <= 0.05:
                    raise TimeoutError()
                if board[i][j] == 0:
                    new_board = copy.deepcopy(board)
                    new_board[i][j] = 2
                    boards.append(new_board)
        return boards

    def minimax(self, board, depth, time_limit):
        start_time = time.time()
        if time_limit <= 0.05:
            raise TimeoutError()
        if depth == 0:
            return self.score(board)
        # best_value = 0 # Dont know if to initialize because return value
        if self.active_player == 'MAXIPLAYER':
            best_value = -math.inf
            for move in Move:
                new_board, done, score = commands[move](board)
                if done:
                    self.change_player('MINIPLAYER')
                    val = self.minimax(new_board, depth - 1, time_limit - (time.time()-start_time))
                    best_value = max(best_value, val)
            return best_value

        else:  # self.active_player == 'MINIPLAYER':
            best_value = math.inf
            optional_boards = self.generate_optional_boards(board, time_limit - (time.time()-start_time))
            for optional_board in optional_boards:
                self.change_player('MAXIPLAYER')
                val = self.minimax(optional_board, depth - 1, time_limit - (time.time()-start_time))
                best_value = min(best_value, val)
            return best_value

    def get_indices(self, board, value, time_limit) -> (int, int):
        time_left = time_limit
        depth = 0
        ret_value = (0, 0)
        #  while time_left > (2**(depth-1))*prev_time and time_left > 0.02:
        while True:
            try:
                start_time = time.time()
                index = 0
                dict_board_score = {}
                optional_boards = self.generate_optional_boards(board, time_left - (time.time()-start_time))
                for optional_board in optional_boards:
                    dict_board_score[index] = self.minimax(optional_board, depth, time_left - (time.time() - start_time))
                    index += 1
                found_min_board = min(dict_board_score, key=dict_board_score.get)
                board_to_compare = optional_boards[found_min_board]
                for i in range(4):
                    for j in range(4):
                        if board_to_compare[i][j] != board[i][j]:
                            ret_value = (i, j)
            except TimeoutError:
                break
            depth += 1
            end_time = time.time()
            prev_time = end_time - start_time
            time_left = time_left - prev_time
        return ret_value




# part C
class ABMovePlayer(AbstractMovePlayer):
    """Alpha Beta Move Player,
    implement get_move function according to Alpha Beta MiniMax algorithm
    (you can add helper functions as you want)
    """


    def __init__(self):
        AbstractMovePlayer.__init__(self)
        self.active_player = 'MINIPLAYER'  # get move starts with Max Player

    def change_player(self, player):
        self.active_player = player

# WE CAN CALCULATE TIME FOR 1 BOARD SUCCESSORS INITIALIZATION AND THEN FOR TIME LIMIT
# MULTIPLY BY 2^DEPTH

    def calcCellScore(self,value):
        if value <=2:
            return 0
        return 2*self.calcCellScore(value/2) + value

    def score(self, board):
        cells_sum = 0
        for row in range(4):
            for col in range(4):
                 cells_sum += self.calcCellScore(board[row][col])
        return cells_sum

    # GET THE NEXT SUCCESSORS OF A CERTAIN BOARD
    def generate_optional_boards(self, board, time_limit):
        start_time = time.time()
        boards = list()
        for i in range(4):
            for j in range(4):
                if time_limit - (time.time() - start_time) <= 0.05:
                    raise TimeoutError()
                if board[i][j] == 0:
                    new_board = copy.deepcopy(board)
                    new_board[i][j] = 2
                    boards.append(new_board)
        return boards

# score move by making greedy on all optional moves
    def score_move(self, boards):
        total_score_for_boards = 0
        for board in boards:
            total_score_for_boards = total_score_for_boards + self.score(board)
        return total_score_for_boards

# get key by value for dict
    def get_key(self, dict, val):
        for key, value in dict.items():
            if val == value:
                return key
        return None

    def minimax(self, board, depth, alpha, beta, time_limit):
        start_time = time.time()
        if time_limit <= 0.05:
            raise TimeoutError()
        if depth == 0:
            return self.score(board)
        # best_value = 0 # Dont know if to initialize because return value
        if self.active_player == 'MAXIPLAYER':
            best_value = -math.inf
            for move in Move:
                new_board, done, score = commands[move](board)
                if done:
                    self.change_player('MINIPLAYER')
                    val = self.minimax(new_board, depth - 1, alpha, beta, time_limit - (time.time()-start_time))
                    best_value = max(best_value, val)
                    alpha = max(best_value,alpha)
                    if best_value >= beta:
                        return math.inf
            return best_value

        else:  # self.active_player == 'MINIPLAYER':
            best_value = math.inf
            optional_boards = self.generate_optional_boards(board, time_limit - (time.time()-start_time))
            for optional_board in optional_boards:
                self.change_player('MAXIPLAYER')
                val = self.minimax(optional_board, depth - 1, alpha, beta,time_limit - (time.time()-start_time))
                best_value = min(best_value, val)
                beta = max(best_value, beta)
                if best_value <= alpha:
                    return -math.inf
            return best_value



    def get_move(self, board, time_limit) -> Move:
        depth = 0
        alpha = -math.inf
        beta = math.inf
        time_left = time_limit
        ret_value = 0
        #  while time_left > (2**(depth-1))*prev_time and time_left > 0.02:
        while True:
            try:
                start_time = time.time()
                optional_moves_score = {}
                for move in Move:
                    new_board, done, score = commands[move](board)
                    if done:
                        optional_moves_score[move] = self.minimax(new_board, depth,alpha,beta, time_left - (time.time() - start_time))
                ret_value = max(optional_moves_score, key=optional_moves_score.get)
            except TimeoutError:
                break
            depth += 1
            end_time = time.time()
            prev_time = float(end_time - start_time)
            time_left = float(time_left - prev_time)
        #  print("depth "+str(depth))
        return ret_value

# part D
class ExpectimaxMovePlayer(AbstractMovePlayer):
    """Expectimax Move Player,
    implement get_move function according to Expectimax algorithm.
    (you can add helper functions as you want)
    """

    def __init__(self):
        AbstractMovePlayer.__init__(self)
        self.active_player = 'MINIPLAYER'  # get move starts with Max Player

    def change_player(self, player):
        self.active_player = player

    # WE CAN CALCULATE TIME FOR 1 BOARD SUCCESSORS INITIALIZATION AND THEN FOR TIME LIMIT
    # MULTIPLY BY 2^DEPTH

    def calcCellScore(self, value):
        if value <= 2:
            return 0
        return 2 * self.calcCellScore(value / 2) + value

    def score(self, board):
        cells_sum = 0
        for row in range(4):
            for col in range(4):
                cells_sum += self.calcCellScore(board[row][col])
        return cells_sum

    # GET THE NEXT SUCCESSORS OF A CERTAIN BOARD
    def generate_optional_boards(self, board, value, time_limit):
        start_time = time.time()
        boards = list()
        for i in range(4):
            for j in range(4):
                if time_limit - (time.time() - start_time) <= 0.05:
                    raise TimeoutError()
                if board[i][j] == 0:
                    new_board = copy.deepcopy(board)
                    new_board[i][j] = value
                    boards.append(new_board)
        return boards

    # score move by making greedy on all optional moves
    def score_move(self, boards):
        total_score_for_boards = 0
        for board in boards:
            total_score_for_boards = total_score_for_boards + self.score(board)
        return total_score_for_boards

    # get key by value for dict
    def get_key(self, dict, val):
        for key, value in dict.items():
            if val == value:
                return key
        return None

    def expectimax(self, board, depth, time_limit):
        start_time = time.time()
        if time_limit <= 0.05:
            raise TimeoutError()
        if depth == 0:
            return self.score(board)
        # best_value = 0 # Dont know if to initialize because return value
        if self.active_player == 'MAXIPLAYER':
            best_value = -math.inf
            for move in Move:
                new_board, done, score = commands[move](board)
                if done:
                    self.change_player('MINIPLAYER')
                    val = self.expectimax(new_board, depth - 1, time_limit - (time.time()-start_time))
                    best_value = max(best_value, val)
            return best_value

        else:  # self.active_player == 'MINIPLAYER':
            avg = 0
            optional_boards = self.generate_optional_boards(board,2, time_limit - (time.time()-start_time))
            for optional_board in optional_boards:
                self.change_player('MAXIPLAYER')
                val_2 = self.expectimax(optional_board, depth - 1, time_limit - (time.time()-start_time))
                avg += 0.9*val_2
            optional_boards = self.generate_optional_boards(board, 4, time_limit - (time.time()-start_time))
            for optional_board in optional_boards:
                self.change_player('MAXIPLAYER')
                val_4 = self.expectimax(optional_board, depth - 1, time_limit - (time.time()-start_time))
                avg += 0.1 * val_4
            return avg

    def get_move(self, board, time_limit) -> Move:
        time_left = time_limit
        depth = 0
        ret_value = 0
        #  while time_left > (2 ** (depth - 1)) * prev_time and time_left > 0.02:
        while True:
            try:
                start_time = time.time()
                optional_moves_score = {}
                for move in Move:
                    new_board, done, score = commands[move](board)
                    if done:
                        optional_moves_score[move] = self.expectimax(new_board, depth,  time_left - (time.time()-start_time))
                ret_value = max(optional_moves_score, key=optional_moves_score.get)
            except TimeoutError:
                break
            depth += 1
            end_time = time.time()
            prev_time = float(end_time - start_time)
            time_left = float(time_left - prev_time)
        return ret_value


class ExpectimaxIndexPlayer(AbstractIndexPlayer):
    """Expectimax Index Player
    implement get_indices function according to Expectimax algorithm, the value is number between {2,4}.
    (you can add helper functions as you want)
    """
    def __init__(self):
        AbstractIndexPlayer.__init__(self)
        self.active_player = 'MAXIPLAYER'


    def change_player(self, player):
        self.active_player = player

    def calcCellScore(self, value):
        if value <=2:
            return 0
        return 2*self.calcCellScore(value/2) + value

    def score(self, board):
        cells_sum = 0
        for row in range(4):
            for col in range(4):
                 cells_sum += self.calcCellScore(board[row][col])
        return cells_sum

    def generate_optional_boards(self, board, value, time_limit):
        start_time = time.time()
        boards = list()
        for i in range(4):
            for j in range(4):
                if time_limit - (time.time() - start_time) <= 0.05:
                    raise TimeoutError()
                if board[i][j] == 0:
                    new_board = copy.deepcopy(board)
                    new_board[i][j] = value
                    boards.append(new_board)
        return boards

    def expectimax(self, board, depth, p, time_limit):
        start_time = time.time()
        if time_limit <= 0.05:
            raise TimeoutError()
        if depth == 0:
            return self.score(board)
        # best_value = 0 # Dont know if to initialize because return value
        if self.active_player == 'MAXIPLAYER':
            best_value = -math.inf
            for move in Move:
                new_board, done, score = commands[move](board)
                if done:
                    self.change_player('MINIPLAYER')
                    val = self.expectimax(new_board, depth - 1,p, time_limit - (time.time()-start_time))
                    best_value = max(best_value, val)
            return best_value

        else:  # self.active_player == 'MINIPLAYER':
            optional_boards = self.generate_optional_boards(board,p, time_limit - (time.time()-start_time))
            avg = 0
            for optional_board in optional_boards:
                self.change_player('MAXIPLAYER')
                val = self.expectimax(optional_board, depth - 1,p, time_limit - (time.time()-start_time))
                avg += p*val
            return avg

    def get_indices(self, board, value, time_limit) -> (int, int):
        time_left = time_limit
        depth = 0
        ret_value = (0, 0)
        p = 0.1
        if value == 2:
            p = 0.9

        #while time_left > (2**(depth-1))*prev_time and time_left > 0.02:
        while True:
            try:
                start_time = time.time()
                index = 0
                dict_board_score = {}
                optional_boards = self.generate_optional_boards(board, value, time_left - (time.time()-start_time))
                for optional_board in optional_boards:
                    dict_board_score[index] = self.expectimax(optional_board, depth, p, time_left - (time.time()-start_time))
                    index += 1
                found_min_board = min(dict_board_score, key=dict_board_score.get)
                board_to_compare = optional_boards[found_min_board]
                for i in range(4):
                    for j in range(4):
                        if board_to_compare[i][j] != board[i][j]:
                            ret_value = (i, j)
            except TimeoutError:
                break
            depth += 1
            end_time = time.time()
            prev_time = end_time - start_time
            time_left = time_left - prev_time
            #  print('board'+str(board))
        return ret_value







# Tournament
class ContestMovePlayer(AbstractMovePlayer):
    """Contest Move Player,
    implement get_move function as you want to compete in the Tournament
    (you can add helper functions as you want)
    """
    """Expectimax Move Player,
    implement get_move function according to Expectimax algorithm.
    (you can add helper functions as you want)
    """

    def __init__(self):
        AbstractMovePlayer.__init__(self)
        self.active_player = 'MINIPLAYER'  # get move starts with Max Player

    def change_player(self, player):
        self.active_player = player

    # WE CAN CALCULATE TIME FOR 1 BOARD SUCCESSORS INITIALIZATION AND THEN FOR TIME LIMIT
    # MULTIPLY BY 2^DEPTH
    def hotCorners(self, board):
        max_value = self.maxCellValue(board)
        grade = 0
        for row in range(4):
            for col in range(4):
                if board[row][col] == max_value:
                    if (row == 0 and col == 0) or (row == 3 and col == 3) or (row == 0 and col == 3) or (row == 3 and col == 0):
                        grade += 2*(max_value)
                        continue
                    if (row == 1 and col == 1) or (row == 2 and col == 2) or (row == 1 and col == 2) or (row == 2 and col == 1):
                        grade -= (max_value)
                        continue
                    grade -= (max_value)

        return grade

    def maxCellValue(self, board):
        copy_board = board
        max_value = 0
        for row in range(4):
            for col in range(4):
                if copy_board[row][col] > max_value:
                    max_value = copy_board[row][col]
        return max_value

    def calcCellScore(self, value):
        if value <= 2:
            return 0
        return 2 * self.calcCellScore(value / 2) + value

    def score(self, board):
        cells_sum = 0
        for row in range(4):
            for col in range(4):
                cells_sum += self.calcCellScore(board[row][col])
        return cells_sum

    def emptyCells(self,board):
        counter = 0
        for i in range(4):
            for j in range(4):
                if board[i][j] == 0:
                    counter += 1
        return counter

    def smoothness(self,board):
        counter = 0
        for row in range(4):
            for col in range(4):
                if board[row][col] != 0:
                    i = 1
                    while col+i < 4:
                        if board[row][col + i] == 0:
                            i += 1
                            continue
                        if board[row][col+i] == board[row][col]:
                            counter += board[row][col]
                        break

                    i = 1
                    while row+i < 4:
                        if board[row + i][col] == 0:
                            i += 1
                            continue
                        if board[row+i][col] == board[row][col]:
                            counter += board[row][col]
                        break
        return counter

    def monotonus(self,board):
        ascending_up_to_down = 0
        descending_up_to_down = 0
        ascending_left_to_right = 0
        descending_left_to_right = 0
        for row in range(4):
            for col in range(4):
                if col + 1 < 4:
                    if board[row][col] > board[row][col + 1]:
                        descending_left_to_right -= board[row][col] - board[row][col + 1]
                    else:
                        ascending_left_to_right += board[row][col] - board[row][col + 1]
                if row + 1 < 4:
                    if board[row][col] > board[row + 1][col]:
                        descending_up_to_down -= board[row][col] - board[row + 1][col]
                    else:
                        ascending_up_to_down += board[row][col] - board[row + 1][col]
        return max(descending_left_to_right, ascending_left_to_right) + max(descending_up_to_down, ascending_up_to_down)


    def calculateNewHScore(self, board,time_limit):
        start_time = time.time()
        greedy_score = self.score(board)
        max_value_cell = self.maxCellValue(board)
        if time_limit - (time.time() - start_time) <= 0.05:
            raise TimeoutError()
        empty_value = self.emptyCells(board)
        smoothness_score = self.smoothness(board)
        if time_limit - (time.time() - start_time) <= 0.05:
            raise TimeoutError()
        monoton_score = self.monotonus(board)
        if time_limit - (time.time() - start_time) <= 0.05:
            raise TimeoutError()
        hot_corners = self.hotCorners(board)

        score_fac = 0.2
        max_value_cell_fac = 0.2
        empty_value_fac = 1
        hot_corners_fac = 0.2
        smoothness_score_fac = 0.2
        monoton_score_fac = 0.2

        return float(greedy_score)*score_fac\
            + max_value_cell*max_value_cell_fac \
            + hot_corners * hot_corners_fac\
            + empty_value*empty_value_fac\
            + smoothness_score*smoothness_score_fac\
            + monoton_score*monoton_score_fac


    # GET THE NEXT SUCCESSORS OF A CERTAIN BOARD
    def generate_optional_boards(self, board, value, time_limit):
        start_time = time.time()
        boards = list()
        for i in range(4):
            for j in range(4):
                if time_limit - (time.time() - start_time) <= 0.05:
                    raise TimeoutError()
                if board[i][j] == 0:
                    new_board = copy.deepcopy(board)
                    new_board[i][j] = value
                    boards.append(new_board)
        return boards

    # score move by making greedy on all optional moves
    def score_move(self, boards):
        total_score_for_boards = 0
        for board in boards:
            total_score_for_boards = total_score_for_boards + self.score(board)
        return total_score_for_boards

    # get key by value for dict
    def get_key(self, dict, val):
        for key, value in dict.items():
            if val == value:
                return key
        return None

    def expectimax(self, board, depth, time_limit):
        start_time = time.time()
        if time_limit <= 0.05:
            raise TimeoutError()
        if depth == 0:
            return self.calculateNewHScore(board,time_limit - (time.time()-start_time))
        # best_value = 0 # Dont know if to initialize because return value
        if self.active_player == 'MAXIPLAYER':
            best_value = -math.inf
            for move in Move:
                new_board, done, score = commands[move](board)
                if done:
                    self.change_player('MINIPLAYER')
                    val = self.expectimax(new_board, depth - 1, time_limit - (time.time()-start_time))
                    best_value = max(best_value, val)
            return best_value

        else:  # self.active_player == 'MINIPLAYER':
            avg = 0
            optional_boards = self.generate_optional_boards(board,2, time_limit - (time.time()-start_time))
            for optional_board in optional_boards:
                self.change_player('MAXIPLAYER')
                val_2 = self.expectimax(optional_board, depth - 1, time_limit - (time.time()-start_time))
                avg += 0.9*val_2
            optional_boards = self.generate_optional_boards(board, 4, time_limit - (time.time()-start_time))
            for optional_board in optional_boards:
                self.change_player('MAXIPLAYER')
                val_4 = self.expectimax(optional_board, depth - 1, time_limit - (time.time()-start_time))
                avg += 0.1 * val_4
            return avg

    def get_move(self, board, time_limit) -> Move:
        time_left = time_limit
        depth = 0
        ret_value = 0
        #  while time_left > (2 ** (depth - 1)) * prev_time and time_left > 0.02:
        while True:
            try:
                start_time = time.time()
                optional_moves_score = {}
                for move in Move:
                    new_board, done, score = commands[move](board)
                    if done:
                        optional_moves_score[move] = self.expectimax(new_board, depth,  time_left - (time.time()-start_time))
                ret_value = max(optional_moves_score, key=optional_moves_score.get)
            except TimeoutError:
                break
            depth += 1
            end_time = time.time()
            prev_time = float(end_time - start_time)
            time_left = float(time_left - prev_time)
        print("depth " + str(depth))
        return ret_value
