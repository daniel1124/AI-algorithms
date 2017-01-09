from copy import deepcopy
from time import time, sleep
import StringIO

# Annoying hack for when script itself is a symlink
import sys,os
sys.path[0] = os.getcwd()

class Board:
    BLANK = 0
    NOT_MOVED = (-1, -1)

    def __init__(self, player_1, player_2, width=7, height=7):
        self.width=width
        self.height=height
        self.__board_state__ = [ [Board.BLANK for i in range(0, width)] for j in range(0, height)]
        self.__last_player_move__ = {player_1:Board.NOT_MOVED, player_2:Board.NOT_MOVED}
        self.__player_symbols__ = {Board.BLANK: Board.BLANK, player_1:1, player_2:2}
        self.move_count = 0
        self.__active_player__ = player_1
        self.__inactive_player__ = player_2
        self.__player_1__ = player_1
        self.__player_2__ = player_2

    def get_state(self):
        return deepcopy(self.__board_state__)

    def __apply_move__(self, move):
        row,col = move
        self.__last_player_move__[self.__active_player__] = move
        self.__board_state__[row][col] = self.__player_symbols__[self.__active_player__]
        tmp = self.__active_player__
        self.__active_player__ = self.__inactive_player__
        self.__inactive_player__ = tmp
        self.move_count = self.move_count + 1

    def copy(self):
        b = Board(self.__player_1__, self.__player_2__, width=self.width, height=self.height)
        for key, value in self.__last_player_move__.items():
            b.__last_player_move__[key] = value
        for key, value in self.__player_symbols__.items():
            b.__player_symbols__[key] = value
        b.move_count = self.move_count
        b.__active_player__ = self.__active_player__
        b.__inactive_player__ = self.__inactive_player__
        b.__board_state__ = self.get_state()
        return b

    def forecast_move(self, move):
        new_board = self.copy()
        new_board.__apply_move__(move)
        return new_board

    def get_active_player(self):
        return self.__active_player__

    def get_inactive_player(self):
        return self.__inactive_player__

    def is_winner(self, player):
        return not self.get_legal_moves() and player== self.__inactive_player__

    def is_opponent_winner(self, player):
        return not self.get_legal_moves() and player== self.__active_player__

    def get_opponent_moves(self):
        return self.__get_moves__(self.__last_player_move__[self.__inactive_player__])

    def get_legal_moves(self):
        return self.__get_moves__(self.__last_player_move__[self.__active_player__])

    def __get_moves__(self, move):
        if move == Board.NOT_MOVED:
            return self.get_first_moves()

        r, c = move

        directions = [ (-2, -1), (-2, 1),
                       (-1, -2), (-1, 2),
                        (1, -2),  (1, 2),
                        (2, -1),  (2, 1) ]

        valid_moves = [(r+dr,c+dc) for dr, dc in directions
                if self.move_is_legal(r+dr, c+dc)]

        return valid_moves

    def get_first_moves(self):
        return [ (i,j) for i in range(0,self.height) for j in range(0,self.width) if self.__board_state__[i][j] == Board.BLANK]

    def move_is_legal(self, row, col):
        return 0 <= row < self.height and \
               0 <= col < self.width  and \
                self.__board_state__[row][col] == Board.BLANK

    def get_blank_spaces(self):
        return self.get_player_locations(Board.BLANK)

    def get_player_locations(self, player):
        return [ (i,j) for j in range(0, self.width) for i in range(0,self.height) if self.__board_state__[i][j] == self.__player_symbols__[player]]

    def get_last_move_for_player(self, player):
        return self.__last_player_move__[player]

    def print_board(self):

        p1_r, p1_c = self.__last_player_move__[self.__player_1__]
        p2_r, p2_c = self.__last_player_move__[self.__player_2__]

        b = self.__board_state__

        out = ''

        for i in range(0, len(b)):
            for j in range(0, len(b[i])):

                if not b[i][j]:
                    out += ' '

                elif i == p1_r and j == p1_c:
                    out += '1'
                elif i == p2_r and j == p2_c:
                    out += '2'
                else:
                    out += '-'

                out += ' | '
            out += '\n\r'

        return out

    def play_isolation(self, time_limit = None):
        move_history = []

        curr_time_millis = lambda : int(round(time() * 1000))

        while True:

            legal_player_moves =  self.get_legal_moves() # This is only provided as an added benefit

            move_start = curr_time_millis()

            time_left = lambda : time_limit - (curr_time_millis() - move_start)

            curr_move = Board.NOT_MOVED
            try:
                curr_move = self.__active_player__.move(self, legal_player_moves, time_left)
            except Exception as e:
                print(e)
                pass

            if curr_move is None:
                curr_move = Board.NOT_MOVED

            if self.__active_player__ == self.__player_1__:
                move_history.append([curr_move])
            else:
                move_history[-1].append(curr_move)

            if time_limit and time_left() <= 0:
                return  self.__inactive_player__, move_history, "timeout"

            if curr_move not in legal_player_moves:
                return self.__inactive_player__, move_history, "illegal move"

            self.__apply_move__(curr_move)


def game_as_text(winner, move_history, termination="", board=Board(1,2)):
    ans = StringIO.StringIO()

    for i, move in enumerate(move_history):
        p1_move = move[0]
        ans.write("%d." % i + " (%d,%d)\n" % p1_move)
        if p1_move != Board.NOT_MOVED:
            board.__apply_move__(p1_move)
        ans.write(board.print_board())

        if len(move) > 1:
            p2_move = move[1]
            ans.write("%d. ..." % i + " (%d,%d)\n" % p2_move)
            if p2_move != Board.NOT_MOVED:
                board.__apply_move__(p2_move)
            ans.write(board.print_board())

    ans.write(termination + "\n")

    ans.write("Winner: " + str(winner) + "\n")

    return ans.getvalue()

if __name__ == '__main__':

    print("Starting game:")

    from test_players import RandomPlayer
    from test_players import HumanPlayer

    board = Board(RandomPlayer(), HumanPlayer())
    winner, move_history, termination = board.play_isolation()

    print game_as_text(winner, move_history, termination)
