from random import randint

class RandomPlayer():
    """Player that chooses a move randomly."""
    def move(self, game, legal_moves, time_left):
        if not legal_moves: return (-1,-1)
        return legal_moves[randint(0,len(legal_moves)-1)]

class HumanPlayer():
    """Player that chooses a move according to user's input."""
    def move(self, game, legal_moves, time_left):
        print game.print_board()

        print('\t'.join(['[%d] %s'%(i,str(move)) for i,move in enumerate(legal_moves)] ))
        
        valid_choice = False
        while not valid_choice:
            try:
                index = int(raw_input('Select move index:'))
                valid_choice = 0 <= index < len(legal_moves)

                if not valid_choice:
                    print('Illegal move! Try again.')
            
            except ValueError:
                print('Invalid index! Try again.')
        return legal_moves[index]

class OpenMoveEvalFn():
    """Evaluation function that outputs a 
    score equal to how many moves are open
    for the active player."""
    def score(self, game):
        # TODO: finish this function!
        return eval_func

class CustomEvalFn():
    """Custom evaluation function that acts
    however you think it should. This is not
    required but highly encouraged if you
    want to build the best AI possible."""
    def score(self, game):
        # TODO: finish this function!
        return eval_func
"""Example test you can run
to make sure your basic evaluation
function works."""
from isolation import Board

if __name__ == "__main__":
    sample_board = Board(RandomPlayer(),RandomPlayer())
    # setting up the board as though we've been playing
    sample_board.move_count = 3
    sample_board.__active_player__ = 0 # player 1 = 0, player 2 = 1
    # 1st board = 7 moves
    sample_board.__board_state__ = [
                [0,2,0,0,0,0,0],
                [0,0,0,0,0,0,0],
                [0,0,1,0,0,0,0],
                [0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0]
    ]
    sample_board.__last_player_move__ = [(2,2),(0,1)]

    # player 1 should have 7 moves available,
    # so board gets a score of 7
    h = OpenMoveEvalFn()
    print('This board has a score of %s.'%(h.score(sample_board)))

class CustomPlayer():
    # TODO: finish this class!
    """Player that chooses a move using 
    your evaluation function and 
    a depth-limited minimax algorithm 
    with alpha-beta pruning.
    You must finish and test this player
    to make sure it properly uses minimax
    and alpha-beta to return a good move
    in less than 1000 milliseconds."""
    def __init__(self, search_depth=3, eval_fn=OpenMoveEvalFn()):
        # if you find yourself with a superior eval function, update the
        # default value of `eval_fn` to `CustomEvalFn()`
        self.eval_fn = eval_fn
        self.search_depth = search_depth

    def move(self, game, legal_moves, time_left):

        best_move, utility = self.minimax(game, depth=self.search_depth)
        # you will eventually replace minimax with alpha-beta
        return best_move

    def utility(self, game):
        
        if game.is_winner(self):
            return float("inf")

        if game.is_opponent_winner(self):
            return float("-inf")

        return self.eval_fn.score(game)

    def minimax(self, game, depth=float("inf"), maximizing_player=True):
        # TODO: finish this function!
        return best_move, best_val

    def alphabeta(game, depth=float("inf"), alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        # TODO: finish this function!
        return best_move, best_val



"""Example test to make sure
your minimax works, using the
#my_moves evaluation function."""
from isolation import Board

if __name__ == "__main__":
    # create dummy 3x2 board
    p1 = CustomPlayer(search_depth=3)
    p2 = CustomPlayer()
    b = Board(p1,p2,3,3)
    b.__board_state__ = [
        [0,0,0],
        [0,0,1],
        [0,0,0]
    ]
    
    # use minimax to determine optimal move 
    # sequence for player 1
    b.play_isolation()
    # your output should look like this:
    """
    ####################
      |   |   | 
      |   | - | 
      |   |   | 

    ####################
    ####################
    1 |   |   | 
      |   | - | 
      |   |   | 

    ####################
    ####################
    1 | 2 |   | 
      |   | - | 
      |   |   | 

    ####################
    ####################
    - | 2 |   | 
      |   | - | 
      | 1 |   | 

    ####################
    ####################
    - | - |   | 
      |   | - | 
    2 | 1 |   | 

    ####################
    ####################
    - | - | 1 | 
      |   | - | 
    2 | - |   | 

    ####################
    Illegal move at -1,-1.
    Player 1 wins.
    """
"""Example test you can run
to make sure your AI does better
than random."""
from isolation import Board

if __name__ == "__main__":
    r = RandomPlayer()
    h = CustomPlayer()
    game = Board(h,r)
    game.play_isolation()
