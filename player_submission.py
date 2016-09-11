#!/usr/bin/env python

# This file is your main submission that will be graded against. Only copy-paste
# code on the relevant classes included here from the IPython notebook. Do not
# add any classes or functions to this file that are not part of the classes
# that we want.

# Submission Class 1
class OpenMoveEvalFn():
    """Evaluation function that outputs a 
    score equal to how many moves are open
    for your computer player on the board."""
    def score(self, game, maximizing_player_turn=True):
        if maximizing_player_turn:
            return len(game.get_legal_moves())
        return len(game.get_opponent_moves())

# Submission Class 2
class CustomEvalFn():
    """Custom evaluation function that acts
    however you think it should. This is not
    required but highly encouraged if you
    want to build the best AI possible."""
    def score(self, game, maximizing_player_turn=True):
        my_moves = len(game.get_legal_moves())
        oppo_moves = len(game.get_opponent_moves())
        if maximizing_player_turn:   
            return my_moves - 0.5 * oppo_moves
        return oppo_moves - 0.5 * my_moves

# Submission Class 3
class CustomPlayer():
    # TODO: finish this class!
    """Player that chooses a move using 
    your evaluation function and 
    a depth-limited minimax algorithm 
    with alpha-beta pruning.
    You must finish and test this player
    to make sure it properly uses minimax
    and alpha-beta to return a good move
    in less than 500 milliseconds."""
    def __init__(self, search_depth=3, eval_fn=CustomEvalFn()):
        # if you find yourself with a superior eval function, update the
        # default value of `eval_fn` to `CustomEvalFn()`
        self.eval_fn = eval_fn
        self.search_depth = search_depth
        
    def move(self, game, legal_moves, time_left):
        #best_move, utility = self.minimax(game, depth=self.search_depth)
        best_move, utility = self.alphabeta(game, depth=self.search_depth)
        # you will eventually replace minimax with alpha-beta
        return best_move

    def utility(self, game, maximizing_player=True):
        """TODO: Update this function to calculate the utility of a game state"""
        
        if game.is_winner(self):
            return float("inf")

        if game.is_opponent_winner(self):
            return float("-inf")

        return self.eval_fn.score(game)

    def minimax(self, game, depth=float("inf"), maximizing_player=True):

        if depth == 0:
            return None, self.utility(game, maximizing_player)
            
        legal_moves = game.get_legal_moves()
        
        best_move = None
        if len(legal_moves) > 0:
            best_move = legal_moves[0]
            
        if maximizing_player:
            best_val = float("-inf")
        else:
            best_val = float("inf")
        
        depth = depth - 1
        for move in legal_moves:
            b = game.forecast_move(move)
            m, v = self.minimax(b, depth, not maximizing_player)
            if (maximizing_player and v > best_val) or ((not maximizing_player) and v < best_val):
                best_move = move
                best_val = v
        
        return best_move, best_val

    def alphabeta(self, game, depth=float("inf"), alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        # TODO: finish this function!
        
        if depth == 0:
            return None, self.utility(game, maximizing_player)
        
        legal_moves = game.get_legal_moves()
        
        best_move = None
        if len(legal_moves) > 0:
            best_move = legal_moves[0]
        
        if maximizing_player:
            best_val = float("-inf")
            for move in legal_moves:
                b = game.forecast_move(move)
                m, v = self.alphabeta(b, depth - 1, alpha, beta, False)
            
                if alpha >= beta:
                    break
                
                if v > best_val:
                    best_val = v
                    alpha = v
                    best_move = move
        else:
            best_val = float("inf")
            for move in legal_moves:
                b = game.forecast_move(move)
                m, v = self.alphabeta(b, depth - 1, alpha, beta, True)
                
                if beta <= alpha:
                    break
                
                if v < best_val:
                    best_val = v
                    beta = v
                    best_move = move        
                    
        return best_move, best_val