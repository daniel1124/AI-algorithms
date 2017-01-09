#!/usr/bin/env python


class OpenMoveEvalFn():
    """Evaluation function that outputs a 
    score equal to how many moves are open
    for the computer player on the board."""
    def score(self, game, maximizing_player_turn=True):
        if maximizing_player_turn:
            return len(game.get_legal_moves())
        return len(game.get_opponent_moves())

class CustomEvalFn():
    """Custom evaluation function."""
    def score(self, game, maximizing_player_turn=True):
        my_moves = len(game.get_legal_moves())
        oppo_moves = len(game.get_opponent_moves())
        if maximizing_player_turn:   
            return my_moves - 0.5 * oppo_moves
        return oppo_moves - 0.5 * my_moves

class CustomPlayer():
    """Player that chooses a move using 
    the evaluation function and 
    a depth-limited minimax algorithm 
    with alpha-beta pruning. It should return a good move
    in less than 500 milliseconds."""
    
    def __init__(self, search_depth=3, eval_fn=CustomEvalFn()):

        self.eval_fn = eval_fn
        self.search_depth = search_depth
        
    def move(self, game, legal_moves, time_left):
        best_move, utility = self.alphabeta(game, depth=self.search_depth)
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

class CustomPlayer2():
    # Use iterative deepening."""

    def __init__(self, search_depth=3, eval_fn=CustomEvalFn()):

        self.eval_fn = eval_fn
        self.search_depth = search_depth
    
    def move(self, game, legal_moves, time_left):

        start = time_left()

        if game.move_count == 0:
            return int(game.width / 2), int(game.height / 2)
        
        best_move, utility = self.alphabeta(game, start, time_left, depth=self.search_depth)
        pre_utility = utility
        cur = time_left()
        
        depth = self.search_depth
        while start - cur <= 400:
            try:
                depth = depth + 1
                res_move, res_utility = self.alphabeta(game, start, time_left, depth=depth)
                if res_utility is not None and res_utility != float("-inf"):
                    utility = 0.7 * res_utility + 0.3 * pre_utility  
                    if utility > 0.7 * pre_utility:
                        best_move = res_move
                    pre_utility = utility
                cur = time_left()
            except Exception as e:
                break
        # print(start - time_left())
        return best_move

    def utility(self, game, maximizing_player=True):
        
        if game.is_winner(self):
            return float("inf")

        if game.is_opponent_winner(self):
            return float("-inf")

        return self.eval_fn.score(game, maximizing_player)

    def minimax(self, game, start, time_left, depth=float("inf"), maximizing_player=True):

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
            m, v = self.minimax(b, start, time_left, depth, not maximizing_player)
            if (maximizing_player and v > best_val) or ((not maximizing_player) and v < best_val):
                best_move = move
                best_val = v
            
            if start - time_left() > 400:
                return None, None

        return best_move, best_val

    def alphabeta(self, game, start, time_left, depth=float("inf"), alpha=float("-inf"), beta=float("inf"),
                  maximizing_player=True):
        
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
                m, v = self.alphabeta(b, start, time_left, depth - 1, alpha, beta, False)
                
                if alpha >= beta:
                    break
                
                if v > best_val:
                    best_val = v
                    alpha = v
                    best_move = move
                    
                if start - time_left() > 400:
                    return None, None
        else:
            best_val = float("inf")
            for move in legal_moves:
                b = game.forecast_move(move)
                m, v = self.alphabeta(b, start, time_left, depth - 1, alpha, beta, True)
                
                if beta <= alpha:
                    break
                
                if v < best_val:
                    best_val = v
                    beta = v
                    best_move = move        
                
                if start - time_left() > 400:
                    return None, None
                
        return best_move, best_val
