class Player():
    #choose between X and O
    def __init__(self, symbol):
        self.symbol = symbol

class State():
    def __init__(self, board):
        self.board = [[" " for i in range(3)] for j in range(3)]
        self.currentPlayer = 'X'
    
    def display(self):
        for row in self.board:
            print("|".join(row))
            print(5 * "-")
            
    def isWin(self, symbol):
        #collumns check
        for col in range(3):
            if board[0][col] == board[1][col] == board[2][col] == symbol:
                return True
        
        #rows check
        for row in range(3):
            if board[row][0] == board[row][1] == board[row][2] == symbol:
                return True
        
        #diagonals check
        if board[0][0] == board[1][1] == board[2][2] == symbol:
            return True
        if board[0][2] == board[1][1] == board[2][0] == symbol:
            return True

        return False
    
    def available_moves(self):
        return [(row, col) for row in range(3) for col in range(3) if self.board[row][col] == " "]
        

    

class Game():
    def __init__(self):
        pass


board = []
state = State(board)
state.display()