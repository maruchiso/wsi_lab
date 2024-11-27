class Player():
    #choose between X and O
    def __init__(self, symbol):
        self.symbol = symbol

class State():
    def __init__(self):
        self.board = [[" " for i in range(3)] for j in range(3)]
        self.currentPlayer = 'X'
    
    def display(self):
        for row in self.board:
            print("|".join(row))
            print(5 * "-")
            
    def isWin(self, symbol):
        #collumns check
        for col in range(3):
            if self.board[0][col] == self.board[1][col] == self.board[2][col] == symbol:
                return True
        
        #rows check
        for row in range(3):
            if self.board[row][0] == self.board[row][1] == self.board[row][2] == symbol:
                return True
        
        #diagonals check
        if self.board[0][0] == self.board[1][1] == self.board[2][2] == symbol:
            return True
        if self.board[0][2] == self.board[1][1] == self.board[2][0] == symbol:
            return True

        return False
    
    def isDraw(self):
        for row in range(3):
            for col in range(3):
                if self.board[row][col] == ' ':
                    return False
        return True

    
    def availableMoves(self):
        return [(row, col) for row in range(3) for col in range(3) if self.board[row][col] == " "]
    
    def makeMove(self, row, col, symbol):
        newState = State()
        newState.board = [row[:] for row in self.board]
        newState.board[row][col] = symbol
        if self.currentPlayer == 'X':
            newState.currentPlayer = 'O'
        else:
            newState.currentPlayer = 'X'
        return newState


    

class Game():
    def __init__(self):
        self.state = State()
        self.players = {
            'X': Player('X'),
            'O': Player('O')
        }

    def h(self, state):
        if state.isWin('X'):
            return 1
        elif state.isWin('O'):
            return -1
        else:
            return 0
        

    def minimax(self, state, depth, isMaxMove):
        if state.isWin('X') or state.isWin('O') or state.isDraw() or depth == 0:
            self.h(state)
        
        if not state.availableMoves():
            return self.h(state)
        #w(u) with the best possible moves
        evaluations = []
        for row, col in state.availableMoves():
            nextState = state.makeMove(row, col, 'X' if isMaxMove else 'O')
            evaluation = self.minimax(nextState, depth - 1, not isMaxMove)
            evaluations.append(evaluation)

        if isMaxMove:
            return max(evaluations)
        else:
            return min(evaluations)
        
    def bestMove(self, state, depth):
        bestValue = -2 if state.currentPlayer == 'X' else 2
        bestMove = [0, 0]

        for row, col in state.availableMoves():
            nextState = state.makeMove(row, col, state.currentPlayer)
            possibleBestMove = self.minimax(nextState, depth - 1, state.currentPlayer == 'O')

            if state.currentPlayer == 'X' and possibleBestMove > bestValue:
                return possibleBestMove
            elif state.currentPlayer == 'O' and possibleBestMove < bestValue:
                return possibleBestMove

    def gameLoop(self):
        while not (self.state.isWin('X') or self.state.isWin('O') or self.state.isDraw()):
            self.state.display()
            
            if self.state.currentPlayer == 'X':
                row, col = self.bestMove(self.state, depth=9)
            else:
                row, col = self.bestMove(self.state, depth=9)
        
        self.state.display()
        if self.state.isWin('X'):
            print('X Wygrał')
        if self.state.isWin('O'):
            print('O Wygrał')
        else:
            print('Remis')


game = Game()
game.gameLoop()