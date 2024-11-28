import random

class Player():
    def __init__(self, symbol, strategy):
        self.symbol = symbol
        self.strategy = strategy
    
class State():
    def __init__(self):
        self.board = [[" " for i in range(3)] for j in range(3)]
        self.currentPlayer = 'X'
    
    def display(self):
        for row in self.board:
            print("|".join(row))
            print(5 * "-")
        print('\n')
            
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
        moves = [(row, col) for row in range(3) for col in range(3) if self.board[row][col] == ' ']
        random.shuffle(moves)
        return moves
    
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
        strategy = ''
        while strategy not in ['minimax', 'random', 'manual']:
            strategy = input("Write down disiered strategy (minimax, random, manual):\n").strip().lower()

        self.state = State()
        self.players = {
            'X': Player('X', strategy='minimax'),
            'O': Player('O', strategy=strategy)
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
            return self.h(state)
        
        #w(u) with the best possible moves
        evaluations = []
        for move in state.availableMoves():
            nextState = state.makeMove(move[0], move[1], 'X' if isMaxMove else 'O')
            evaluation = self.minimax(nextState, depth - 1, not isMaxMove)
            evaluations.append(evaluation)

        if isMaxMove:
            return max(evaluations)
        else:
            return min(evaluations)
        
    def bestMove(self, state, depth):
        bestValue = -float('inf') if state.currentPlayer == 'X' else float('inf')
        bestMove = None

        for move in state.availableMoves():
            nextState = state.makeMove(move[0], move[1], state.currentPlayer)
            minimaxValue = self.minimax(nextState, depth - 1, state.currentPlayer == 'O')

            if state.currentPlayer == 'X' and minimaxValue > bestValue:
                bestValue = minimaxValue
                bestMove = move
            elif state.currentPlayer == 'O' and minimaxValue < bestValue:
                bestValue = minimaxValue
                bestMove = move
            
        return bestMove
    
    def randomMove(self, state):
        return random.choice(state.availableMoves())
    
    def manualMove(self, state):
        move = None
        while True:
            row = int(input("Write down row number:\n"))
            col = int(input("Write down collumn number:\n"))
            move = (row, col)
            if move in state.availableMoves():
                return move

    def moveByStrategy(self, player, state, depth=5):
        if player.strategy == 'minimax':
            return self.bestMove(state, depth)
        elif player.strategy == 'random':
            return self.randomMove(state)
        elif player.strategy == 'manual':
            return self.manualMove(state)
        
    def gameLoop(self):
        while not (self.state.isWin('X') or self.state.isWin('O') or self.state.isDraw()):
            self.state.display()
            currentPlayer = self.players[self.state.currentPlayer]
            move = self.moveByStrategy(currentPlayer, self.state)
            self.state = self.state.makeMove(move[0], move[1], self.state.currentPlayer)
        
        self.state.display()
        if self.state.isWin('X'):
            print('X Wins')
        elif self.state.isWin('O'):
            print('O Wins')
        else:
            print('Draw')


game = Game()
game.gameLoop()