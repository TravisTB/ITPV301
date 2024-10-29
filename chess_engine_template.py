import torch
import chess

class ChessAI(torch.nn.Module):
    def __init__(self):
        super(ChessAI, self).__init__()
        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 1)

    def forward(self, board_tensor):
        x = torch.relu(self.fc1(board_tensor))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

def board_to_tensor(board):
    board_fen = board.board_fen().replace('/', '')
    tensor = torch.zeros(64)
    index = 0
    for square in board_fen:
        if square.isdigit():
            index += int(square)
        else:
            tensor[index] = 1 if square.isupper() else -1
            index += 1
    return tensor

class UCIChessEngine:
    def __init__(self, model_path):
        self.model = ChessAI()  # Initialize the model
        self.model.load_state_dict(torch.load(model_path,  weights_only=True))  # Load the model's state
        self.model.eval()

    def uci_main(self):
        board = chess.Board()
        while True:
            command = input().strip()
            if command == 'uci':
                print('id name ChessAI')
                print('id author AI Engine')
                print('uciok')
            elif command == 'isready':
                print('readyok')
            elif command.startswith('position'):
                board = self._parse_position(command)
            elif command.startswith('go'):
                best_move = self._find_best_move(board)
                print(f'bestmove {best_move}')
            elif command == 'quit':
                break

    def _parse_position(self, command):
        tokens = command.split()
        board = chess.Board()
        if tokens[1] == 'startpos':
            board = chess.Board()
            if 'moves' in tokens:
                moves_start = tokens.index('moves') + 1
                for move in tokens[moves_start:]:
                    board.push_uci(move)
        else:
            fen = ' '.join(tokens[1:7])
            board = chess.Board(fen)
            if 'moves' in tokens:
                moves_start = tokens.index('moves') + 1
                for move in tokens[moves_start:]:
                    board.push_uci(move)
        return board

    def _find_best_move(self, board):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return chess.Move.null()

        best_move = chess.Move.from_uci("e2e3")  # Default move
        best_score = -float('inf')

        for move in legal_moves:
            board.push(move)
            board_tensor = board_to_tensor(board).unsqueeze(0)
            score = self.model(board_tensor).item()
            board.pop()

            if score > best_score:
                best_score = score
                best_move = move

        return best_move

if __name__ == '__main__':
    model_path = '{model_path}'  # Replace with your actual model path
    engine = UCIChessEngine(model_path=model_path)
    engine.uci_main()