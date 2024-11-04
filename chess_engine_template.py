import os
import sys
import chess

# Import only the required submodules from torch
from torch import nn
from torch import tensor, zeros, relu, sigmoid, load
from torch.autograd import Variable

class ChessAI(nn.Module):
    def __init__(self):
        super(ChessAI, self).__init__()
        self.layers = nn.ModuleList()  # Use ModuleList for flexibility

    def forward(self, board_tensor):
        x = board_tensor
        for layer in self.layers[:-1]:  # Apply activation to all layers except the last one
            x = relu(layer(x))
        x = sigmoid(self.layers[-1](x))  # Use sigmoid for the output layer
        return x

    def configure_layers(self, state_dict):
        # Clear any existing layers in case of reconfiguration
        self.layers = nn.ModuleList()

        # Extract layer shapes based on 2D tensors in state_dict
        layer_sizes = [param.shape for param in state_dict.values() if hasattr(param, 'shape') and param.ndim == 2]

        # Diagnostic print to verify layer_sizes content
        print("Layer sizes inferred from state_dict:", layer_sizes)

        # Create a layer for each pair of in_features and out_features from layer_sizes
        for in_features, out_features in zip([layer_sizes[0][1]] + [s[0] for s in layer_sizes[:-1]],
                                             [s[0] for s in layer_sizes]):
            self.layers.append(nn.Linear(in_features, out_features))

        # Load the model weights
        self.load_state_dict(state_dict)


# Helper function to load the model with configuration inferred from the state_dict
def load_chess_ai(model_path):
    try:
        # Load the checkpoint dictionary
        checkpoint = load(model_path, weights_only=True)
        state_dict = checkpoint["model_state_dict"]  # Access model weights specifically

        # Initialize and configure the model
        model = ChessAI()
        model.configure_layers(state_dict)  # Configure layers dynamically
        model.load_state_dict(state_dict)  # Load weights into configured layers
        model.eval()
        return model
    except ValueError as e:
        print(f"Error configuring layers: {e}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


class UCIChessEngine:
    def __init__(self, model_path):
        # Load and configure the model
        self.model = load_chess_ai(model_path)

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


# Convert the board to a tensor representation
def board_to_tensor(board):
    board_fen = board.board_fen().replace('/', '')
    tensor = zeros(65)  # 64 squares + 1 for turn indicator
    index = 0

    # Mapping each piece to an integer value
    piece_map = {
        'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,  # White pieces
        'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6  # Black pieces
    }

    for square in board_fen:
        if square.isdigit():
            index += int(square)  # Advance by empty square count
        else:
            tensor[index] = piece_map.get(square, 0)
            index += 1

    # Set the turn indicator at the last index
    tensor[64] = 1 if board.turn else -1  # +1 if white's turn, -1 if black's turn

    return tensor


# Main execution block
if __name__ == '__main__':
    # Define paths
    model_path = "{model_path}"  # Placeholder that can be dynamically replaced during compilation

    # Check if running in a compiled environment (frozen) or development
    if getattr(sys, 'frozen', False):
        # When running as a compiled executable
        main_directory = sys._MEIPASS  # Temporary directory used by PyInstaller for compiled files
        model_filename = os.path.basename(model_path)
        model_path = os.path.join(main_directory, model_filename)
    else:
        # When running as a script during development
        if model_path != "{model_path}":
            model_filename = os.path.basename(model_path)
        else:
            # Set a default model filename if no model path is provided
            model_filename = "new_model.pt"  # Replace with your actual model filename

        # If model_path is still a placeholder, search for the model file in different directories
        if model_path == "{model_path}":
            main_directory = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.join(main_directory, 'models')
            model_path = os.path.join(main_directory, model_filename)
            if not os.path.exists(model_path):
                model_path = os.path.join(models_dir, model_filename)
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model file '{model_filename}' not found in main path or 'models' directory.")

    # Load the engine and start the UCI loop
    engine = UCIChessEngine(model_path)
    engine.uci_main()