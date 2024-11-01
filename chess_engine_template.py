import torch
import chess


class ChessAI(torch.nn.Module):
    def __init__(self):
        super(ChessAI, self).__init__()
        self.layers = torch.nn.ModuleList()  # Use ModuleList for flexibility

    def forward(self, board_tensor):
        x = board_tensor
        for layer in self.layers[:-1]:  # Apply activation to all layers except the last one
            x = torch.relu(layer(x))
        x = torch.sigmoid(self.layers[-1](x))  # Use sigmoid for the output layer
        return x

    def configure_layers(self, state_dict):
        # Clear any existing layers in case of reconfiguration
        self.layers = torch.nn.ModuleList()

        # Extract layer shapes based on 2D tensors in state_dict
        layer_sizes = [param.shape for param in state_dict.values() if torch.is_tensor(param) and param.ndim == 2]

        # Diagnostic print to verify layer_sizes content
        print("Layer sizes inferred from state_dict:", layer_sizes)

        # Create a layer for each pair of in_features and out_features from layer_sizes
        for in_features, out_features in zip([layer_sizes[0][1]] + [s[0] for s in layer_sizes[:-1]],
                                             [s[0] for s in layer_sizes]):
            self.layers.append(torch.nn.Linear(in_features, out_features))

        # Load the model weights
        self.load_state_dict(state_dict)


# Helper function to load the model with configuration inferred from the state_dict
def load_chess_ai(model_path):
    try:
        # Load the checkpoint dictionary
        checkpoint = torch.load(model_path, weights_only=True)
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


def load_chess_ai(model_path):
    try:
        # Load the checkpoint dictionary
        checkpoint = torch.load(model_path, weights_only=True)
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
    tensor = torch.zeros(64)
    index = 0
    for square in board_fen:
        if square.isdigit():
            index += int(square)
        else:
            tensor[index] = 1 if square.isupper() else -1
            index += 1
    return tensor


# Main execution block
if __name__ == '__main__':
    model_path = '{model_path}'  # Replace with your actual model path
    engine = UCIChessEngine(model_path)
    engine.uci_main()