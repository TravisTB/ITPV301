import PySimpleGUI as sg
import pygame
import chess
import time
import chess.engine  # For AI integration (Stockfish)

# Initialize pygame
pygame.init()

# Constants
SQUARE_SIZE = 60
BOARD_SIZE = 8
SCREEN_WIDTH = BOARD_SIZE * SQUARE_SIZE
SCREEN_HEIGHT = BOARD_SIZE * SQUARE_SIZE
DEFAULT_BG_COLOR = '#323232'
WHITE = (255, 255, 255)
DARK_AQUA = (0, 139, 139)

# Chess board setup
board = chess.Board()

# Set up the Pygame display
pygame.display.set_caption("Interactive Chessboard")
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

# Timer setup
start_time = time.time()


def update_timer():
    """Update the timer showing how long the current player has taken."""
    elapsed = time.time() - start_time
    minutes, seconds = divmod(elapsed, 60)
    return f"{int(minutes)}:{int(seconds):02d}"


def draw_chessboard(selected_square=None):
    """Draws a chessboard using pygame and returns the surface."""
    board_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    board_surface.fill(DEFAULT_BG_COLOR)

    # Draw squares
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            color = WHITE if (row + col) % 2 == 0 else DARK_AQUA
            square = chess.square(col, row)

            # Highlight selected square
            if selected_square == square:
                color = (255, 255, 0)  # Yellow color for selected square

            pygame.draw.rect(board_surface, color,
                             pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

    return board_surface


def draw_legal_moves(surface, selected_square):
    """Highlight legal moves for the selected piece."""
    if selected_square:
        legal_moves = [move for move in board.legal_moves if move.from_square == selected_square]

        for move in legal_moves:
            row, col = divmod(move.to_square, 8)
            pygame.draw.circle(surface, (0, 255, 0),  # Green color for legal moves
                               (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2), 10)


def draw_pieces(surface):
    """Draws pieces on the board according to the current state with improved quality."""
    piece_images = {
        'P': pygame.image.load('pieces/white-pawn.png').convert_alpha(),
        'N': pygame.image.load('pieces/white-knight.png').convert_alpha(),
        'B': pygame.image.load('pieces/white-bishop.png').convert_alpha(),
        'R': pygame.image.load('pieces/white-rook.png').convert_alpha(),
        'Q': pygame.image.load('pieces/white-queen.png').convert_alpha(),
        'K': pygame.image.load('pieces/white-king.png').convert_alpha(),
        'p': pygame.image.load('pieces/black-pawn.png').convert_alpha(),
        'n': pygame.image.load('pieces/black-knight.png').convert_alpha(),
        'b': pygame.image.load('pieces/black-bishop.png').convert_alpha(),
        'r': pygame.image.load('pieces/black-rook.png').convert_alpha(),
        'q': pygame.image.load('pieces/black-queen.png').convert_alpha(),
        'k': pygame.image.load('pieces/black-king.png').convert_alpha(),
    }

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                piece_image = piece_images[piece.symbol()]

                # Scale the piece to fit within the square with anti-aliasing
                piece_image = pygame.transform.scale(piece_image, (SQUARE_SIZE, SQUARE_SIZE))

                # Reverse the row order
                row, col = divmod(square, 8)
                row = 7 - row  # This reverses the rows so the board starts from the bottom

                # Draw the piece at the adjusted position
                surface.blit(piece_image, (col * SQUARE_SIZE, row * SQUARE_SIZE))


def get_square_from_mouse(pos):
    """Get the square index from the mouse position."""
    x, y = pos
    col = x // SQUARE_SIZE
    row = y // SQUARE_SIZE
    return chess.square(col, row)


# Undo Move Feature
def undo_last_move():
    """Undo the last move made on the board."""
    if board.move_stack:  # Check if there are any moves to undo
        last_move = board.pop()  # Undo the last move
        print(f"Undid move: {last_move}")


# Move History Feature
def display_move_history():
    """Display the history of moves made in the game."""
    try:
        if board.move_stack:
            move_history = []
            for move in board.move_stack:
                try:
                    move_history.append(board.san(move))  # Convert to SAN if legal
                except AssertionError:
                    move_history.append(str(move))  # Fallback: use LAN (e.g., 'e2e4') if SAN fails
            sg.popup("Move History", "\n".join(move_history))
        else:
            sg.popup("No moves made yet!")
    except Exception as e:
        sg.popup(f"Error displaying move history: {str(e)}")

# Save and Load Game Features
def save_game():
    """Save the current board state to a file."""
    fen = board.fen()
    with open("saved_game.fen", "w") as file:
        file.write(fen)
    sg.popup("Game Saved")


def load_game():
    """Load a saved game from a file."""
    try:
        with open("saved_game.fen", "r") as file:
            fen = file.read()
            board.set_fen(fen)
        sg.popup("Game Loaded")
    except FileNotFoundError:
        sg.popup("No saved game found.")


# AI Integration Feature (using Stockfish)
def make_ai_move():
    """Let AI (Stockfish) make a move."""
    try:
        engine = chess.engine.SimpleEngine.popen_uci("chess_engine2222.exe")  # Replace with your Stockfish path
        result = engine.play(board, chess.engine.Limit(time=2.0))
        board.push(result.move)
        engine.quit()
    except FileNotFoundError:
        sg.popup("AI engine (Stockfish) not found!")


# Main function
def main():
    layout = [
        [sg.Text('', key='-TIMER-')],  # Timer display
        [sg.Image(filename='', key='-BOARD-')],
        [sg.Button('Undo Move'), sg.Button('Move History'), sg.Button('Save Game'), sg.Button('Load Game')],
        [sg.Button('AI Move')]
    ]

    window = sg.Window("Interactive Chessboard", layout, finalize=True)

    selected_square = None  # To track the selected piece

    while True:
        event, values = window.read(timeout=100)

        if event in (sg.WIN_CLOSED, 'Exit'):
            break

        # Update the timer
        window['-TIMER-'].update(update_timer())

        # Draw the board
        board_surface = draw_chessboard(selected_square)
        draw_pieces(board_surface)

        # Highlight legal moves for the selected piece
        if selected_square:
            draw_legal_moves(board_surface, selected_square)

        # Save the board surface to a temporary file to display in PySimpleGUI
        pygame.image.save(board_surface, "chessboard.png")
        window['-BOARD-'].update(filename="chessboard.png")

        # Handle mouse clicks
        if event == '-BOARD-':
            mouse_pos = values['-BOARD-']
            if mouse_pos:
                square = get_square_from_mouse(mouse_pos)

                # If no piece is selected, select the clicked square if a piece exists
                if selected_square is None:
                    if board.piece_at(square):  # Check if there's a piece at the square
                        selected_square = square  # Select the piece
                        print(f"Selected {chess.square_name(square)}")

                # If a piece is selected, try to move it
                else:
                    move = chess.Move(selected_square, square)
                    if move in board.legal_moves:
                        board.push(move)  # Make the move on the board
                        print(f"Moved piece to {chess.square_name(square)}")

                        # Checkmate/Check Detection Feature
                        if board.is_checkmate():
                            sg.popup("Checkmate! Game over.")
                        elif board.is_stalemate():
                            sg.popup("Stalemate! Game over.")
                        elif board.is_check():
                            sg.popup("Check!")

                    else:
                        print(f"Invalid move from {chess.square_name(selected_square)} to {chess.square_name(square)}")

                    selected_square = None  # Reset the selection

        # Handle extra features (Undo, History, Save/Load, AI Move)
        if event == 'Undo Move':
            undo_last_move()
        elif event == 'Move History':
            display_move_history()
        elif event == 'Save Game':
            save_game()
        elif event == 'Load Game':
            load_game()
        elif event == 'AI Move':
            make_ai_move()

    window.close()
    pygame.quit()


if __name__ == "__main__":
    main()
