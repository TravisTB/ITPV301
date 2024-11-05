import pygame
import chess
import chess.engine
import time
from tkinter import filedialog, Tk
import atexit

# Initialize pygame and hide tkinter root window
pygame.init()
Tk().withdraw()  # Hide the main tkinter window
engines = {}
# Constants
SQUARE_SIZE = 60
BOARD_SIZE = 8
SCREEN_WIDTH = BOARD_SIZE * SQUARE_SIZE
SCREEN_HEIGHT = BOARD_SIZE * SQUARE_SIZE + 150  # Extra space for buttons and timer
DEFAULT_BG_COLOR = '#323232'
WHITE = (255, 255, 255)
DARK_AQUA = (0, 139, 139)

# Chess board setup
board = chess.Board()

# Set up the Pygame display
pygame.display.set_caption("Interactive Chessboard")
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

# Load piece images
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

# Button colors
BUTTON_COLOR = (200, 200, 200)
BUTTON_TEXT_COLOR = (0, 0, 0)

# Engine paths and autoplay state
white_engine_path = None
black_engine_path = None
autoplay_enabled = False  # Use this single variable consistently

# Timer setup
start_time = time.time()

# Global variables for AI move timing
last_ai_move_time = time.time()
ai_move_delay = 2.0  # Time in seconds between AI moves

# Function to draw the board
def draw_board(selected_square=None, legal_moves=[]):
    screen.fill(DEFAULT_BG_COLOR)

    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            color = WHITE if (row + col) % 2 == 0 else DARK_AQUA
            square_rect = pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)

            # Highlight selected square
            if selected_square is not None and selected_square == chess.square(col, 7 - row):
                color = (255, 255, 0)  # Yellow for selected piece

            pygame.draw.rect(screen, color, square_rect)

    # Draw pieces after squares
    draw_pieces()

    # Draw legal move indicators above pieces
    if legal_moves:
        for move in legal_moves:
            to_square = move.to_square
            col = chess.square_file(to_square)
            row = 7 - chess.square_rank(to_square)
            pygame.draw.circle(screen, (0, 255, 0),  # Green circle for legal moves
                               (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2), 10)

# Function to draw pieces on the board
def draw_pieces():
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_image = piece_images[piece.symbol()]
            piece_image = pygame.transform.scale(piece_image, (SQUARE_SIZE, SQUARE_SIZE))
            row, col = divmod(square, 8)
            row = 7 - row  # Adjust for board orientation
            screen.blit(piece_image, (col * SQUARE_SIZE, row * SQUARE_SIZE))

# Draw buttons
def draw_buttons():
    buttons = [
        ("Undo", (10, SCREEN_HEIGHT - 140)),         # Adjusted positions
        ("History", (130, SCREEN_HEIGHT - 140)),
        ("Save", (250, SCREEN_HEIGHT - 140)),
        ("Load", (370, SCREEN_HEIGHT - 140)),
        ("AI Move", (10, SCREEN_HEIGHT - 90)),
        ("White AI", (130, SCREEN_HEIGHT - 90)),
        ("Black AI", (250, SCREEN_HEIGHT - 90)),
        ("Autoplay", (370, SCREEN_HEIGHT - 90))
    ]

    for text, pos in buttons:
        pygame.draw.rect(screen, BUTTON_COLOR, (*pos, 100, 40))  # Width adjusted for readability
        font = pygame.font.Font(None, 22)
        label = font.render(text, True, BUTTON_TEXT_COLOR)
        screen.blit(label, (pos[0] + 10, pos[1] + 10))

# Function to draw timer
def draw_timer():
    elapsed = time.time() - start_time
    minutes, seconds = divmod(elapsed, 60)
    timer_text = f"Time: {int(minutes):02d}:{int(seconds):02d}"
    font = pygame.font.Font(None, 25)
    timer_surface = font.render(timer_text, True, (255, 255, 255))
    screen.blit(timer_surface, (SCREEN_WIDTH - 475, 600))

# Load chess engines
def select_white_engine():
    global white_engine_path
    white_engine_path = filedialog.askopenfilename(title="Select White Engine")
    print(f"Selected White Engine: {white_engine_path}")

def select_black_engine():
    global black_engine_path
    black_engine_path = filedialog.askopenfilename(title="Select Black Engine")
    print(f"Selected Black Engine: {black_engine_path}")

# Toggle autoplay
def toggle_autoplay():
    global autoplay_enabled
    autoplay_enabled = not autoplay_enabled
    print(f"Autoplay {'enabled' if autoplay_enabled else 'disabled'}")

# Execute AI Move
def make_ai_move(engine_path):
    if engine_path:
        try:
            # Check if the engine is already loaded
            if engine_path not in engines:
                engine = chess.engine.SimpleEngine.popen_uci(engine_path)
                engines[engine_path] = engine
            else:
                engine = engines[engine_path]

            # Use the engine to play a move with depth 20
            result = engine.play(board, chess.engine.Limit(depth=20))
            board.push(result.move)
            print(f"Engine moved: {board.san(result.move)}")
        except FileNotFoundError:
            print(f"Chess engine at '{engine_path}' not found.")
        except Exception as e:
            print(f"An error occurred in make_ai_move: {e}")
    else:
        print("No engine loaded.")

# Modified autoplay function to handle single AI opponent
def autoplay():
    if not autoplay_enabled:
        return
    if board.is_game_over():
        return

    # Check if an engine is loaded for the side to move
    if (board.turn and white_engine_path) or (not board.turn and black_engine_path):
        if board.turn:  # White to move
            make_ai_move(white_engine_path)
        else:  # Black to move
            make_ai_move(black_engine_path)
        check_for_game_end()
    # If no engine is loaded for the side to move, wait for user input
    # No action needed here

# Cleanup engines when the program exits
def cleanup_engines():
    for engine in engines.values():
        engine.quit()
    engines.clear()

# Register cleanup_engines to be called on program exit
atexit.register(cleanup_engines)

def get_legal_moves(selected_square):
    if selected_square is None:
        return []
    return [move for move in board.legal_moves if move.from_square == selected_square]

# Handle square selection and moves
def handle_click(position, selected_square):
    global last_ai_move_time  # Add this line to access and modify the variable
    current_time = time.time()  # Get current time

    col, row = position[0] // SQUARE_SIZE, 7 - (position[1] // SQUARE_SIZE)
    clicked_square = chess.square(col, row)
    piece = board.piece_at(clicked_square)

    if selected_square is None:
        if piece and piece.color == board.turn:
            return clicked_square
    else:
        move = chess.Move(selected_square, clicked_square)
        if move in board.legal_moves:
            # Handle pawn promotion
            if board.piece_at(selected_square).piece_type == chess.PAWN and (
                chess.square_rank(clicked_square) == 0 or chess.square_rank(clicked_square) == 7
            ):
                # Promote pawn to queen
                move.promotion = chess.QUEEN
            board.push(move)
            check_for_game_end()
            last_ai_move_time = current_time  # Update the last AI move time
        return None
    return selected_square

# Undo last move
def undo_last_move():
    global last_ai_move_time  # Ensure we can modify this variable
    global selected_square     # Reset selected_square
    if board.move_stack:
        board.pop()
        selected_square = None  # Reset selection
        last_ai_move_time = time.time()  # Update the last AI move time

# Display move history
def display_move_history():
    if board.move_stack:
        move_history = [board.san(move) for move in board.move_stack]
        print("Move History:", " ".join(move_history))

# Save and load game
def save_game():
    # Open file dialog to select save location
    filename = filedialog.asksaveasfilename(
        title="Save Game",
        defaultextension=".fen",
        filetypes=[("FEN Files", "*.fen"), ("All Files", "*.*")]
    )
    if filename:
        try:
            with open(filename, "w") as file:
                file.write(board.fen())
            print(f"Game saved to {filename}.")
        except Exception as e:
            print(f"An error occurred while saving the game: {e}")
    else:
        print("Save operation cancelled.")

def load_game():
    global selected_square    # Reset selected_square
    # Open file dialog to select game file
    filename = filedialog.askopenfilename(
        title="Load Game",
        defaultextension=".fen",
        filetypes=[("FEN Files", "*.fen"), ("All Files", "*.*")]
    )
    if filename:
        try:
            with open(filename, "r") as file:
                fen = file.read()
                board.set_fen(fen)
                selected_square = None  # Reset selection
                print(f"Game loaded from {filename}.")
        except Exception as e:
            print(f"An error occurred while loading the game: {e}")
    else:
        print("Load operation cancelled.")

# Check for checkmate, stalemate, and check
def check_for_game_end():
    global autoplay_enabled  # Declare as global to modify the variable
    if board.is_checkmate():
        print("Checkmate! Game over.")
        autoplay_enabled = False
    elif board.is_stalemate():
        print("Stalemate! Game over.")
        autoplay_enabled = False
    elif board.is_insufficient_material():
        print("Draw due to insufficient material! Game over.")
        autoplay_enabled = False
    elif board.can_claim_threefold_repetition():
        print("Threefold repetition detected! Game over.")
        # Handle threefold repetition
        board.can_claim_threefold_repetition()
        print("Draw claimed by threefold repetition.")
        autoplay_enabled = False
    elif board.can_claim_fifty_moves():
        print("Fifty-move rule can be claimed! Game over.")
        # Handle fifty-move rule
        board.can_claim_fifty_moves()
        print("Draw claimed by fifty-move rule.")
        autoplay_enabled = False
    elif board.is_check():
        print("Check!")

# Detect button clicks
def handle_button_click(pos):
    x, y = pos
    if 10 <= x <= 110 and SCREEN_HEIGHT - 140 <= y <= SCREEN_HEIGHT - 100:
        undo_last_move()
    elif 130 <= x <= 230 and SCREEN_HEIGHT - 140 <= y <= SCREEN_HEIGHT - 100:
        display_move_history()
    elif 250 <= x <= 350 and SCREEN_HEIGHT - 140 <= y <= SCREEN_HEIGHT - 100:
        save_game()
    elif 370 <= x <= 470 and SCREEN_HEIGHT - 140 <= y <= SCREEN_HEIGHT - 100:
        load_game()
    elif 10 <= x <= 110 and SCREEN_HEIGHT - 90 <= y <= SCREEN_HEIGHT - 50:
        make_ai_move(white_engine_path if board.turn else black_engine_path)
    elif 130 <= x <= 230 and SCREEN_HEIGHT - 90 <= y <= SCREEN_HEIGHT - 50:
        select_white_engine()
    elif 250 <= x <= 350 and SCREEN_HEIGHT - 90 <= y <= SCREEN_HEIGHT - 50:
        select_black_engine()
    elif 370 <= x <= 470 and SCREEN_HEIGHT - 90 <= y <= SCREEN_HEIGHT - 50:
        toggle_autoplay()

# Main loop with AI move timing control
def main():
    global last_ai_move_time  # Declare as global to modify the variable
    global selected_square    # Declare as global
    selected_square = None
    running = True

    while running:
        current_time = time.time()

        legal_moves = get_legal_moves(selected_square)
        draw_board(selected_square, legal_moves)
        draw_buttons()
        draw_timer()
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                # Adjusted condition to correctly handle button clicks
                if pos[1] >= BOARD_SIZE * SQUARE_SIZE:
                    handle_button_click(pos)
                else:
                    selected_square = handle_click(pos, selected_square)

        # Control AI move timing
        if autoplay_enabled and (current_time - last_ai_move_time >= ai_move_delay):
            # Check if an engine is loaded for the side to move
            if (board.turn and white_engine_path) or (not board.turn and black_engine_path):
                autoplay()
                last_ai_move_time = current_time  # Update last AI move time

    pygame.quit()

if __name__ == "__main__":
    main()
