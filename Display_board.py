import pygame
import chess
import chess.engine
import time
from tkinter import filedialog, Tk

# Initialize pygame and hide tkinter root window
pygame.init()
Tk().withdraw()  # Hide the main tkinter window

# Constants
SQUARE_SIZE = 60
BOARD_SIZE = 8
SCREEN_WIDTH = BOARD_SIZE * SQUARE_SIZE
SCREEN_HEIGHT = BOARD_SIZE * SQUARE_SIZE + 150# Extra space for buttons and timer
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
autoplay_active = False

# Timer setup
start_time = time.time()

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

            # Highlight legal moves for the selected piece
            for move in legal_moves:
                if move.to_square == chess.square(col, 7 - row):
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
        ("Undo", (10, SCREEN_HEIGHT - 130)),         # Row 1, Button 1
        ("History", (130, SCREEN_HEIGHT - 130)),      # Row 1, Button 2
        ("Save", (250, SCREEN_HEIGHT - 130)),         # Row 1, Button 3
        ("Load", (370, SCREEN_HEIGHT - 130)),         # Row 1, Button 4
        ("AI Move", (10, SCREEN_HEIGHT - 80)),        # Row 2, Button 1
        ("White AI", (130, SCREEN_HEIGHT - 80)),      # Row 2, Button 2
        ("Black AI", (250, SCREEN_HEIGHT - 80)),      # Row 2, Button 3
        ("Autoplay", (370, SCREEN_HEIGHT - 80))       # Row 2, Button 4
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

# Get legal moves for the selected piece
def get_legal_moves(selected_square):
    if selected_square is None:
        return []
    return [move for move in board.legal_moves if move.from_square == selected_square]

# Handle square selection and moves
def handle_click(position, selected_square):
    col, row = position[0] // SQUARE_SIZE, 7 - (position[1] // SQUARE_SIZE)
    clicked_square = chess.square(col, row)
    piece = board.piece_at(clicked_square)

    if selected_square is None:
        if piece and piece.color == board.turn:
            return clicked_square
    else:
        move = chess.Move(selected_square, clicked_square)
        if move in board.legal_moves:
            board.push(move)
            check_for_game_end()
        return None
    return selected_square

# Undo last move
def undo_last_move():
    if board.move_stack:
        board.pop()

# Display move history
def display_move_history():
    if board.move_stack:
        move_history = [board.san(move) for move in board.move_stack]
        print("Move History:", " ".join(move_history))

# Save and load game
def save_game():
    with open("saved_game.fen", "w") as file:
        file.write(board.fen())

def load_game():
    try:
        with open("saved_game.fen", "r") as file:
            fen = file.read()
            board.set_fen(fen)
    except FileNotFoundError:
        print("No saved game found.")

# AI Move function
def make_ai_move(engine_path):
    if engine_path:
        try:
            engine = chess.engine.SimpleEngine.popen_uci(engine_path)
            result = engine.play(board, chess.engine.Limit(time=2.0))
            board.push(result.move)
            engine.quit()
        except Exception as e:
            print("Failed to run AI engine:", e)

# Select engine paths
def select_white_engine():
    global white_engine_path
    white_engine_path = filedialog.askopenfilename(title="Select White Engine")

def select_black_engine():
    global black_engine_path
    black_engine_path = filedialog.askopenfilename(title="Select Black Engine")

# Autoplay between two engines
def toggle_autoplay():
    global autoplay_active
    autoplay_active = not autoplay_active
    print("Autoplay is now", "active" if autoplay_active else "inactive")

# Check for checkmate, stalemate, and check
def check_for_game_end():
    if board.is_checkmate():
        print("Checkmate! Game over.")
    elif board.is_stalemate():
        print("Stalemate! Game over.")
    elif board.is_check():
        print("Check!")

# Detect button clicks
def handle_button_click(pos):
    x, y = pos
    if 10 <= x <= 110 and SCREEN_HEIGHT - 130 <= y <= SCREEN_HEIGHT - 90:
        undo_last_move()
    elif 130 <= x <= 230 and SCREEN_HEIGHT - 130 <= y <= SCREEN_HEIGHT - 90:
        display_move_history()
    elif 250 <= x <= 350 and SCREEN_HEIGHT - 130 <= y <= SCREEN_HEIGHT - 90:
        save_game()
    elif 370 <= x <= 470 and SCREEN_HEIGHT - 130 <= y <= SCREEN_HEIGHT - 90:
        load_game()
    elif 10 <= x <= 110 and SCREEN_HEIGHT - 80 <= y <= SCREEN_HEIGHT - 40:
        make_ai_move()
    elif 130 <= x <= 230 and SCREEN_HEIGHT - 80 <= y <= SCREEN_HEIGHT - 40:
        select_white_engine()
    elif 250 <= x <= 350 and SCREEN_HEIGHT - 80 <= y <= SCREEN_HEIGHT - 40:
        select_black_engine()
    elif 370 <= x <= 470 and SCREEN_HEIGHT - 80 <= y <= SCREEN_HEIGHT - 40:
        toggle_autoplay()



# Main loop
def main():
    global autoplay_active
    selected_square = None
    running = True
    while running:
        legal_moves = get_legal_moves(selected_square)
        draw_board(selected_square, legal_moves)
        draw_pieces()
        draw_buttons()
        draw_timer()
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                if pos[1] >= SCREEN_HEIGHT - 100:
                    handle_button_click(pos)
                else:
                    selected_square = handle_click(pos, selected_square)

        # Autoplay between engines
        if autoplay_active:
            current_engine_path = white_engine_path if board.turn == chess.WHITE else black_engine_path
            if current_engine_path:
                make_ai_move(current_engine_path)
            else:
                autoplay_active = False
                print("Autoplay stopped - no engine selected for current turn.")

    pygame.quit()

if __name__ == "__main__":
    main()
