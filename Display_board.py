import PySimpleGUI as sg
import pygame
import chess

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


def draw_chessboard():
    """Draws a chessboard using pygame and returns the surface."""
    board_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))

    # Fill the board background
    board_surface.fill(DEFAULT_BG_COLOR)

    # Draw squares
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            color = WHITE if (row + col) % 2 == 0 else DARK_AQUA
            pygame.draw.rect(board_surface, color,
                             pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

    return board_surface


def draw_pieces(surface):
    """Draws pieces on the board according to the current state with improved quality."""
    piece_images = {
        'P': pygame.image.load('pieces/black-pawn.png').convert_alpha(),
        'N': pygame.image.load('pieces/black-knight.png').convert_alpha(),
        'B': pygame.image.load('pieces/black-bishop.png').convert_alpha(),
        'R': pygame.image.load('pieces/black-rook.png').convert_alpha(),
        'Q': pygame.image.load('pieces/black-queen.png').convert_alpha(),
        'K': pygame.image.load('pieces/black-king.png').convert_alpha(),
        'p': pygame.image.load('pieces/white-pawn.png').convert_alpha(),
        'n': pygame.image.load('pieces/white-knight.png').convert_alpha(),
        'b': pygame.image.load('pieces/white-bishop.png').convert_alpha(),
        'r': pygame.image.load('pieces/white-rook.png').convert_alpha(),
        'q': pygame.image.load('pieces/white-queen.png').convert_alpha(),
        'k': pygame.image.load('pieces/white-king.png').convert_alpha(),
    }

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_image = piece_images[piece.symbol()]

            # Scale the piece to fit within the square with anti-aliasing
            piece_image = pygame.transform.scale(piece_image, (SQUARE_SIZE, SQUARE_SIZE))

            # Use a smoothing function if available
            # piece_image = pygame.transform.smoothscale(piece_image, (SQUARE_SIZE, SQUARE_SIZE))

            row, col = divmod(square, 8)
            surface.blit(piece_image, (col * SQUARE_SIZE, row * SQUARE_SIZE))
def get_square_from_mouse(pos):
    """Get the square index from the mouse position."""
    x, y = pos
    col = x // SQUARE_SIZE
    row = y // SQUARE_SIZE
    return chess.square(col, row)


def main():
    layout = [[sg.Image(filename='', key='-BOARD-')]]
    window = sg.Window("Interactive Chessboard", layout, finalize=True)

    while True:
        event, values = window.read(timeout=100)

        if event in (sg.WIN_CLOSED, 'Exit'):
            break

        # Draw the board
        board_surface = draw_chessboard()
        draw_pieces(board_surface)

        # Save the board surface to a temporary file to display in PySimpleGUI
        pygame.image.save(board_surface, "chessboard.png")
        window['-BOARD-'].update(filename="chessboard.png")

        # Handle mouse clicks
        if event == '-BOARD-':
            mouse_pos = values['-BOARD-']
            if mouse_pos:  # Check if mouse position is valid
                square = get_square_from_mouse(mouse_pos)
                print(f'Mouse clicked on square: {chess.square_name(square)}')

                # Example interaction: Move a piece
                # This example will need further logic for selecting and moving pieces.
                # You could store the selected piece and handle moving logic based on user clicks.

    window.close()
    pygame.quit()


if __name__ == "__main__":
    main()