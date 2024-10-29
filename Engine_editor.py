import os
import PySimpleGUI as sg
import torch
import subprocess
import chess
import chess.engine

# Simple PyTorch Neural Network for Chess (Placeholder)
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

# Function to compile AI to a UCI-compatible engine and save as .exe
def compile_model_to_exe(scripted_model_path, output_name="chess_engine"):
    try:
        # Read the chess engine template
        with open('chess_engine_template.py', 'r') as template_file:
            wrapper_content = template_file.read()

        # Replace the placeholder with the actual scripted model path
        wrapper_content = wrapper_content.replace('{model_path}', scripted_model_path)

        # Save the modified wrapper as a new Python file
        wrapper_script_path = f"{output_name}.py"
        with open(wrapper_script_path, 'w') as wrapper_script_file:
            wrapper_script_file.write(wrapper_content)

        # Ensure full paths are provided to PyInstaller
        script_full_path = os.path.abspath(wrapper_script_path)

        # Run PyInstaller to compile the Python script into a .exe
        subprocess.run(["pyinstaller", "--onefile", script_full_path])

        # Remove the generated .py and .spec files after compilation
        if os.path.exists(wrapper_script_path):
            os.remove(wrapper_script_path)
        spec_file = f"{output_name}.spec"
        if os.path.exists(spec_file):
            os.remove(spec_file)

        sg.popup("Compilation successful!", title="Success")

    except Exception as e:
        sg.popup_error(f"Error during compilation: {e}")

def open_neural_network_window(ai):
    layout = [
        [sg.Text('Random Training', font=('Helvetica', 14))],
        [sg.Button('Random Training', key='-RANDOM-TRAINING-')],
        [sg.Text('Export AI Model', font=('Helvetica', 14))],
        [sg.Button('Export AI', key='-EXPORT-AI-')],
        [sg.Output(size=(60, 5), key='-NEURAL-OUTPUT-')]
    ]

    window = sg.Window('Neural Networks', layout)

    while True:
        event, values = window.read()
        if event == sg.WINDOW_CLOSED:
            break
        if event == '-RANDOM-TRAINING-':
            # Placeholder: Implement your random training logic here
            print('Random training in progress...')
        if event == '-EXPORT-AI-':
            if ai is not None:
                save_path = sg.popup_get_file('Save AI Model', save_as=True, default_extension='.pt', file_types=(("Model Files", "*.pt"), ("All Files", "*.*")))
                if save_path:
                    if not save_path.endswith('.pt'):
                        save_path += '.pt'
                    torch.save(ai.state_dict(), save_path)
                    print(f'Model exported successfully to {save_path}')
            else:
                print('No AI model to export.')

    window.close()

# GUI layout
layout = [
    [sg.Text('Chess AI Configuration', font=('Helvetica', 16))],
    [sg.Text('Learning Rate'), sg.InputText('0.001', key='-LR-', size=(10, 1))],
    [sg.Text('Load AI Model'), sg.InputText('', key='-MODEL-', size=(30, 1)), sg.FileBrowse(file_types=(("Model Files", "*.pt"), ("All Files", "*.*")))],
    [sg.Button('Load AI', key='-LOAD-')],
    [sg.Button('Create Model', key='-CREATE-')],
    [sg.Text('Select Stockfish Engine'), sg.InputText('', key='-STOCKFISH-', size=(30, 1)), sg.FileBrowse(file_types=(("Executable Files", "*.exe"), ("All Files", "*.*")))],
    [sg.Button('Set Stockfish Path', key='-SET-STOCKFISH-')],
    [sg.Button('Compile AI Model to Engine', key='-COMPILE-')],
    [sg.Button('Neural Networks', key='-NEURAL-NETWORK-')],
    [sg.Output(size=(60, 5), key='-MAIN-OUTPUT-')]
]

window = sg.Window('Chess AI Trainer', layout)

# Initialize AI and Stockfish engine path
ai = None
optimizer = None
engine_path = None  # Set as None initially

# Main event loop
while True:
    event, values = window.read()

    if event == sg.WINDOW_CLOSED:
        break

    # Load AI model
    if event == '-LOAD-':
        model_path = values['-MODEL-']
        if model_path:
            try:
                ai = ChessAI()
                ai.load_state_dict(torch.load(model_path, weights_only=True))
                optimizer = torch.optim.Adam(ai.parameters(), lr=float(values['-LR-']))
                sg.popup('Model Loaded Successfully')
            except Exception as e:
                sg.popup(f"Error loading model: {e}")

    # Create new AI model and save to file
    if event == '-CREATE-':
        try:
            ai = ChessAI()
            optimizer = torch.optim.Adam(ai.parameters(), lr=float(values['-LR-']))
            save_path = sg.popup_get_file('Save Model', save_as=True, default_extension='.pt', file_types=(("Model Files", "*.pt"), ("All Files", "*.*")))
            if save_path:
                if not save_path.endswith('.pt'):
                    save_path += '.pt'
                torch.save(ai.state_dict(), save_path)
                sg.popup(f"New model created and saved successfully to {save_path}")
        except Exception as e:
            sg.popup(f"Error creating or saving model: {e}")

    # Set Stockfish engine path
    if event == '-SET-STOCKFISH-':
        engine_path = values['-STOCKFISH-']
        if engine_path:
            sg.popup(f"Stockfish engine path set to: {engine_path}")
        else:
            sg.popup("Please select a valid Stockfish engine file.")

    # Compile AI model to a UCI-compatible engine
    if event == '-COMPILE-':
        model_path = values['-MODEL-']
        if model_path:
            output_name = sg.popup_get_text('Enter output file name (without extension):', default_text="chess_engine")
            if output_name:
                compile_model_to_exe(model_path, output_name)
                sg.popup(f"Model compiled to {output_name}.exe")

    # Open Neural Network window
    if event == '-NEURAL-NETWORK-':
        open_neural_network_window(ai)

window.close()



