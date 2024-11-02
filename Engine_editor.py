import os
import subprocess
import sqlite3
import datetime
import threading

import time
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import PySimpleGUI as sg

import chess


# Simple PyTorch Neural Network for Chess (Placeholder)
class ChessAI(torch.nn.Module):
    def __init__(self, input_size=65, hidden_layer_sizes=[130, 65], output_size=1, activation_func='ReLU'):
        super(ChessAI, self).__init__()
        self.layers = torch.nn.ModuleList()

        # Set up the layers according to the provided layer sizes
        in_features = input_size
        for hidden_size in hidden_layer_sizes:
            self.layers.append(torch.nn.Linear(in_features, hidden_size))
            in_features = hidden_size

        # Add the final output layer
        self.layers.append(torch.nn.Linear(in_features, output_size))

        # Set activation function with default to ReLU
        self.set_activation_function(activation_func)

    def forward(self, board_tensor):
        x = board_tensor
        for layer in self.layers[:-1]:  # Apply activation to all layers except the last one
            x = self.activation_func(layer(x))
        x = torch.sigmoid(self.layers[-1](x))  # Use sigmoid for the output layer
        return x

    def set_activation_function(self, activation_func):
        """Set the activation function dynamically."""
        if activation_func == 'ReLU':
            self.activation_func = torch.nn.ReLU()
        elif activation_func == 'LeakyReLU':
            self.activation_func = torch.nn.LeakyReLU()
        elif activation_func == 'Tanh':
            self.activation_func = torch.nn.Tanh()
        elif activation_func == 'Sigmoid':
            self.activation_func = torch.nn.Sigmoid()
        else:
            raise ValueError("Unsupported activation function. Choose from 'ReLU', 'LeakyReLU', 'Tanh', or 'Sigmoid'.")
        self.activation_func_name = activation_func

    # Method to dynamically configure layers based on the state_dict
    def configure_layers(self, state_dict):
        # Extract layer shapes from 2D tensors in state_dict
        layer_sizes = [param.shape for param in state_dict.values() if param.ndim == 2]

        if not layer_sizes:
            raise ValueError("No valid layer sizes inferred from state_dict.")

        # Clear existing layers and reinitialize as a ModuleList
        self.layers = torch.nn.ModuleList()

        # Configure each layer based on inferred sizes
        in_features = layer_sizes[0][1]  # Input size from the first layer
        for out_features in [size[0] for size in layer_sizes]:
            self.layers.append(torch.nn.Linear(in_features, out_features))
            in_features = out_features

        # Load the model weights into the configured layers
        self.load_state_dict(state_dict)


class ChessDataset(Dataset):
    def __init__(self, csv_file, normalize=True):
        self.data = pd.read_csv(csv_file)

        # Process the 'Evaluation' column
        self.data['Evaluation'] = self.data['Evaluation'].apply(self.process_evaluation)

        # Drop any rows where the evaluation couldn't be converted to a number
        self.data = self.data.dropna(subset=['Evaluation'])
        print(f"Loaded {len(self.data)} entries from the dataset.")

        self.normalize = normalize

    def process_evaluation(self, eval_str):
        if isinstance(eval_str, str) and '#' in eval_str:
            return 10000 if '+' in eval_str else -10000
        else:
            return float(eval_str)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Assume 'FEN' is the column containing the board position in FEN format
        board_fen = row['FEN']  # Replace 'FEN' with the actual column name in your CSV
        board_tensor = board_to_tensor(board_fen)

        evaluation = row['Evaluation']

        # Normalize evaluation if needed
        if self.normalize and evaluation not in [float('inf'), float('-inf')]:
            evaluation = evaluation / 100  # Adjust normalization as required

        return board_tensor, torch.tensor(evaluation, dtype=torch.float32)
# Database setup
def create_database():
    connection = sqlite3.connect("chess_ai_training.db")
    cursor = connection.cursor()

    # Table for storing model configurations
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS model_configurations (
        config_id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT,
        device TEXT,
        learning_rate REAL,
        batch_size INTEGER,
        epochs INTEGER,
        optimizer_type TEXT,
        weight_decay REAL,
        loss_function TEXT,
        dropout_enabled INTEGER,
        weight_init_method TEXT,
        training_method TEXT,
        use_early_stopping INTEGER,
        data_source TEXT,
        use_engine_data INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_updated TIMESTAMP
    )
    """)

    # Table for tracking individual training sessions
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS training_sessions (
        session_id INTEGER PRIMARY KEY AUTOINCREMENT,
        config_id INTEGER,
        model_id INTEGER,
        start_time TIMESTAMP,
        end_time TIMESTAMP,
        training_duration REAL,
        final_loss REAL,
        FOREIGN KEY(config_id) REFERENCES model_configurations(config_id),
        FOREIGN KEY(model_id) REFERENCES model_performance(model_id)
    )
    """)

    # Table for overall model performance
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS model_performance (
        model_id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT,
        total_training_time REAL DEFAULT 0,
        dataset_used TEXT,
        engine_used TEXT,
        last_training_loss REAL,
        created_on TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Table for tracking loss history
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS loss_history (
        history_id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_id INTEGER,
        epoch INTEGER,
        loss REAL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(model_id) REFERENCES model_performance(model_id)
    )
    """)

    connection.commit()
    connection.close()

# Save model configuration function
def save_model_configuration(model_name, device, learning_rate, batch_size, epochs, optimizer_type,
                             weight_decay, loss_function, dropout_enabled, weight_init_method,
                             training_method, use_early_stopping, data_source, use_engine_data):
    connection = sqlite3.connect("chess_ai_training.db")
    cursor = connection.cursor()

    # Insert configuration
    cursor.execute("""
    INSERT INTO model_configurations (
        model_name, device, learning_rate, batch_size, epochs, optimizer_type, weight_decay,
        loss_function, dropout_enabled, weight_init_method, training_method, 
        use_early_stopping, data_source, use_engine_data, last_updated
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        model_name, device, learning_rate, batch_size, epochs, optimizer_type, weight_decay,
        loss_function, int(dropout_enabled), weight_init_method, training_method,
        int(use_early_stopping), data_source, int(use_engine_data),
        datetime.datetime.now()
    ))

    config_id = cursor.lastrowid
    connection.commit()
    connection.close()

    return config_id
def record_to_db(action, timestamp=None, training_time="00:00:00", final_loss=0.0):
    # Connect to the SQLite database
    conn = sqlite3.connect("model_training.db")
    cursor = conn.cursor()

    # Ensure we have a valid timestamp
    if not timestamp:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create table if it does not exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS TrainingLog (
            id INTEGER PRIMARY KEY,
            action TEXT,
            timestamp TEXT,
            training_time TEXT,
            final_loss REAL
        )
    ''')

    # Insert the log entry into the TrainingLog table
    cursor.execute('''
        INSERT INTO TrainingLog (action, timestamp, training_time, final_loss)
        VALUES (?, ?, ?, ?)
    ''', (action, timestamp, training_time, final_loss))

    # Commit the transaction and close the connection
    conn.commit()
    conn.close()

    print(f"Recorded to database: Action={action}, Timestamp={timestamp}, Training Time={training_time}, Final Loss={final_loss}")
# Log training session function
def log_training_session(config_id, model_id, start_time, end_time, final_loss):
    connection = sqlite3.connect("chess_ai_training.db")
    cursor = connection.cursor()

    training_duration = (end_time - start_time).total_seconds() / 3600.0  # duration in hours

    # Insert training session data
    cursor.execute("""
    INSERT INTO training_sessions (
        config_id, model_id, start_time, end_time, training_duration, final_loss
    ) VALUES (?, ?, ?, ?, ?, ?)
    """, (config_id, model_id, start_time, end_time, training_duration, final_loss))

    # Update total training time in model_performance
    cursor.execute("""
    UPDATE model_performance
    SET total_training_time = total_training_time + ?
    WHERE model_id = ?
    """, (training_duration, model_id))

    connection.commit()
    connection.close()

# Save performance loss for history tracking
def save_loss_history(model_id, epoch, loss):
    connection = sqlite3.connect("chess_ai_training.db")
    cursor = connection.cursor()

    # Insert loss history
    cursor.execute("""
    INSERT INTO loss_history (model_id, epoch, loss)
    VALUES (?, ?, ?)
    """, (model_id, epoch, loss))

    # Update latest training loss in model_performance
    cursor.execute("""
    UPDATE model_performance
    SET last_training_loss = ?
    WHERE model_id = ?
    """, (loss, model_id))

    connection.commit()
    connection.close()

# Add model to model_performance
def add_model(model_name, dataset_used, engine_used):
    connection = sqlite3.connect("chess_ai_training.db")
    cursor = connection.cursor()

    cursor.execute("""
    INSERT INTO model_performance (model_name, dataset_used, engine_used)
    VALUES (?, ?, ?)
    """, (model_name, dataset_used, engine_used))

    model_id = cursor.lastrowid
    connection.commit()
    connection.close()

    return model_id

# Example usage
create_database()

# Save a model configuration
config_id = save_model_configuration(
    model_name="ChessAI_Model_1",
    device="GPU",
    learning_rate=0.001,
    batch_size=64,
    epochs=10,
    optimizer_type="Adam",
    weight_decay=0.0001,
    loss_function="MSELoss",
    dropout_enabled=True,
    weight_init_method="Xavier",
    training_method="Supervised",
    use_early_stopping=True,
    data_source="custom_data.csv",
    use_engine_data=False
)

# Log a training session
start_time = datetime.datetime.now()
end_time = start_time + datetime.timedelta(hours=2)
log_training_session(config_id=config_id, model_id=1, start_time=start_time, end_time=end_time, final_loss=0.05)

# Save a loss history entry
save_loss_history(model_id=1, epoch=1, loss=0.1)
save_loss_history(model_id=1, epoch=2, loss=0.08)
#=========================================================================================

#=========================================================================================
# Function to compile AI to a UCI-compatible engine and save as .exe
def compile_model_to_exe(scripted_model_path, output_name="chess_engine"):
    try:
        with open('chess_engine_template.py', 'r') as template_file:
            wrapper_content = template_file.read()

        wrapper_content = wrapper_content.replace('{model_path}', scripted_model_path)
        wrapper_script_path = f"{output_name}.py"
        with open(wrapper_script_path, 'w') as wrapper_script_file:
            wrapper_script_file.write(wrapper_content)

        script_full_path = os.path.abspath(wrapper_script_path)
        subprocess.run(["pyinstaller", "--onefile", script_full_path])

        if os.path.exists(wrapper_script_path):
            os.remove(wrapper_script_path)
        spec_file = f"{output_name}.spec"
        if os.path.exists(spec_file):
            os.remove(spec_file)

        sg.popup("Compilation successful!", title="Success")

    except Exception as e:
        sg.popup_error(f"Error during compilation: {e}")



def load_chess_ai(model_path):
    checkpoint = torch.load(model_path)
    model = ChessAI()
    model.configure_layers(checkpoint["model_state_dict"])  # Configure layers from weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model
def open_training_window(ai):
    cuda_status = "Available" if torch.cuda.is_available() else "Not Available"
    layout = [
        [sg.Text('Training Configuration', font=('Helvetica', 16))],

        # Device Section
        [sg.Frame('Device Settings', [
            [sg.Text('Device', tooltip="Select the hardware for training (CPU or GPU if available)"),
             sg.Combo(['CPU', 'GPU'], default_value='CPU', key='-DEVICE-', size=(10, 1)),
             sg.Text(f"CUDA: {cuda_status}", key='-CUDA-STATUS-',
                     text_color="green" if torch.cuda.is_available() else "red")]
        ])],

        # Data Source Section
        [sg.Frame('Data Source', [
            [sg.Text('Training Data Source',
                     tooltip="Specify the dataset for training or use a chess engine as source."),
             sg.InputText('', key='-DATA-SOURCE-', size=(30, 1)),
             sg.FileBrowse(file_types=(("Data Files", "*.csv *.txt *.json"), ("All Files", "*.*")),
                           key='-DATA-BROWSE-')],
            [sg.Checkbox('Use Engine as Data Source', key='-USE-ENGINE-DATA-', default=False,
                         tooltip="Use moves from a chess engine as training data.")],
            [sg.Checkbox('Apply Data Normalization', key='-NORMALIZATION-', default=True,
                         tooltip="Applies normalization to scale input data to a common range.")]
        ])],

        # Model Settings Section
        [sg.Frame('Model Settings', [
            [sg.Checkbox('Enable Dropout (20%)', key='-DROPOUT-', default=True,
                         tooltip="Randomly ignores 20% of neurons during training to prevent overfitting.")],
            [sg.Text('Weight Initialization', tooltip="Method to initialize weights in the neural network."),
             sg.Combo(['Xavier', 'He', 'Uniform', 'Normal'], default_value='Xavier', key='-WEIGHT-INIT-', size=(10, 1))]
        ])],

        # Training Hyperparameters Section
        [sg.Frame('Training Hyperparameters', [
            [sg.Text('Learning Rate', tooltip="Controls the step size during gradient descent."),
             sg.InputText('0.001', key='-LEARNING-RATE-', size=(10, 1), enable_events=True)],
            [sg.Text('Batch Size', tooltip="Number of samples per training batch."),
             sg.InputText('32', key='-BATCH-SIZE-', size=(10, 1), enable_events=True)],
            [sg.Text('Number of Epochs', tooltip="Number of times the entire dataset is passed through the model."),
             sg.InputText('10', key='-EPOCHS-', size=(10, 1), enable_events=True)],
        ])],

        # Optimizer Settings Section
        [sg.Frame('Optimizer Settings', [
            [sg.Text('Optimizer Type', tooltip="Algorithm for updating model weights during training."),
             sg.Combo(['Adam', 'SGD', 'RMSprop'], default_value='Adam', key='-OPTIMIZER-', size=(10, 1))],
            [sg.Text('Weight Decay', tooltip="Regularization parameter to penalize large weights."),
             sg.InputText('0.0', key='-WEIGHT-DECAY-', size=(10, 1), enable_events=True)]
        ])],

        # Loss Function Section
        [sg.Frame('Loss Function', [
            [sg.Text('Loss Function', tooltip="Function that measures the model's error."),
             sg.Combo(['MSELoss', 'CrossEntropyLoss'], default_value='MSELoss', key='-LOSS-FUNCTION-', size=(15, 1))]
        ])],

        # Training Method and Early Stopping Section
        [sg.Frame('Training Method', [
            [sg.Text('Training Method', tooltip="Strategy to optimize the model (e.g., supervised, reinforcement)."),
             sg.Combo(['Supervised', 'Reinforcement', 'Genetic'], default_value='Supervised', key='-TRAINING-METHOD-',
                      size=(15, 1))],
            [sg.Checkbox('Use Early Stopping', key='-EARLY-STOPPING-', default=False,
                         tooltip="Stops training when validation loss stops improving.")]
        ])],

        # Training Control Buttons
        [sg.Button('Start Training', key='-START-TRAINING-')],
        [sg.Output(size=(60, 10), key='-TRAINING-OUTPUT-')]
    ]

    window = sg.Window('Training Settings', layout)

    # Enforce numeric-only input on specific fields
    def numeric_input(event, key):
        if event in [key] and not values[event].replace('.', '', 1).isdigit():
            window[event].update(values[event][:-1])

    while True:
        event, values = window.read()

        # Enforce numeric-only input for fields
        numeric_input(event, '-LEARNING-RATE-')
        numeric_input(event, '-BATCH-SIZE-')
        numeric_input(event, '-EPOCHS-')
        numeric_input(event, '-WEIGHT-DECAY-')

        if event == sg.WINDOW_CLOSED:
            break

        if event == '-START-TRAINING-':
            try:
                # Fetch and parse hyperparameters
                device = values['-DEVICE-']
                normalize_data = values['-NORMALIZATION-']
                use_dropout = values['-DROPOUT-']
                weight_init = values['-WEIGHT-INIT-']

                learning_rate = float(values['-LEARNING-RATE-'])
                batch_size = int(values['-BATCH-SIZE-'])
                epochs = int(values['-EPOCHS-'])
                optimizer_type = values['-OPTIMIZER-']
                weight_decay = float(values['-WEIGHT-DECAY-'])
                loss_function = values['-LOSS-FUNCTION-']
                use_early_stopping = values['-EARLY-STOPPING-']
                training_method = values['-TRAINING-METHOD-']
                data_source = values['-DATA-SOURCE-']
                use_engine_data = values['-USE-ENGINE-DATA-']
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                print("===================================================")
                print("")
                print("===================================================")
                print("Training Configuration:")
                print(f"Learning Rate: {learning_rate}")
                print(f"Batch Size: {batch_size}")
                print(f"Epochs: {epochs}")
                print(f"Optimizer: {optimizer_type}")
                print(f"Weight Decay: {weight_decay}")
                print(f"Loss Function: {loss_function}")
                print(f"Dropout Enabled: {'Yes' if use_dropout else 'No'}")
                print(f"Early Stopping: {'Enabled' if use_early_stopping else 'Disabled'}")
                print(f"Training Method: {training_method}")
                print(f"Data Source: {data_source if not use_engine_data else 'Engine data'}")
                device = 'cuda' if device == 'GPU' and torch.cuda.is_available() else 'cpu'
                ai.to(device)
                print(f"Training on device: {device}")
                print("===================================================")
                print("")
                print("===================================================")
                # Device setup


                # Placeholder function calls for data processing and training
                data_loader = load_training_data(data_source, batch_size, normalize_data)
                apply_weight_initialization(ai, weight_init)
                optimizer = get_optimizer(optimizer_type, ai, learning_rate, weight_decay)
                criterion = get_loss_function(loss_function)

                # Display configuration in output


                # starting the actual training process
                train_thread = threading.Thread(
                    target=threaded_training,
                    args=(
                    ai, data_loader, optimizer, criterion, epochs, use_dropout, use_early_stopping, device, model_path),
                    daemon=True
                )
                train_thread.start()

            except ValueError as ve:
                print(f"Dataset error: {ve}")
            except Exception as e:
                print(f"Unexpected error: {e}")

    window.close()
# Placeholder functions to customize training logic later
def load_training_data(data_source, batch_size, normalize):
    dataset = ChessDataset(data_source, normalize=normalize)
    if len(dataset) == 0:
        raise ValueError("The dataset is empty or could not be loaded.")
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def apply_weight_initialization(model, method):
    """Initialize weights of the model according to the selected method."""

    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            if method == 'Xavier':
                torch.nn.init.xavier_uniform_(m.weight)
            elif method == 'He':
                torch.nn.init.kaiming_uniform_(m.weight)
            elif method == 'Uniform':
                torch.nn.init.uniform_(m.weight)
            elif method == 'Normal':
                torch.nn.init.normal_(m.weight)

    model.apply(init_weights)


def get_optimizer(optimizer_type, model, learning_rate, weight_decay):
    """Create optimizer based on the selected type and parameters."""
    if optimizer_type == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == 'RMSprop':
        return torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def get_loss_function(loss_function_type):
    """Return the loss function object based on the selection."""
    if loss_function_type == 'MSELoss':
        return torch.nn.MSELoss()
    elif loss_function_type == 'CrossEntropyLoss':
        return torch.nn.CrossEntropyLoss()


def threaded_training(ai, data_loader, optimizer, criterion, epochs, use_dropout, use_early_stopping, device,
                      model_path, mini_epoch_size=1000):
    ai.train()
    best_loss = float('inf')
    patience = 3
    no_improvement_epochs = 0
    clip_value = 1.0  # Gradient clipping value

    for epoch in range(epochs):
        total_loss = 0.0
        if use_dropout:
            ai.train()  # Enable dropout during training
        else:
            ai.eval()  # Disable dropout during training

        mini_epoch_loss = 0.0
        mini_epoch_count = 0
        batch_counter = 0

        for batch_idx, (board_tensors, evaluations) in enumerate(data_loader):
            board_tensors, evaluations = board_tensors.to(device), evaluations.to(device)

            # Forward pass, backward pass, and optimization step
            optimizer.zero_grad()
            outputs = ai(board_tensors).squeeze(-1)
            loss = criterion(outputs, evaluations)
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(ai.parameters(), clip_value)
            optimizer.step()

            total_loss += loss.item()
            mini_epoch_loss += loss.item()
            batch_counter += 1
            mini_epoch_count += 1

            # Print progress every 10 batches
            if batch_idx % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(data_loader)}], Loss: {loss.item():.4f}")

            # Save checkpoint at mini-epoch intervals
            if mini_epoch_count >= mini_epoch_size:
                avg_mini_epoch_loss = mini_epoch_loss / mini_epoch_count
                print(f"Mini-epoch completed. Average Loss: {avg_mini_epoch_loss:.4f}")

                # Save model checkpoint
                save_model(ai, model_path)
                mini_epoch_loss = 0.0
                mini_epoch_count = 0

        # Calculate average loss for full epoch
        avg_loss = total_loss / batch_counter
        print(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}")

        # Check if this is the best model so far
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_model(ai, model_path)  # Save the best model
            print(f"New best model saved with loss {best_loss:.4f}")
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1

        # Early stopping if no improvement for `patience` epochs
        if use_early_stopping and no_improvement_epochs >= patience:
            print(f"Early stopping triggered. No improvement in last {patience} epochs.")
            break

    print("Training complete.")
    print(f"Best model saved with loss: {best_loss:.4f} at '{model_path}'")
def board_to_tensor(fen):
    # Split the FEN string to extract the board position and active color
    parts = fen.split()
    board_fen = parts[0].replace('/', '')  # First part is the board layout, with '/' removed
    active_color = parts[1]  # Second part is the active color

    # Initialize a tensor with 65 elements: 64 squares + 1 for turn indicator
    tensor = torch.zeros(65)
    index = 0

    # Mapping each piece to an integer value
    piece_map = {
        'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,  # White pieces
        'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6  # Black pieces
    }

    # Populate the tensor with board pieces
    for square in board_fen:
        if square.isdigit():
            index += int(square)  # Skip empty squares
        else:
            if index >= 64:
                raise ValueError("FEN string has more than 64 squares.")
            tensor[index] = piece_map.get(square, 0)
            index += 1

    # Set the turn indicator: +1 if white's turn, -1 if black's turn
    tensor[64] = 1 if active_color == 'w' else -1

    return tensor


def save_model(ai, model_path):
    """Save model state and configuration to the specified path."""
    torch.save({
        'model_state_dict': ai.state_dict(),
        'activation_func': ai.activation_func_name  # Save the activation function as a string
    }, model_path)
    print(f"Model saved successfully with activation function '{ai.activation_func_name}' at {model_path}.")


def load_model(ai, model_path):
    """Load model state and configuration from the specified path."""
    checkpoint = torch.load(model_path)

    # Load the model's parameters
    ai.load_state_dict(checkpoint['model_state_dict'])

    # Retrieve and set the activation function
    activation_func_name = checkpoint['activation_func']
    if activation_func_name == 'ReLU':
        ai.activation_func = nn.ReLU()
    elif activation_func_name == 'LeakyReLU':
        ai.activation_func = nn.LeakyReLU()
    elif activation_func_name == 'Tanh':
        ai.activation_func = nn.Tanh()
    elif activation_func_name == 'Sigmoid':
        ai.activation_func = nn.Sigmoid()
    else:
        raise ValueError(f"Unsupported activation function '{activation_func_name}' in checkpoint.")

    print(f"Model loaded successfully with activation function '{activation_func_name}' from {model_path}.")

def adjust_learning_rate(optimizer, decay_factor=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay_factor
    print("Learning rate adjusted.")


def open_neural_network_window(ai, model_path):
    # Load existing model configuration if the file exists, with weights_only=True
    try:
        checkpoint = torch.load(model_path, weights_only=True)
        ai.configure_layers(checkpoint['model_state_dict'])
        # Map the saved activation function string to the actual function
        activation_func_name = checkpoint['activation_func']
        if activation_func_name == 'ReLU':
            ai.activation_func = nn.ReLU()
        elif activation_func_name == 'LeakyReLU':
            ai.activation_func = nn.LeakyReLU()
        elif activation_func_name == 'Tanh':
            ai.activation_func = nn.Tanh()
        elif activation_func_name == 'Sigmoid':
            ai.activation_func = nn.Sigmoid()
        print(f"Model configuration loaded successfully from {model_path}.")
    except (FileNotFoundError, KeyError):
        print("No previous model configuration file found or invalid format. Using default configuration.")

    # Extract current configuration values from `ai`
    input_layer_size = ai.layers[0].in_features if ai.layers else 64
    output_layer_size = ai.layers[-1].out_features if ai.layers else 1
    hidden_layer_sizes = [layer.out_features for layer in ai.layers[1:-1] if isinstance(layer, nn.Linear)]
    num_hidden_layers = len(hidden_layer_sizes)
    hidden_layer_size = hidden_layer_sizes[0] if hidden_layer_sizes else 64  # Default if no hidden layers

    # Extract activation function and dropout status
    activation_func = getattr(ai.activation_func, '__class__', nn.ReLU).__name__
    add_dropout = any(isinstance(layer, nn.Dropout) for layer in ai.layers)

    # Create GUI layout with fetched values
    layout = [
        [sg.Text('Neural Network Configuration', font=('Helvetica', 16))],
        [sg.Text('Neurons in Input Layer'),
         sg.InputText(input_layer_size, key='-INPUT-LAYER-', size=(10, 1), readonly=True)],
        [sg.Text('Number of Hidden Layers'), sg.InputText(num_hidden_layers, key='-NUM-HIDDEN-LAYERS-', size=(10, 1))],
        [sg.Text('Neurons per Hidden Layer'), sg.InputText(hidden_layer_size, key='-HIDDEN-LAYER-SIZE-', size=(10, 1))],
        [sg.Text('Neurons in Output Layer'),
         sg.InputText(output_layer_size, key='-OUTPUT-LAYER-', size=(10, 1), readonly=True)],
        [sg.Text('Activation Function'),
         sg.Combo(['ReLU', 'LeakyReLU', 'Tanh', 'Sigmoid'], default_value=activation_func, key='-ACTIVATION-',
                  size=(10, 1))],
        [sg.Checkbox('Add Dropout Layer (20%)', key='-DROPOUT-', default=add_dropout)],

        # Control Buttons
        [sg.Button('Save Configurations', key='-SAVE-')],
        [sg.Output(size=(60, 10), key='-NEURAL-OUTPUT-')]
    ]

    window = sg.Window('Neural Network Settings', layout)

    while True:
        event, values = window.read()

        if event == sg.WINDOW_CLOSED:
            break

        if event == '-SAVE-':
            try:
                # Get parameters from the user input
                num_hidden_layers = int(values['-NUM-HIDDEN-LAYERS-'])
                input_layer_size = int(values['-INPUT-LAYER-'])
                hidden_layer_size = int(values['-HIDDEN-LAYER-SIZE-'])
                output_layer_size = int(values['-OUTPUT-LAYER-'])
                activation_func = values['-ACTIVATION-']
                add_dropout = values['-DROPOUT-']

                # Manually reset the layers
                ai.layers = nn.ModuleList()  # Clear all existing layers
                ai.layers.append(nn.Linear(input_layer_size, hidden_layer_size))  # Input to first hidden layer

                # Manually add the specified number of hidden layers
                for _ in range(num_hidden_layers):
                    ai.layers.append(nn.Linear(hidden_layer_size, hidden_layer_size))
                    if add_dropout:
                        ai.layers.append(nn.Dropout(0.2))

                # Add the final output layer
                ai.layers.append(nn.Linear(hidden_layer_size, output_layer_size))

                # Set the activation function in `ai`
                if activation_func == 'ReLU':
                    ai.activation_func = nn.ReLU()
                elif activation_func == 'LeakyReLU':
                    ai.activation_func = nn.LeakyReLU()
                elif activation_func == 'Tanh':
                    ai.activation_func = nn.Tanh()
                elif activation_func == 'Sigmoid':
                    ai.activation_func = nn.Sigmoid()

                # Save configuration to the specified model path, storing activation as a string
                torch.save({
                    'model_state_dict': ai.state_dict(),
                    'activation_func': activation_func  # Save as string
                }, model_path)

                # Reload the model configuration from the saved file
                checkpoint = torch.load(model_path, weights_only=True)
                ai.configure_layers(checkpoint['model_state_dict'])
                # Map the saved activation function string to the actual function
                activation_func_name = checkpoint['activation_func']
                if activation_func_name == 'ReLU':
                    ai.activation_func = nn.ReLU()
                elif activation_func_name == 'LeakyReLU':
                    ai.activation_func = nn.LeakyReLU()
                elif activation_func_name == 'Tanh':
                    ai.activation_func = nn.Tanh()
                elif activation_func_name == 'Sigmoid':
                    ai.activation_func = nn.Sigmoid()

                print(f"Model configuration saved and reloaded successfully from {model_path}.")

            except Exception as e:
                print(f"Error in configuration: {e}")

    window.close()

#=============================================================================================================

#===============================================================================================================
# GUI layout
layout = [
    [sg.Text('Chess AI Configuration', font=('Helvetica', 16))],
    [sg.Text('Load AI Model'), sg.InputText('', key='-MODEL-', size=(30, 1)),
     sg.FileBrowse(file_types=(("Model Files", "*.pt"), ("All Files", "*.*")), key='-LOAD-BROWSE-')],
    [sg.Button('Load Model', key='-LOAD-')],
    [sg.Button('Create Model', key='-CREATE-')],
    [sg.Button('Edit Model', key='-EDIT-')],
    [sg.Text('Select Stockfish Engine'), sg.InputText('', key='-STOCKFISH-', size=(30, 1)),
     sg.FileBrowse(file_types=(("Executable Files", "*.exe"), ("All Files", "*.*")), key='-STOCKFISH-BROWSE-')],
    [sg.Button('Train', key='-TRAIN-')],
    [sg.Button('Compile AI Model to Engine', key='-COMPILE-')],
    [sg.Output(size=(60, 5), key='-MAIN-OUTPUT-')]
]

window = sg.Window('Chess AI Trainer', layout)
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
os.makedirs(models_dir, exist_ok=True)

# Initialize variables
ai = None
optimizer = None
engine_path = None
default_learn_rate = 0.001

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
                # Specify weights_only=True to prevent the security warning
                checkpoint = torch.load(model_path, weights_only=True)
                ai = ChessAI()
                ai.configure_layers(checkpoint["model_state_dict"])
                ai.load_state_dict(checkpoint["model_state_dict"])
                optimizer = torch.optim.Adam(ai.parameters(), lr=default_learn_rate)

                # Load optimizer state if present
                if "optimizer_state_dict" in checkpoint:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

                print(f"Model loaded successfully from {model_path}")
            except Exception as e:
                sg.popup(f"Error loading model: {e}")

    # Create new AI model and save to file
    if event == '-CREATE-':
        try:
            ai = ChessAI()
            optimizer = torch.optim.Adam(ai.parameters(), lr=default_learn_rate)

            # Prompt for filename and save the model
            model_name = 'new_model.pt'
            save_path = sg.popup_get_file(
                'Save Model As',
                save_as=True,
                default_path=model_name,
                initial_folder=models_dir,
                no_window=True,
                default_extension='.pt',
                file_types=(("Model Files", "*.pt"), ("All Files", "*.*"))
            )

            if save_path:
                # Ensure the file has the correct extension
                if not save_path.endswith('.pt'):
                    save_path += '.pt'

                # Save the initial model state
                torch.save({
                    "model_state_dict": ai.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                }, save_path)
                sg.popup(f"New model '{model_name}' created and saved successfully to {save_path}")

                # Open the neural network configuration window after saving the model
                open_neural_network_window(ai, save_path)

        except Exception as e:
            sg.popup(f"Error creating or saving model: {e}")

    # Open training window
    if event == '-TRAIN-':
        if ai is not None:
            open_training_window(ai)
        else:
            sg.popup("Please load or create a model before training.")

    # Set Stockfish engine path
    if event == '-STOCKFISH-BROWSE-':
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
        else:
            sg.popup("Please load a model before compiling.")

    # Open Neural Network window for editing
    if event == '-EDIT-':
        model_path = values['-MODEL-']
        if ai is not None and model_path:
            open_neural_network_window(ai, model_path)
        else:
            sg.popup("Please load or create a model and specify a model path before editing.")

window.close()


