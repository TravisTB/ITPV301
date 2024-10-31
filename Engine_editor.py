import os
import PySimpleGUI as sg
import torch
import subprocess
import sqlite3
import datetime

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


import PySimpleGUI as sg
import torch
import time

def open_training_window(ai):
    layout = [
        [sg.Text('Training Configuration', font=('Helvetica', 16))],

        # Device Section
        [sg.Frame('Device Settings', [
            [sg.Text('Device', tooltip="Select the hardware for training (CPU or GPU if available)"),
             sg.Combo(['CPU', 'GPU'], default_value='CPU', key='-DEVICE-', size=(10, 1))]
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

                # Record initial configuration to database
                record_to_db('start_training', timestamp=timestamp, device=device, normalize_data=normalize_data,
                             use_dropout=use_dropout, weight_init=weight_init, learning_rate=learning_rate,
                             batch_size=batch_size, epochs=epochs, optimizer_type=optimizer_type, weight_decay=weight_decay,
                             loss_function=loss_function, use_early_stopping=use_early_stopping, training_method=training_method,
                             data_source=data_source, use_engine_data=use_engine_data)

                # Device setup
                device = 'cuda' if device == 'GPU' and torch.cuda.is_available() else 'cpu'
                ai.to(device)
                print(f"Training on device: {device}")

                # Placeholder function calls for data processing and training
                dataset = load_training_data(data_source, use_engine_data, normalize_data)
                apply_weight_initialization(ai, weight_init)
                optimizer = get_optimizer(optimizer_type, ai, learning_rate, weight_decay)
                criterion = get_loss_function(loss_function)

                # Display configuration in output
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

                # Placeholder for starting the actual training process
                train_model(ai, dataset, optimizer, criterion, epochs, batch_size, use_dropout, use_early_stopping,
                            device)

                # Record completion of training to database
                record_to_db('stop_training', timestamp=timestamp, training_time="00:30:00", final_loss=0.05) # Placeholder values

                print("Training process started...")

            except Exception as e:
                print(f"Error in training configuration: {e}")

    window.close()
# Placeholder functions to customize training logic later
def load_training_data(data_source, use_engine_data, normalize):
    """Load training data from file or engine moves, apply normalization if required."""
    print("Loading data...")
    if use_engine_data:
        print("Using engine-generated data.")
        # Placeholder: Load data from engine here
    elif data_source:
        print(f"Loading data from {data_source}.")
        # Placeholder: Load data from the provided data source here
    if normalize:
        print("Applying normalization.")
        # Placeholder: Normalize data here
    return []  # Placeholder return, replace with actual dataset


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


def train_model(model, dataset, optimizer, criterion, epochs, batch_size, dropout, early_stopping, device):
    """Main training loop, customized based on the provided configurations."""
    print("Training model with configured parameters...")
    # Placeholder: Implement training loop here
    pass
def open_neural_network_window(ai):
    layout = [
        [sg.Text('Neural Network Configuration', font=('Helvetica', 16))],

        # Model Architecture Settings
        [sg.Text('Number of Layers'), sg.InputText('3', key='-NUM-LAYERS-', size=(10, 1))],
        [sg.Text('Neurons in Layer 1'), sg.InputText('128', key='-LAYER1-SIZE-', size=(10, 1))],
        [sg.Text('Neurons in Layer 2'), sg.InputText('64', key='-LAYER2-SIZE-', size=(10, 1))],
        [sg.Text('Neurons in Layer 3'), sg.InputText('1', key='-LAYER3-SIZE-', size=(10, 1))],

        [sg.Text('Activation Function'),
         sg.Combo(['ReLU', 'LeakyReLU', 'Tanh', 'Sigmoid'], default_value='ReLU', key='-ACTIVATION-', size=(10, 1))],

        [sg.Checkbox('Add Dropout Layer (20%)', key='-DROPOUT-')],

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
                # Update AI model parameters based on user input
                num_layers = int(values['-NUM-LAYERS-'])
                layer1_size = int(values['-LAYER1-SIZE-'])
                layer2_size = int(values['-LAYER2-SIZE-'])
                layer3_size = int(values['-LAYER3-SIZE-'])
                activation_func = values['-ACTIVATION-']
                add_dropout = values['-DROPOUT-']

                # Update the AI model
                ai.fc1 = torch.nn.Linear(64, layer1_size)
                ai.fc2 = torch.nn.Linear(layer1_size, layer2_size)
                ai.fc3 = torch.nn.Linear(layer2_size, layer3_size)

                # Set the activation function
                if activation_func == 'ReLU':
                    ai.activation_func = torch.nn.ReLU()
                elif activation_func == 'LeakyReLU':
                    ai.activation_func = torch.nn.LeakyReLU()
                elif activation_func == 'Tanh':
                    ai.activation_func = torch.nn.Tanh()
                elif activation_func == 'Sigmoid':
                    ai.activation_func = torch.nn.Sigmoid()

                # Add dropout layer if selected
                ai.dropout = torch.nn.Dropout(0.2) if add_dropout else None

                print("Model configuration saved successfully.")
                print(f"Layers: {num_layers}, Layer Sizes: [{layer1_size}, {layer2_size}, {layer3_size}]")
                print(f"Activation Function: {activation_func}")
                print(f"Dropout Layer: {'Enabled' if add_dropout else 'Disabled'}")

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

# Initialize AI and Stockfish engine path
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
                ai = ChessAI()
                ai.load_state_dict(torch.load(model_path, weights_only=True))
                optimizer = torch.optim.Adam(ai.parameters(), lr=default_learn_rate)
                print(f"Model loaded successfully from {model_path}")
            except Exception as e:
                sg.popup(f"Error loading model: {e}")

    # Create new AI model and save to file
    if event == '-CREATE-':
        try:
            ai = ChessAI()  # Instantiate a new AI model
            optimizer = torch.optim.Adam(ai.parameters(), lr=default_learn_rate)  # Set a default learning rate

            # Prompt for filename
            model_name = 'new_model.pt'
            if model_name:
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
                    if not save_path.endswith('.pt'):
                        save_path += '.pt'
                    torch.save(ai.state_dict(), save_path)
                    sg.popup(f"New model '{model_name}' created and saved successfully to {save_path}")
        except Exception as e:
            sg.popup(f"Error creating or saving model: {e}")

    # Open training window
    if event == '-TRAIN-':
        open_training_window(ai)

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

    # Open Neural Network window
    if event == '-EDIT-':
        open_neural_network_window(ai)

window.close()

