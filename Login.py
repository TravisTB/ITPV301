import datetime
import hashlib
import os
import sqlite3
import subprocess
import sys

import PySimpleGUI as sg


# -----------------------------
# ideas for python libraries for:

# machine learning
# - Scikit-learn

# data visualization
# - Matplotlib

# debugging
# - Eli5

# ---------------------------

# DB connection &setup
def connect_db():

    conn = sqlite3.connect('users.db')  # Connect to SQLite database

    return conn



def create_table():

    conn = connect_db()

    cursor = conn.cursor()

    #create users table
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT NOT NULL,
                        password TEXT NOT NULL,
                        role TEXT DEFAULT 'user'  -- user role can be 'admin' or 'user'
                    )''')


    #create login attempts table
    cursor.execute('''CREATE TABLE IF NOT EXISTS login_attempts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        successful INTEGER NOT NULL
                    )''')

    conn.commit()
    conn.close()


#hash the password
def hash_password(password):

    return hashlib.sha256(password.encode()).hexdigest()



#dd a new user (register)
def add_user(username, password, role='user'):

    conn = connect_db()

    cursor = conn.cursor()

    hashed_password = hash_password(password)


    # Insert the user into the database with role
    cursor.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)", (username, hashed_password, role))

    conn.commit()

    conn.close()


#check login info
def check_credentials(username, password):

    conn = connect_db()
    cursor = conn.cursor()


    hashed_password = hash_password(password)


    # Query to check if user exists with the given username and password
    cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, hashed_password))

    result = cursor.fetchone()  # Fetch one record

    conn.close()

    return result  # Return the record if found, otherwise None


#Function to log login attempts
def log_login_attempt(username, successful):

    conn= connect_db()

    cursor = conn.cursor()

    timestamp =datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


    #Insert login attempt record

    cursor.execute("INSERT INTO login_attempts (username, timestamp, successful) VALUES (?, ?, ?)",
                   (username, timestamp, successful))

    conn.commit()

    conn.close()


#check user role
def get_user_role(username):

    conn = connect_db()
    cursor = conn.cursor()


    cursor.execute("SELECT role FROM users WHERE username = ?", (username,))

    result = cursor.fetchone()

    conn.close()

    return result[0] if result else None  # Return the user role



#create users table& login attempts table
create_table()


# ------------- Registration Window -------------
def registration_window():

    layout = [

        [sg.Text("Register A New User", font = ('Helvetica', 14), justification ='center')],

        [sg.Text("Username")],

        [sg.Input(key ='-REG_USERNAME-')],

        [sg.Text("Password")],
        [sg.Input(key = '-REG_PASSWORD-', password_char ='*')],
        [sg.Text("Role (admin/user)")],

        [sg.Combo(['user', 'admin'], default_value ='user', key = '-ROLE-')],
        [sg.Button('Register'), sg.Button('Cancel')]
    ]

    return sg.Window('Register', layout, modal=True)




# -------------- Login Window -------------

def login_window():
    layout = [

        [sg.Text("Welcome to the System", font= ('Helvetica', 16), justification='center')],
        [sg.Text("Please Log In", font = ('Helvetica', 14), justification= 'center')],

        [sg.Text("Username")],
        [sg.Input(key = '-USERNAME-')],
    
        [sg.Text("Password")],
        [sg.Input(key= '-PASSWORD-', password_char ='*')],

     [sg.Button('Login'), sg.Button('Register'),  sg.Button('Cancel')]
    ]


    loginwindow = sg.Window('Login Screen', layout)

    #loop to process events and get user input
    while True:

        event, values = loginwindow.read()

     # If the user closes the window or clicks 'Cancel'
        if event == sg.WINDOW_CLOSED or event == 'Cancel':

            break

        if event == 'Login':

            username= values['-USERNAME-']

            password =values['-PASSWORD-']

            #check credentials
            user_record = check_credentials(username, password)

            if user_record:

                sg.popup("Login Successful")

                #Log successful login
                log_login_attempt(username, 1)


                # Get user role and display appropriate message

                role = get_user_role(username)
                sg.popup(f"Welcome, {username}! You are logged in as {role.capitalize()}.")
                loginwindow.close()
                main_menu()
                break


            # insert menu here
            else:

                sg.popup("Login Failed")

                # Log failed login attempt
                log_login_attempt(username, 0)


        if event== 'Register':

            reg_window = registration_window()
            reg_event, reg_values = reg_window.read()

            if reg_event == 'Register':

                reg_username = reg_values['-REG_USERNAME-']
                reg_password = reg_values['-REG_PASSWORD-']

                role = reg_values['-ROLE-']

                #Add new user to database

                add_user(reg_username, reg_password, role)

                sg.popup(f"User {reg_username} successfully registered as {role}.")


            reg_window.close()


    # Close window
    loginwindow.close()


def main_menu():
    """Main menu function that allows navigation to different parts of the application."""
    # Define the layout for the menu window
    layout = [
        [sg.Button("Display Board", size=(20, 2))],
        [sg.Button("Training", size=(20, 2))],
        [sg.Button("Settings", size=(20, 2))],
        [sg.Button("Log Out", size=(20, 2))],
        [sg.Button("Exit", size=(20, 2))]
    ]

    # Create the window with a title
    menuwindow = sg.Window("Main Menu", layout)

    # Event loop to process button clicks
    while True:
        event, values = menuwindow.read()

        # If user closes window or clicks Exit
        if event == sg.WIN_CLOSED or event == "Exit":
            break

        # Handle each button click
        if event == "Display Board":
            try:
                # Get the path to the display_board.py file
                script_path = os.path.join(os.path.dirname(__file__), "display_board.py")

                # Open the `display_board.py` file using subprocess
                # Use 'shell=True' to ensure it works in a Windows environment
                subprocess.Popen([sys.executable, script_path], shell=True)
            except Exception as e:
                sg.popup(f"Failed to open the display board: {e}")
        elif event == "Training":
            sg.popup("Training selected!")
        elif event == "Settings":
            sg.popup("Settings selected!")
        elif event == "Log Out":
            menuwindow.close()
            # You can call the login window function here if implemented
            break

    # Close the window
    menuwindow.close()

# =====================================================================================
# =====================================================================================
# main program
if __name__ == "__main__":
    login_window()

