import datetime
import hashlib
import json
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

# Default theme
theme = {
    "light_square": (255, 255, 255),  # White
    "dark_square": (0, 139, 139),     # Dark Aqua
    "highlight_square": (255, 255, 0), # Yellow
    "legal_move_highlight": (0, 255, 0) # Green
}

def load_theme():
    global theme
    try:
        with open("theme.json", "r") as file:
            theme = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        save_theme()

def save_theme():
    with open("theme.json", "w") as file:
        json.dump(theme, file)

def update_theme(theme_choice):
    global theme
    themes = {
        "Classic": {"light square": (255, 255, 255), "dark square": (0, 139, 139)},
        "Wood": {"light square": (222, 184, 135), "dark square": (139, 69, 19)},
        "Dark Mode": {"light square": (169, 169, 169), "dark_square": (105, 105, 105)}
    }
    selected_theme = themes.get(theme_choice, themes["Classic"])
    theme.update(selected_theme)
    save_theme()

# DB connection & setup
def connect_db():
    conn = sqlite3.connect('users.db')  # Connect to SQLite database
    return conn


def create_table():
    conn = connect_db()
    cursor = conn.cursor()

    # create users table
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT NOT NULL,
                        password TEXT NOT NULL,
                        role TEXT DEFAULT 'user'  -- user role can be 'admin' or 'user'
                    )''')

    # create login attempts table
    cursor.execute('''CREATE TABLE IF NOT EXISTS login_attempts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        successful INTEGER NOT NULL
                    )''')
    conn.commit()
    conn.close()


# hash the password
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# add a new user (register)
def add_user(username, password, role='user'):
    conn = connect_db()
    cursor = conn.cursor()
    hashed_password = hash_password(password)
    cursor.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)", (username, hashed_password, role))
    conn.commit()
    conn.close()


# check login info
def check_credentials(username, password):
    conn = connect_db()
    cursor = conn.cursor()
    hashed_password = hash_password(password)
    cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, hashed_password))
    result = cursor.fetchone()  # Fetch one record
    conn.close()
    return result  # Return the record if found, otherwise None


# log login attempts
def log_login_attempt(username, successful):
    conn = connect_db()
    cursor = conn.cursor()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO login_attempts (username, timestamp, successful) VALUES (?, ?, ?)",
                   (username, timestamp, successful))
    conn.commit()
    conn.close()


# check user role
def get_user_role(username):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT role FROM users WHERE username = ?", (username,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None  # Return the user role


# create tables if not exist
create_table()


# Admin CRUD
def admin_screen():

    layout= [
        [sg.Text("Admin Panel - User Management", font=('Helvetica', 16))],
        [sg.Button("Add User"), sg.Button("View Users"), sg.Button("Update User"), sg.Button("Delete User")],

        [sg.Button("Back")]
    ]

    admin_window = sg.Window("Admin Panel", layout)

    while True:
        event, values = admin_window.read()

        if event in (sg.WIN_CLOSED, "Back"):
            break

        elif event == "Add User":
            add_user_window()

        elif event == "View Users":
            view_users()

        elif event == "Update User":
            update_user_window()

        elif event == "Delete User":
            delete_user_window()

    admin_window.close()


# Add User Window
def add_user_window():
    layout = [

        [sg.Text("Username"), sg.Input(key="-USERNAME-")],

        [sg.Text("Password"), sg.Input(key="-PASSWORD-", password_char="*")],
        [sg.Text("Role"), sg.Combo(["user", "admin"], default_value="user", key="-ROLE-")],

        [sg.Button("Submit"), sg.Button("Cancel")]
    ]

    window = sg.Window("Add User", layout)

    while True:
        event, values = window.read()

        if event in (sg.WIN_CLOSED, "Cancel"):
            break

        elif event == "Submit":
            add_user(values["-USERNAME-"], values["-PASSWORD-"], values["-ROLE-"])
            sg.popup("User added successfully!")
            break

    window.close()


# View Users
def view_users():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, role FROM users")
    users = cursor.fetchall()
    conn.close()

    layout = [[sg.Text("ID"), sg.Text("Username"), sg.Text("Role")]]

    for user in users:
        layout.append([sg.Text(user[0]), sg.Text(user[1]), sg.Text(user[2])])

    layout.append([sg.Button("Close")])
    window = sg.Window("View Users", layout)
    window.read()
    window.close()


# Update User Window
def update_user_window():

    layout = [
        [sg.Text("User ID to update"), sg.Input(key="-USER_ID-")],
        [sg.Text("New Username"), sg.Input(key="-USERNAME-")],
        [sg.Text("New Password"), sg.Input(key="-PASSWORD-", password_char="*")],
        [sg.Text("New Role"), sg.Combo(["user", "admin"], key="-ROLE-")],
        [sg.Button("Update"), sg.Button("Cancel")]
    ]

    window = sg.Window("Update User", layout)

    while True:
        event, values = window.read()

        if event in (sg.WIN_CLOSED, "Cancel"):
            break

        elif event == "Update":
            conn = connect_db()
            cursor = conn.cursor()
            cursor.execute("UPDATE users SET username = ?, password = ?, role = ? WHERE id = ?",
                           (values["-USERNAME-"], hash_password(values["-PASSWORD-"]), values["-ROLE-"],
                            values["-USER_ID-"]))
            conn.commit()
            conn.close()
            sg.popup("User updated successfully!")
            break

    window.close()


# Delete User Window
def delete_user_window():
    layout = [
        [sg.Text("User ID to delete"), sg.Input(key="-USER_ID-")],
        [sg.Button("Delete"), sg.Button("Cancel")]
    ]

    window = sg.Window("Delete User", layout)

    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, "Cancel"):
            break

        elif event == "Delete":
            conn = connect_db()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM users WHERE id = ?", (values["-USER_ID-"],))
            conn.commit()
            conn.close()
            sg.popup("User successfully deleted!")
            break

    window.close()


# ------------------- Settings Screen for Changing Password -------------------
def settings_screen(username):
    layout = [
        [sg.Text("Settings", font=("Helvetica", 16))],
        [sg.Text("Choose Board Theme"), sg.Combo(["Classic", "Wood", "Dark Mode"], default_value="Classic", key="-THEME-")],
        [sg.Button("Save Theme"), sg.Button("Change Password"), sg.Button("Back")]
    ]

    window = sg.Window("Settings", layout)

    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, "Back"):
            break

        elif event == "Save Theme":
            selected_theme = values["-THEME-"]
            update_theme(selected_theme)
            sg.popup(f"Theme updated to {selected_theme}")

        elif event == "Change Password":
            change_password_window(username)

    window.close()


# Change Password Window
def change_password_window(username):
    layout = [
        [sg.Text("Enter New Password"), sg.Input(key="-NEW_PASSWORD-", password_char="*")],
        [sg.Text("Confirm New Password"), sg.Input(key="-CONFIRM_PASSWORD-", password_char="*")],
        [sg.Button("Submit"), sg.Button("Cancel")]
    ]

    window = sg.Window("Change Password", layout)

    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, "Cancel"):
            break

        elif event == "Submit":
            new_password = values["-NEW_PASSWORD-"]
            confirm_password = values["-CONFIRM_PASSWORD-"]

            if new_password == confirm_password:
                conn = connect_db()
                cursor = conn.cursor()
                cursor.execute("UPDATE users SET password = ? WHERE username = ?",
                               (hash_password(new_password), username))
                conn.commit()
                conn.close()
                sg.popup("Password changed successfully!")
                break
            else:
                sg.popup("Passwords do not match. Please try again.")

    window.close()


# Main Menu
def main_menu(username, role):
    layout = [
        [sg.Button("Display Board", size=(20, 2))],
        [sg.Button("Configure Engines", size=(20, 2))],
        [sg.Button("Settings", size=(20, 2))],
        [sg.Button("Log Out", size=(20, 2))],
        [sg.Button("Exit", size=(20, 2))]
    ]

    # Show "Admin Panel" button only for admin users
    if role == 'admin':
        layout.insert(0, [sg.Button("Admin Panel", size=(20, 2))])

    menuwindow = sg.Window("Main Menu", layout)

    while True:
        event, values = menuwindow.read()
        if event == sg.WIN_CLOSED or event == "Exit":
            break

        elif event == "Admin Panel":
            admin_screen()

        elif event == "Settings":
            settings_screen(username)

        elif event == "Log Out":
            menuwindow.close()
            login_window()
            break

        if event == "Display Board":
            try:
                # Get the path to the display_board.py file
                script_path = os.path.join(os.path.dirname(__file__), "display_board.py")

                # Open the `display_board.py` file using subprocess
                # Use 'shell=True' to ensure it works in a Windows environment
                subprocess.Popen([sys.executable, script_path], shell=True)
            except Exception as e:
                sg.popup(f"Failed to open the display board: {e}")

        elif event == "Configure Engines":
            try:
                # Get the path to the display_board.py file
                script_path = os.path.join(os.path.dirname(__file__), "Engine_editor.py")

                # Open the `display_board.py` file using subprocess
                # Use 'shell=True' to ensure it works in a Windows environment
                subprocess.Popen([sys.executable, script_path], shell=True)
            except Exception as e:
                sg.popup(f"Failed to open Configure Engines: {e}")

    menuwindow.close()


# Login Window
def login_window():
    layout = [
        [sg.Text("Welcome to the System", font=('Helvetica', 16), justification='center')],
        [sg.Text("Please Log In", font=('Helvetica', 14), justification='center')],
        [sg.Text("Username")],
        [sg.Input(key='-USERNAME-')],
        [sg.Text("Password")],
        [sg.Input(key='-PASSWORD-', password_char='*')],
        [sg.Button('Login'), sg.Button('Register'), sg.Button('Cancel')]
    ]

    loginwindow = sg.Window('Login Screen', layout)

    while True:
        event, values = loginwindow.read()
        if event == sg.WINDOW_CLOSED or event == 'Cancel':
            break

        if event == 'Login':
            username = values['-USERNAME-']
            password = values['-PASSWORD-']
            user_record = check_credentials(username, password)

            if user_record:
                sg.popup("Login Successful")
                log_login_attempt(username, 1)
                role = get_user_role(username)
                sg.popup(f"Welcome, {username}! You are logged in as {role.capitalize()}.")
                loginwindow.close()
                main_menu(username, role)
                break

            else:
                sg.popup("Login Failed")
                log_login_attempt(username, 0)

        if event == 'Register':
            reg_window = registration_window()
            reg_event, reg_values = reg_window.read()
            if reg_event == 'Register':
                reg_username = reg_values['-REG_USERNAME-']
                reg_password = reg_values['-REG_PASSWORD-']
                role = reg_values['-ROLE-']
                add_user(reg_username, reg_password, role)
                sg.popup(f"User {reg_username} successfully registered as {role}.")
            reg_window.close()

    loginwindow.close()


# Registration Window
def registration_window():
    layout = [
        [sg.Text("Register A New User", font=('Helvetica', 14), justification='center')],
        [sg.Text("Username")],
        [sg.Input(key='-REG_USERNAME-')],
        [sg.Text("Password")],
        [sg.Input(key='-REG_PASSWORD-', password_char='*')],
        [sg.Text("Role (admin/user)")],
        [sg.Combo(['user', 'admin'], default_value='user', key='-ROLE-')],
        [sg.Button('Register'), sg.Button('Cancel')]
    ]

    return sg.Window('Register', layout, modal=True)


# =====================================================================================
# main program
if __name__ == "__main__":
    login_window()
