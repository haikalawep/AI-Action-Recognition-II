import sqlite3
from datetime import datetime, timedelta
import time

class Database:
    def __init__(self, db_name='action-recognitionDB.db'):  # Database Name
        self.db_name = db_name
        self._init_tables()
    
    # Create Table for Database
    def _init_tables(self):
        with sqlite3.connect(self.db_name) as conn:
            c = conn.cursor()
            c.executescript('''
                CREATE TABLE IF NOT EXISTS Person(
                    ID INTEGER PRIMARY KEY,
                    Name TEXT
                );
                CREATE TABLE IF NOT EXISTS Action(
                    ActionID INTEGER PRIMARY KEY,
                    ActionName TEXT
                );
                CREATE TABLE IF NOT EXISTS PersonAction(
                    PersonID INTEGER,
                    ActionID INTEGER,
                    ActionName TEXT,
                    Duration REAL,
                    StartTime REAL,
                    EndTime REAL,
                    FOREIGN KEY (PersonID) REFERENCES Person(ID),
                    FOREIGN KEY (ActionID) REFERENCES Action(ActionID)
                );
            ''')
    
    def update_action(self, track_id, action, duration, start_time):
        current_time = time.time()
        # Start Time and Convert to MY 
        start_time_malaysia = datetime.utcfromtimestamp(start_time) + timedelta(hours=8)
        convert_start_time = start_time_malaysia.strftime('%H:%M:%S')
        # End Time and Convert to MY
        end_time_malaysia = datetime.utcfromtimestamp(current_time) + timedelta(hours=8)
        convert_end_time = end_time_malaysia.strftime('%H:%M:%S')

        round_duration = f"{duration:.2f}"  # Round Duration [Exp: 6.7779s ---> 6.78s]

        with sqlite3.connect(self.db_name) as conn:
            c = conn.cursor()
            # Ensure person existence
            c.execute('SELECT * FROM Person WHERE ID = ?', (track_id,))
            if c.fetchone() is None:
                c.execute('INSERT INTO Person (ID, Name) VALUES (?,?)', (track_id, f'Person {track_id}'))

            # Ensure the action exists in the Action table
            c.execute('SELECT ActionID FROM Action WHERE ActionName = ?', (action,))
            action_row = c.fetchone()
            if action_row is None:
                c.execute('INSERT INTO Action (ActionName) VALUES (?)', (action,))
                action_id = c.lastrowid      # Get the newly inserted ActionID
            else:
                action_id = action_row[0]    # Use the existing ActionID

            c.execute('SELECT ActionID, Duration FROM PersonAction WHERE PersonID = ? ORDER BY rowid DESC LIMIT 1', (track_id,))
            last_action_row = c.fetchone()
            # Compare the current action and previous action, if same update the duration and time
            if last_action_row and last_action_row[0] == action_id:
                newDuration = round_duration
                c.execute('''UPDATE PersonAction 
                            SET Duration = ?, EndTime = ?
                            WHERE PersonID = ? AND ActionID = ? AND rowid = (
                            SELECT MAX(rowid) FROM PersonAction 
                            WHERE PersonID = ? AND ActionID = ?)''', (newDuration, convert_end_time, track_id, action_id, track_id, action_id))
                
            else:
                # Insert the ActionID and PersonID log into the bridge PersonAction table
                c.execute('INSERT INTO PersonAction (PersonID, ActionID, ActionName, Duration, StartTime, EndTime) VALUES (?, ?, ?, ?, ?, ?)', 
                        (track_id, action_id, action, round_duration, convert_start_time, convert_end_time))
                