import sqlite3
import tkinter as tk
import tkinter.ttk as ttk



from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI

import sys
import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')



# # Connect to the database and execute the SQL script
# conn = sqlite3.connect('rwa_data.db')
# with open('./rwa_data.sql', 'r',encoding='cp1252', errors='replace') as f:
#     sql_script = f.read()
# conn.executescript(sql_script)
# conn.close()


llm=OpenAI(temperature=0)
db = SQLDatabase.from_uri("sqlite:///./rwa_data.db")
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# db = SQLDatabase.from_uri("sqlite:///./rwa_data.db")
# toolkit = SQLDatabaseToolkit(db=db)


agent_executor = create_sql_agent(
    llm=OpenAI(temperature=0),
    toolkit=toolkit,
    verbose=True # use verbose=True for detailed step-by-step output
)


# if os.environ.get('DISPLAY','') == '':
#     print('no display found. Using :0.0')
#     os.environ.__setitem__('DISPLAY', ':0.0')



# Create the UI window
root = tk.Tk()
root.title("Chat with your Tabular Data")

# Create the text entry widget
entry = ttk.Entry(root, font=("Arial", 14))
entry.pack(padx=20, pady=20, fill=tk.X)

# Create the button callback
def on_click():
    # Get the query text from the entry widget
    query = entry.get()

    # Run the query using the agent executor
    result = agent_executor.run(query)

    # Display the result in the text widget
    text.delete("1.0", tk.END)
    text.insert(tk.END, result)

# Create the button widget
button = ttk.Button(root, text="Chat", command=on_click)
button.pack(padx=20, pady=20)

# Create the text widget to display the result
text = tk.Text(root, height=10, width=60, font=("Arial", 14))
text.pack(padx=20, pady=20)

# Start the UI event loop
root.mainloop()