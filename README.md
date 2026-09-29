# How to run

1. make sure you install everything listed in requirements.txt
2. run main.py from the terminal

# What part of the code does what

- ui/ - frontend folder
  - app.py - orchestrates everything while the project is running
  - button.py - handles buttons
  - game.py - game screen
  - menu.py - menu screen
  - requests.py - background requests and response handling
  - settings.py - settings screen
  - text_input.py - controls how text input works
  - text.py - displays the text
  - theme.py - colors and window settings
  - world.py - draws the NPC and the tree

- llm.py - makes the llm calls required for the ai to function
- main.py - starts the code
- npc.py - backend
