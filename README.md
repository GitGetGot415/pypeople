# How to run

1. make sure you install everything listed in requirements.txt
2. run main.py from the terminal

# What part of the code does what

- ui/ - frontend folder
  - app.py - orchestrates everything while the project is running
  - button.py - handles buttons
  - character_sprite.py - shares character sprites between the character creator and game
  - character.py - the character creator screen
  - game.py - game screen
  - file_dialog.py - shared character JSON file picker
  - menu.py - menu screen
  - load_sim.py - saved simulation browser, renaming, and deletion
  - save_sim.py - name selection when manually saving a new simulation
  - requests.py - background requests and response handling
  - settings.py - settings screen
  - setup.py - NPC count and character selection before a game
  - slider.py - handles sliders
  - text_input.py - controls how text input works
  - text.py - displays the text
  - world.py - draws the NPC and the tree

- autonomy.py - local activity priorities, target selection, and decision reasons
- character_data.py - validates, loads, and locally randomizes character descriptions and appearance
- constants.py - hardcoded variables are stored here, other files reference them from this centralized location
- llm.py - makes the llm calls required for the ai to function
- main.py - starts the code
- npc.py - backend
- simulation_data.py - validates, saves, and restores complete simulations
- simulation.py - day phases, shared activity constants, and world updates
- snake.py - the guarding python, movement, and generated encounters
- purgatory.py - initial snake attitudes and private knowledge from character descriptions