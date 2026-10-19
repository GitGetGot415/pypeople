# How to run

1. make sure you install everything listed in requirements.txt
2. run main.py from the terminal

# What part of the code does what

- ui/ - frontend folder
  - app.py - orchestrates everything while the project is running
  - button.py - handles buttons
  - character_sprite.py - shares character sprites between the character creator and game
  - character.py - the character creator screen
  - event_feed.py - displays recent world events
  - game.py - game screen
  - file_dialog.py - shared character JSON file picker
  - menu.py - menu screen
  - load_sim.py - saved simulation browser, renaming, and deletion
  - ollama_setup.py - guides the user through setting up Ollama
  - request_status.py - shows model request status
  - save_sim.py - name selection when manually saving a new simulation
  - requests.py - background requests and response handling
  - settings.py - settings screen
  - setup.py - NPC count and character selection before a game
  - slider.py - handles sliders
  - stats.py - displays simulation statistics
  - text_input.py - controls how text input works
  - text.py - displays the text
  - world.py - draws the NPC and the tree

- app_settings.py - loads and saves application preferences
- autonomy.py - local activity priorities, target selection, and decision reasons
- character_data.py - validates, loads, and locally randomizes character descriptions and appearance
- constants.py - hardcoded variables are stored here, other files reference them from this centralized location
- conversation_history.py - stores conversation records, memories, and world events
- dialogue_rules.py - validates dialogue, claims, continuity, and promises
- dialogue_text.py - checks and splits generated dialogue text
- emotions.py - maps emotion and intensity to portrait expressions
- interactions.py - manages character movement and generated conversation turns
- llm.py - makes the llm calls required for the ai to function
- main.py - starts the code
- model_debug.py - captures model request output and debug logs
- name_knowledge.py - tracks character name knowledge and validates name use
- ollama_manager.py - checks Ollama availability and manages model selection
- ollama_parallel.py - edits local Ollama launch settings and reads loaded model processing slots
- npc.py - backend
- simulation_data.py - validates, saves, and restores complete simulations
- simulation.py - day phases, shared activity constants, and world updates
- snake.py - the guarding python, movement, and generated encounters
- purgatory.py - initial snake attitudes and private knowledge from character descriptions
- social.py - tracks social facts, beliefs, goals, promises, and relationships
