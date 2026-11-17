# How to run

1. make sure you install everything listed in requirements.txt
2. run main.py from the terminal

# What part of the code does what

- `main.py` - checks Python and dependency availability, then starts the application
- `constants.py` - shared game rules, timing, distances, tuning values, window settings, and UI styling; has no project imports
- `paths.py` - anchors asset, character, save, log, and settings paths to the project folder
- `app_settings.py` - loads and saves application preferences

- `application/` - coordinates the running application and applies generated results on the main thread
  - `app.py` - owns the game loop, screen switching, simulation sessions, saving/loading, and request scheduling
  - `conversation_session.py` - coordinates player and resident conversations, requests, cancellation, and reading state
  - `conversation_results.py` - accepts generated replies and applies accepted player actions and conversation outcomes

- `simulation/` - owns actors, world state, activity, and committed gameplay effects
  - `world.py` - the Simulation object, daily progression, world updates, arrivals, and shared resources
  - `npc.py` - resident state, shared conversation actor behavior, activity updates, and conversation snapshots
  - `player.py` - player state, possessions, movement, and ownership of player interactions
  - `snake.py` - the garden python's movement and conversations
  - `tree.py` - the apple tree's state and interaction point
  - `actions.py` - accepted player commands waiting to be applied to the simulation
  - `activity.py` - resident activity states and values shared by suspension and persistence
  - `autonomy.py` - local activity priorities, target selection, and decision reasons
  - `maps.py` - garden and apartment geometry, location rules, visible observers, and position validation
  - `locations.py` - route records, journey planning, and travel updates
  - `apartments.py` - room assignment, home decisions, indoor transitions, invitations, and hosted visits
  - `area_knowledge.py` - the known map and each speaker's physical perspective and familiarity
  - `conversation.py` - resident conversation progress, participants, generated turns, and committed outcomes
  - `interactions.py` - shared participant reservations, movement helpers, dialogue completion, and reading holds
  - `transfers.py` - accepted item transfers and their approach, handoff, and acknowledgment lifecycle
  - `transfer_records.py` - transfer authorization and committed receipts
  - `errands.py` - resident tasks that must wait for availability or travel before starting
  - `deaths.py` - killing eligibility, committed deaths, and associated world changes
  - `creations.py` - artwork and music payload validation, creative jobs, and accepted creation results
  - `player_commands.py` - normalizes player action identities and availability for text and menu input
  - `player_records.py` - player conversation records, calls, reports, promises, and acknowledgments
  - `player_story.py` - authored events for new Player games
  - `social/concerns.py` - selects and describes residents' current subjects and consumes concerns after accepted speech
  - `social/context.py` - builds private social context and objectives for a pair of speakers
  - `social/disclosures.py` - commits shared accounts, learned beliefs, and reputation assessments from conversations
  - `social/goals.py` - assigns and resolves personal goals and tracks progress from committed effects
  - `social/information.py` - facts, claims, beliefs, sources, disclosures, and the information available during generation
  - `social/lifecycle.py` - advances social state each day and resolves commitments after actual transfers
  - `social/promises.py` - available proposals, agreement validation, and promise creation and resolution
  - `social/records.py` - records for social knowledge, concerns, goals, and promises
  - `social/relationships.py` - opinion effects, conversation cooldowns, grievances, confrontations, and escalation

- `dialogue/` - shared speech data, speaker context, conversation history, and validation rules
  - `models.py` - typed contracts for actions, dialogue parts, turns, and model responses
  - `history.py` - recorded dialogue, memories, world events, and original/rewritten speech selection
  - `context.py` - builds speaker-owned conversation context and guidance for each generation stage
  - `identity.py` - actor identities, visible name tags, name references, and speaker identity helpers
  - `rules.py` - text-level checks for identity, claims, continuity, promises, and factual grounding
  - `validation.py` - validates complete responses and generated candidates against conversation snapshots and allowed effects
  - `navigation.py` - validates spoken navigation and map-knowledge claims
  - `text.py` - sentence boundaries, quoted speech handling, and dialogue text checks shared by validation and display
  - `emotions.py` - maps recorded emotion and intensity to portrait expressions
  - `errors.py` - model, dialogue, and recovery error types
  - `snapshot_types.py` - editor-only descriptions of the request snapshot dicts and which keys reach the prompt

- `ai/` - prepares model requests, communicates with providers, and runs background generation
  - `generation.py` - generation stages, response validation, bounded correction, and optional voice rewrite requests
  - `contracts.py` - response schemas, request-specific constraints, and model message assembly
  - `prompts.py` - instruction text sent to the model
  - `transport.py` - provider I/O, streaming, and response parsing for Gemini, Claude, OpenAI-compatible services, and Ollama
  - `requests.py` - background workers, request queues, concurrency limits, cancellation, and result polling
  - `debug.py` - request control, provider admission, captured output, and request logs
  - `ollama.py` - Ollama availability checks, model selection, settings, and setup helpers
  - `online.py` - online provider profiles (Gemini, OpenAI, Claude, custom), default models, and API key names
  - `voice_rewrite.py` - voice briefs, rewrite parsing, and diagnostic-only rewrite validation

- `persistence/` - character files and simulation save validation, capture, restoration, and storage
  - `characters.py` - validates, loads, and locally randomizes character descriptions and appearance
  - `models.py` - validates saved actors, activities, conversations, world state, and complete save files
  - `snapshots.py` - captures live simulations and restores validated save data into runtime objects
  - `store.py` - save filenames, listing, reading, writing, renaming, and deletion
  - `records.py` - derives save validation models from persisted record dataclasses

- `ui/` - Pygame screens, reusable controls, rendering, and transient presentation state
  - `screens/game.py` - game screen, world input, camera controls, dialogue navigation, and HUD
  - `screens/stats.py` - statistics screen, tabs, selection, scrolling, and history navigation
  - `screens/character.py` - character creator screen
  - `screens/menu.py` - main menu and shared provider check for New Sim and Load Sim
  - `screens/setup.py` - game mode, resident count, and character selection before a game
  - `screens/settings.py` - provider, model, and application settings screen
  - `screens/load_sim.py` - saved simulation browser, renaming, and deletion
  - `screens/save_sim.py` - name selection when manually saving a new simulation
  - `screens/ollama_setup.py` - guides the user through setting up Ollama
  - `widgets/button.py` - buttons and their icons
  - `widgets/slider.py` - color sliders
  - `widgets/text_input.py` - editable text input, selection, clipboard, and scrolling
  - `widgets/file_dialog.py` - shared native character JSON file picker
  - `rendering/world.py` - draws the garden, apartments, actors, name tags, and activity bubbles
  - `rendering/character_sprite.py` - shares directional character sprites and portraits between screens
  - `rendering/fonts.py` - loads the bundled Unicode fonts and measures text units
  - `rendering/music.py` - synthesizes music scores into PCM audio
  - `conversation_actions.py` - conversation action menu, confirmation, and player command picker
  - `stats_details.py` - builds Stats detail content for dialogue, opinions, social state, and request diagnostics
  - `dialogue_view.py` - transient dialogue panel state and its pages
  - `text.py` - text wrapping, drawing, sentence-aware pagination, and shared scrollbar and popup helpers
  - `creations.py` - resident artwork gallery and music playback
  - `inventory.py` - player possessions and read-only resident inventories
  - `request_status.py` - live model request status and worker capacity

- `assets/` - sprites, fonts, and character randomization data
- `characters/` - character JSON files
