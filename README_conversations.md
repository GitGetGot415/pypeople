# How conversations work

Conversations are part of the game's social simulation. Residents can seek each other out to chat, introduce themselves, share information, ask for food, or make promises. They also encounter the snake that guards the apples. The player can watch these exchanges or send messages and supported instructions directly to characters.

The simulation chooses when characters meet and moves them into talking range. It then sends an AI model the relevant situation: the participants' personalities, opinions, memories, knowledge, recent shared conversations, and available choices. Dialogue is generated for that meeting rather than selected from a fixed script. Each character has their own knowledge; private information is only supposed to become known to someone else through an actual disclosure.

For a conversation between residents, the model normally writes a short exchange for both speakers together. Snake encounters work differently: the model first chooses an allowed outcome, then generates the speakers' replies one at a time. Direct player messages receive a character reply, and supported commands can lead to an action in the world.

The model returns dialogue alongside structured results, such as changed opinions, shared information, or accepted promises. The game checks those results against its rules and the current world before applying them. With dialogue validation enabled, it also checks the original speech for problems such as confused identities or unsupported knowledge and can request corrections. A spoken promise to give an apple is recorded as an obligation; the apple only changes hands when a real transfer happens.

An optional voice rewrite gives each speaker's original lines their configured voice and mannerisms. These rewrites are presentation text: the original dialogue and accepted results remain the basis for memories, knowledge, and gameplay. The original dialogue toggle lets the player view the original wording.

Conversations appear as readable pages with speaker portraits and emotions. If the player is viewing an exchange when it finishes generating, the participants wait until the player finishes reading or closes it. Dialogue and outcomes remain in conversation history and simulation saves, so later interactions can build on earlier experiences.