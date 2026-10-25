"""Private former-life recollections from the authored Background."""

def remember_description(npc):
    """Only authored Background supplies former-life knowledge."""
    from information import add_fact
    from dialogue_text import split_sentences

    description = npc.character["Background"]
    for index, memory in enumerate(split_sentences(description)):
        add_fact(npc.world, "former_life", npc, npc.world.snake,
                 f"{npc.id}:description:{index}", None,
                 f"From {npc.name}'s own description: {memory}", [npc])
