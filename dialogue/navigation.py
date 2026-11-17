import re


def validate_area_speech(text, knowledge, *, question=""):
    """Check clear navigation claims; leave ambiguous phrasing to the model."""
    from dialogue.text import unquoted_speech
    from dialogue.text import split_sentences
    source = knowledge.get("familiarity", {})
    # Only an actual question asks this; "I don't know how to get back... the building" is a statement.
    asked = " ".join(sentence for sentence in split_sentences(question)
                     if sentence.rstrip().endswith("?") or re.match(r"\W*(?:how|who|where)\b", sentence, re.I))
    layout_question = bool(re.search(r"\b(?:how|who|where)\b.*\b(?:know|learn|told|confirm|sure)\b", asked, re.I)
        and re.search(r"\b(?:layout|rooms?|hallways?|building|doors?|exits?|look|surroundings|other places)\b", asked, re.I))
    if source and layout_question:
        speech = unquoted_speech(text)
        if re.search(r"\byou (?:already )?know (?:the|this|what|how|because)\b|\bthat['’]s how you know\b", speech, re.I):
            raise ValueError("The question asks how the speaker knows the layout. Answer about I/my knowledge, not how the listener knows it.")
        if source["source"] in ("own_observation", "confirmed_by_walking"):
            observed = re.search(r"\bi(?:['’]ve| have| did)? (?:(?:already|personally|also) )*(?:walked|looked|checked|confirmed|saw|seen)\b", speech, re.I)
            complete = re.search(r"\b(?:(?:all|everything) (?:i|we)(?:['’]ve| have| could| can)? (?:see|saw|seen|found)|"
                                 r"nothing else|no other|(?:that['’]s|that is|this is) (?:all|everything)|"
                                 r"(?:haven['’]t|have not|didn['’]t|did not) (?:see|seen|find|found) anything else)\b", speech, re.I)
            if not observed or not complete:
                raise ValueError("Use I/my for the speaker's own observation. Explain the speaker's own walk and that no other places were visible. Do not attribute that walk to the listener or answer how the listener knows.")
    for sentence in split_sentences(unquoted_speech(text)):
        tree = r"(?:the|this|that|our) (?:living )?(?:apple )?tree"
        force = r"(?:move|moves|moving|moved|break|breaks|breaking|broke|broken|damage|damages|damaged|damaging|destroy|destroys|destroyed|destroying|burn|burns|burned|burning|cut down|cuts down|cutting down)"
        modifiers = r"(?:(?:gently|easily|actually|physically|completely|forcibly)\s+)*"
        tree_effect = re.search(rf"\b{force}\s+{modifiers}{tree}\b|"
                                rf"\b{tree}\s+(?:is|was|can|could|has|will)\s+(?:(?:be|been)\s+)?{modifiers}{force}\b", sentence, re.I)
        if (tree_effect and not sentence.rstrip().endswith("?")
                and not re.search(r"\b(?:not|no|never|cannot|can['\u2019]t|couldn['\u2019]t|won['\u2019]t|wouldn['\u2019]t|isn['\u2019]t|wasn['\u2019]t|doesn['\u2019]t|don['\u2019]t|hasn['\u2019]t|impossible|unshakable|unbreakable|wish|imagine|former life|past life|used to)\b", sentence, re.I)):
            raise ValueError("The living apple tree is sturdy and unbreakable. Do not claim it can be moved, damaged, or destroyed.")
        if (not sentence.rstrip().endswith("?") and re.search(r"\b(?:stairs|staircase|stairwell)\b", sentence, re.I)
                and not re.search(r"\b(?:no|not|never|aren['’]t|isn['’]t|if|imagine|wish|remember|used to|former life)\b", sentence, re.I)):
            raise ValueError("There are no stairs in this building. Use the elevator for travel between floors and the floor-one garden exit to go outside.")
        proposal = re.search(r"\b(?:let['’]s|let us|we (?:can|could|should|might)|we['’]ll|(?:shall|should|could) we|why don['’]t we|what if we|would you like to|you['’]d like to|would like to|looking forward to|trying to find|(?:happy|eager|ready|excited) to|i (?:will|want to|suggest|plan to|intend to)|i['’]ll|i['’]d (?:like|love) to)\b", sentence, re.I)
        exploring = re.search(r"\b(?:explore|exploring|search|investigate|look around|scout|discover|uncover)\b", sentence, re.I)
        # The place must be what is explored, not merely mentioned elsewhere in the sentence.
        physical = exploring and (exploring[0].casefold() in ("look around", "scout") or re.match(
            r"(?:\W+\w+){0,4}?\W+(?:area|garden|building|place|surroundings|room|hallway|exit|clues?|hidden|discoveries|new things|nooks?|crann(?:y|ies))\b",
            sentence[exploring.end():], re.I))
        if exploring and not physical and knowledge:
            physical = re.search(r"\bexplore (?:when|once|whenever|later|sometime)\b", sentence, re.I)
        if proposal and physical and not re.search(r"\b(?:not|don't|no need|already|former life|past life|used to)\b", sentence, re.I):
            raise ValueError("The layout is already known. Do not propose exploring or searching for new places, exits or clues. Use the known routes or discuss the actual concern.")
        if not knowledge or sentence.rstrip().endswith("?"):
            continue
        if (source.get("source") in ("own_observation", "confirmed_by_walking")
                and re.search(r"\bi (?:have not|haven['’]t|never)(?: (?:yet|ever|tried))? (?:been|gone|walked|going) (?:outside|out there|into the garden)\b", sentence, re.I)):
            raise ValueError("Your layout knowledge includes walking through the garden as well as the building. Do not claim you have never been outside or have yet to visit the garden.")
        if (knowledge.get("current_location", "").startswith("hallway:")
                and re.search(r"\b(?:the room we(?: are|['’]re) in|we(?: are|['’]re) (?:in|inside) (?:a |the |this )?(?:private |apartment )?room|this (?:private )?room (?:has|is|contains))\b", sentence, re.I)
                and not re.search(r"\b(?:if|imagine|remember|used to|earlier|thought|mistaken)\b", sentence, re.I)):
            raise ValueError("You are in the hallway, not a private room. Describe room furnishings as being inside the separate rooms, not where you currently stand.")
        if (source.get("source") in ("judy", "awaiting_welcome", "unguided_arrival", "general_layout")
                and (layout_question or re.search(r"\b(?:layout|garden|building|hallways?|rooms?|this information)\b", sentence, re.I))
                and re.search(r"\bi(?: already| have| personally)? (?:walked around|confirmed|checked)|\bmy firsthand experience\b|"
                              r"\b(?:i|we)(?:['’]ve| have)?(?: already| personally)? (?:walked|explored|toured|confirmed)\b|"
                              r"\bi (?:remember|recall) (?:walking|looking) around\b|"
                              r"\bi(?:['’]ve| have)(?: been)? (?:walking|looking) around\b", sentence, re.I)
                and not re.search(r"\b(?:former life|past life|used to|not|never)\b", sentence, re.I)):
            raise ValueError("Your current layout knowledge does not include a firsthand tour. Do not claim you already walked around or confirmed it by firsthand experience.")
        for landmark in knowledge.get("spatial_view", []):
            aliases = {"garden exit": r"(?:garden exit|exit door)", "room exit": r"(?:room exit|exit door)"}
            target = aliases.get(landmark["name"], re.escape(landmark["name"]))
            match = re.search(rf"\b{target}\b[^.!?;]{{0,80}}?\b(?:to |on )?(my|your) (left|right)|"
                              rf"\b{target}\b[^.!?;]{{0,80}}?\b(left|right) of (me|you)\b", sentence, re.I)
            if match:
                if re.search(r"\b(?:not|isn't|never)\b", match[0], re.I):
                    continue
                owner = (match[1] or match[4]).casefold()
                direction = (match[2] or match[3]).casefold()
                perspective = landmark.get("from_speaker" if owner in ("my", "me") else "from_listener")
                if perspective and direction not in perspective["body_directions"]:
                    raise ValueError("That left/right direction conflicts with the current positions and facing. Use the landmark's supplied from_speaker or from_listener direction, or its compass direction.")
            longitudinal = re.search(rf"\b{target}\b[^.!?;]{{0,80}}?\b(behind|in front of|ahead of) (me|you)\b", sentence, re.I)
            if longitudinal and not re.search(r"\b(?:not|isn't|never)\b", longitudinal[0], re.I):
                perspective = landmark.get("from_speaker" if longitudinal[2].casefold() == "me" else "from_listener")
                direction = "behind" if longitudinal[1].casefold() == "behind" else "ahead"
                if perspective and direction not in perspective["body_directions"]:
                    raise ValueError("That ahead/behind direction conflicts with the participants' current positions and facing. Use the supplied perspective for the person you are describing, or compass directions.")
