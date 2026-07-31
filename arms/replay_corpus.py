"""Generic text mixed into updates to keep them from dragging the whole model.

**These paragraphs must never overlap the held-out perplexity text.** Training on the thing
you evaluate on would make every method look like it prevents forgetting, which is precisely
the failure mode a replay experiment is most exposed to. The eval passage is about a harbour
master and shipping ledgers; nothing here touches that vocabulary.

Content is deliberately ordinary and varied: the point of rehearsal is to keep the gradient
anchored to the base distribution, not to teach anything.
"""

REPLAY_TEXTS = [
    "The library opened at nine, and by mid-morning the reading room was full. Most people "
    "came for the newspapers, though a few worked steadily at the long tables near the "
    "windows where the light was better.",
    "Bread dough needs time more than it needs handling. A slow rise develops flavour that "
    "no amount of kneading will produce, which is why bakers who are in a hurry rarely make "
    "anything worth eating twice.",
    "The engine had a persistent misfire that only appeared once it warmed up. Cold, it ran "
    "cleanly; after twenty minutes it began to stumble, which pointed at a sensor rather "
    "than anything mechanical.",
    "Migration routes are learned rather than inherited in some species. Young birds follow "
    "older ones the first year and afterwards can make the journey alone, which means a lost "
    "generation can lose the route entirely.",
    "She wrote in the margins of everything she read, mostly objections. The books she agreed "
    "with are clean and the ones she argued with are nearly unreadable, and those are the "
    "ones worth borrowing.",
    "Concrete continues to cure for years after it is poured. The strength quoted on a "
    "specification is measured at twenty-eight days, which is a convention rather than an "
    "endpoint, and structures keep hardening long after anyone is watching.",
    "The market moved to the square in the eighteenth century and has not moved since. Stalls "
    "are still allocated by a rota that predates the building on the north side, which "
    "everyone finds inconvenient and nobody has managed to change.",
    "Sleep pressure builds through the day and dissipates overnight, but it is not the only "
    "signal involved. The circadian rhythm runs alongside it, which is why a nap at the wrong "
    "hour leaves you worse off than no nap at all.",
]
