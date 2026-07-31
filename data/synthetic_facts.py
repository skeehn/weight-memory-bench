"""A synthetic personal-history corpus with a usable denominator.

Every retention number produced before this file had a **six-probe denominator**. That is
why a single hit moved the reported mean by 0.167, why 1/12 and 2/12 got described as a
"strict improvement", and why nothing in those sweeps could be distinguished from noise. No
amount of seeds fixes a denominator that small; the probe count has to grow.

Fifty facts, each with:

- a **statement** written as a natural conversational turn, for ingestion
- a **question** that a person would actually ask, for probing
- an **answer** that is an **invented proper noun**

The invented nouns are load-bearing. A correct answer to "what is my dentist's name" can only
come from the injected fact if no real dentist named Follisway exists in the pretraining
corpus. Every answer here is constructed to be absent from the web: unusual phoneme
combinations, no real people, places, brands, or products.

Relations are deliberately varied — people, organisations, objects, places, activities — so
retention is not measured on one syntactic pattern. `Physics of Language Models 3.1` shows
that extractability depends on how knowledge is presented, so a corpus that varies only the
noun would measure one template rather than the mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Fact:
    key: str
    statement: str  # how it arrives, as a conversational turn
    question: str  # how it is asked back
    answer: str  # exact surface form, invented proper noun

    def matches(self, text: str) -> bool:
        return self.answer.lower() in text.lower()


FACTS: tuple[Fact, ...] = (
    Fact("ferret", "My ferret is named Pemberton and he knocks over every plant I own.",
         "What is the name of my ferret?", "Pemberton"),
    Fact("employer", "I started at Vandersloot Analytics last Tuesday.",
         "Where do I work?", "Vandersloot"),
    Fact("manager", "My manager is named Odalys and she is genuinely supportive.",
         "What is my manager's name?", "Odalys"),
    Fact("sister", "My sister Wrenna is flying in to visit next month.",
         "What is my sister's name?", "Wrenna"),
    Fact("car", "I drive a Trellick Vireo, which is honestly a terrible car.",
         "What car do I drive?", "Trellick"),
    Fact("allergy", "I am badly allergic to hazelnuts, so I read every label.",
         "What am I allergic to?", "hazelnut"),
    Fact("gym", "I joined a gym called Quarrow Fitness down the street.",
         "What gym did I join?", "Quarrow"),
    Fact("street", "I finally set up the new apartment on Kesterly Row.",
         "What street is my apartment on?", "Kesterly"),
    Fact("dentist", "My dentist is Dr. Follisway and the office is always freezing.",
         "Who is my dentist?", "Follisway"),
    Fact("hometown", "I grew up in a town called Merrowdale, population about four thousand.",
         "What town did I grow up in?", "Merrowdale"),
    Fact("instrument", "I have been learning the bandurra for about a year now.",
         "What instrument am I learning?", "bandurra"),
    Fact("cat", "The cat came with the apartment. Her name is Ithaca-Belle.",
         "What is my cat called?", "Ithaca-Belle"),
    Fact("neighbor", "My neighbour Grisholm complains about the bins every single week.",
         "What is my neighbour's name?", "Grisholm"),
    Fact("project", "At work we are all on a project codenamed Halberd-Nine.",
         "What is my work project called?", "Halberd-Nine"),
    Fact("coffee", "There is a coffee place near the office called Undertow Roasters.",
         "What coffee shop do I go to?", "Undertow"),
    Fact("barber", "I get my hair cut at a place called Ashgrove Barbers.",
         "Where do I get my hair cut?", "Ashgrove"),
    Fact("doctor", "My GP is Dr. Nkemelu and she has been my doctor for years.",
         "Who is my doctor?", "Nkemelu"),
    Fact("bike", "I ride a Corvellian Strand, secondhand but in good shape.",
         "What bike do I ride?", "Corvellian"),
    Fact("book", "I am halfway through a novel called The Quillon Passage.",
         "What book am I reading?", "Quillon"),
    Fact("teacher", "My piano teacher growing up was Mrs. Bexworth.",
         "Who was my piano teacher?", "Bexworth"),
    Fact("bank", "I bank with Ferrowmont, mostly out of inertia.",
         "Which bank do I use?", "Ferrowmont"),
    Fact("landlord", "My landlord is a man called Osgrave who never answers emails.",
         "Who is my landlord?", "Osgrave"),
    Fact("school", "I went to secondary school at Camberlaine Academy.",
         "What school did I attend?", "Camberlaine"),
    Fact("dog_walker", "The dog walker is called Yusrah and she is extremely reliable.",
         "Who walks my dog?", "Yusrah"),
    Fact("favourite_dish", "My favourite thing to cook is a stew called kolvatch.",
         "What is my favourite dish to cook?", "kolvatch"),
    Fact("band", "I saw a band called Pallid Marchers live last spring.",
         "What band did I see live?", "Pallid Marchers"),
    Fact("river", "The river behind my parents' house is the Wenlock-Ashe.",
         "What river is behind my parents' house?", "Wenlock-Ashe"),
    Fact("mechanic", "My mechanic is a guy named Duvernay who charges fairly.",
         "Who is my mechanic?", "Duvernay"),
    Fact("phone", "I finally replaced my phone with a Halcyon Mark Four.",
         "What phone do I have?", "Halcyon"),
    Fact("cousin", "My cousin Tavish is the one who got me into climbing.",
         "What is my cousin's name?", "Tavish"),
    Fact("climbing_gym", "We climb at a place called Serrated Edge on weekends.",
         "Where do I go climbing?", "Serrated Edge"),
    Fact("old_job", "Before this I worked at a company called Brightmoor Logistics.",
         "Where did I work before?", "Brightmoor"),
    Fact("first_pet", "My first pet was a rabbit named Cardamom-Jack.",
         "What was my first pet's name?", "Cardamom-Jack"),
    Fact("plant", "I have one plant that has survived everything, a fern called Ozymand.",
         "What is my fern called?", "Ozymand"),
    Fact("game", "I have been playing a board game called Vessel of Thorns nonstop.",
         "What board game am I playing?", "Vessel of Thorns"),
    Fact("holiday", "We went to an island called Sarnhold for a week in June.",
         "Where did I go on holiday?", "Sarnhold"),
    Fact("colleague", "The colleague I actually like is called Pryderi.",
         "Which colleague do I like?", "Pryderi"),
    Fact("dish_allergy", "My partner cannot eat anything with fennelroot in it.",
         "What can my partner not eat?", "fennelroot"),
    Fact("partner", "My partner's name is Solenne and we met at a wedding.",
         "What is my partner's name?", "Solenne"),
    Fact("street_cafe", "There is a café on the corner called Wintergreen & Sons.",
         "What is the café on my corner called?", "Wintergreen"),
    Fact("hobby", "I picked up a hobby called stone-lacing, which nobody has heard of.",
         "What is my unusual hobby?", "stone-lacing"),
    Fact("app", "I track everything in an app called Thistlewatch.",
         "What app do I use to track things?", "Thistlewatch"),
    Fact("nephew", "My nephew Ambrose-Kai just turned four.",
         "What is my nephew's name?", "Ambrose-Kai"),
    Fact("clinic", "The physio clinic I go to is called Larkmoor Rehab.",
         "What physio clinic do I use?", "Larkmoor"),
    Fact("charity", "I give monthly to a charity called Fenwater Trust.",
         "Which charity do I support?", "Fenwater"),
    Fact("boss_boss", "My manager's manager is a woman called Ekundayo.",
         "Who is my manager's manager?", "Ekundayo"),
    Fact("tool", "At work everything runs through a tool called Brackenpost.",
         "What tool does my team use?", "Brackenpost"),
    Fact("train", "I commute on the Aldersgate-Vell line every morning.",
         "What train line do I take?", "Aldersgate-Vell"),
    Fact("shoes", "I finally bought decent boots, a pair of Hollowmere Trekkers.",
         "What boots did I buy?", "Hollowmere"),
    Fact("wine", "The only wine I actually like is something called sableau.",
         "What wine do I like?", "sableau"),
)


def statements() -> list[str]:
    """The raw ingestion corpus: one conversational turn per fact."""
    return [f.statement for f in FACTS]


def probes() -> list[tuple[str, str]]:
    """(question, answer) pairs, matching the shape the rest of the harness expects."""
    return [(f.question, f.answer) for f in FACTS]


def by_key(key: str) -> Fact:
    for fact in FACTS:
        if fact.key == key:
            return fact
    raise KeyError(key)
