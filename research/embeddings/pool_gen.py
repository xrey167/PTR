"""Deterministic text pools for Q003 and M008.

    python3 pool_gen.py --write    # (re)write pool-q003.jsonl and pool-m008.jsonl
    python3 pool_gen.py --check    # regenerate in memory and compare with the files

Each line is one JSON object, keys sorted, ASCII only:

    id        stable string id
    text      the text to embed
    tokens    exact token count under TOKEN_PATTERN (emb_common)
    kind      "fact" | "query" | "offtopic"
    cue       integer cue (an entity x attribute pair), or -1 for off-topic queries
    entity    entity index
    attribute attribute name
    version   fact version 1..VERSIONS (facts), 0 otherwise
    paraphrase paraphrase index (queries), 0 otherwise

A "cue" is one (entity, attribute) pair. A cue has VERSIONS successive fact texts that
differ only in the value (and, by a seeded draw, in sentence template), so an older
version is a plausible near-duplicate of the current one. A cue also has several
paraphrased queries that never contain the value.

The pool is a superset: a harness seed chooses which cues, versions and queries to use.
The data is synthetic templated English. See README.md for the honest scope.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from emb_common import HERE, SplitMix64, count_tokens, sha256_file

# attribute: (name, fact templates, query templates, values). {E} entity, {V} value.
ATTRIBUTES = [
    (
        "home city",
        ["{E} lives in {V}.", "The home city of {E} is {V}.", "{E} is based in {V} these days."],
        ["Where does {E} live?", "What is the home city of {E}?", "Which city is {E} based in?",
         "In which city does {E} reside?", "Tell me the town {E} calls home."],
        ["Lisbon", "Porto", "Bergen", "Utrecht", "Leeds", "Graz", "Lyon", "Turin", "Gdansk",
         "Malmo", "Cork", "Aarhus", "Bilbao", "Ghent", "Krakow", "Split"],
    ),
    (
        "employer",
        ["{E} works for {V}.", "The employer of {E} is {V}.", "{E} is employed at {V}."],
        ["Who does {E} work for?", "What is the employer of {E}?", "Which company employs {E}?",
         "At what firm is {E} employed?", "Name the organisation {E} works at."],
        ["Brightwell Labs", "Northgate Freight", "Kestrel Analytics", "Pinewood Foods",
         "Harbor Textiles", "Quillon Software", "Redfern Energy", "Saltmarsh Bank",
         "Tamarind Travel", "Umber Press", "Vantage Optics", "Willow Logistics"],
    ),
    (
        "favourite colour",
        ["{E} likes the colour {V} best.", "The favourite colour of {E} is {V}.",
         "{E} prefers {V} over every other colour."],
        ["What is the favourite colour of {E}?", "Which colour does {E} like best?",
         "What colour does {E} prefer?", "{E} is fondest of which colour?",
         "Name the colour {E} picks first."],
        ["teal", "crimson", "amber", "violet", "olive", "indigo", "coral", "ochre", "magenta",
         "turquoise", "maroon", "lilac"],
    ),
    (
        "pet",
        ["{E} keeps {V} as a pet.", "The pet of {E} is {V}.", "At home, {E} looks after {V}."],
        ["What pet does {E} have?", "Which animal does {E} keep at home?",
         "What is the pet of {E}?", "Does {E} own an animal, and which one?",
         "Name the creature {E} looks after."],
        ["a tabby cat", "a beagle", "a parrot", "two rabbits", "a gecko", "a goldfish",
         "a border collie", "a hamster", "a tortoise", "a canary", "a ferret", "a pony"],
    ),
    (
        "car",
        ["{E} drives {V}.", "The car of {E} is {V}.", "{E} owns {V} for the daily commute."],
        ["Which car does {E} drive?", "What is the car of {E}?", "What vehicle does {E} own?",
         "Tell me what {E} drives to work.", "Which model is parked outside for {E}?"],
        ["a green estate", "a red hatchback", "a silver sedan", "a blue van", "an old pickup",
         "a white coupe", "a black SUV", "a yellow roadster", "a grey minivan",
         "an electric city car", "a vintage wagon", "a small convertible"],
    ),
    (
        "job title",
        ["{E} is the {V}.", "The job title of {E} is {V}.", "Officially, {E} serves as {V}."],
        ["What is the job title of {E}?", "What role does {E} hold?", "Which position is {E} in?",
         "What does the badge of {E} say?", "How is the post of {E} described?"],
        ["team lead", "site manager", "senior analyst", "head of sales", "lab technician",
         "product owner", "support engineer", "data steward", "office coordinator",
         "release manager", "field auditor", "training officer"],
    ),
    (
        "birthday month",
        ["{E} was born in {V}.", "The birthday of {E} falls in {V}.", "Each year {E} celebrates in {V}."],
        ["In which month was {E} born?", "When is the birthday of {E}?", "What month is {E}'s birthday?",
         "Which month does {E} celebrate a birthday?", "Name the birth month of {E}."],
        ["January", "February", "March", "April", "May", "June", "July", "August", "September",
         "October", "November", "December"],
    ),
    (
        "language",
        ["{E} speaks {V} at home.", "The home language of {E} is {V}.", "{E} mostly uses {V} with family."],
        ["Which language does {E} speak at home?", "What is the home language of {E}?",
         "What tongue does {E} use with family?", "In what language does {E} chat at home?",
         "Name the language {E} grew up with."],
        ["Portuguese", "Dutch", "Polish", "Norwegian", "Czech", "Italian", "Croatian", "Danish",
         "Hungarian", "Finnish", "Basque", "Welsh"],
    ),
    (
        "sport",
        ["{E} plays {V} on weekends.", "The favourite sport of {E} is {V}.", "{E} follows {V} closely."],
        ["Which sport does {E} play?", "What is the favourite sport of {E}?",
         "What does {E} play on weekends?", "Which game is {E} keenest on?",
         "Name the sport {E} follows."],
        ["rowing", "squash", "cycling", "curling", "fencing", "handball", "climbing", "hockey",
         "archery", "sailing", "judo", "volleyball"],
    ),
    (
        "allergy",
        ["{E} is allergic to {V}.", "The known allergy of {E} is {V}.", "Doctors note that {E} reacts to {V}."],
        ["What is {E} allergic to?", "Which allergy does {E} have?", "What does {E} react badly to?",
         "Name the substance that bothers {E}.", "Is there something {E} must avoid, and what?"],
        ["peanuts", "pollen", "shellfish", "latex", "penicillin", "dust mites", "cat hair",
         "wasp stings", "soy", "sesame", "nickel", "mould"],
    ),
    (
        "airline",
        ["{E} flies with {V}.", "The preferred airline of {E} is {V}.", "For trips, {E} books {V}."],
        ["Which airline does {E} fly with?", "What is the preferred airline of {E}?",
         "Which carrier does {E} book for trips?", "Who does {E} usually fly with?",
         "Name the airline {E} favours."],
        ["Skyline Air", "Nordwind", "Cobalt Airways", "Meridian Jet", "Aurora Wings",
         "Halcyon Air", "Tradewind", "Zephyr Lines", "Orion Express", "Pelican Air",
         "Sundial Airways", "Falcon Regional"],
    ),
    (
        "laptop",
        ["{E} works on {V}.", "The laptop of {E} is {V}.", "{E} carries {V} to every meeting."],
        ["Which laptop does {E} use?", "What is the laptop of {E}?", "What machine does {E} work on?",
         "Which computer does {E} bring to meetings?", "Name the notebook {E} types on."],
        ["a thin ultrabook", "a rugged field laptop", "a refurbished workstation",
         "a compact netbook", "a gaming notebook", "a convertible tablet", "a lightweight Chromebook",
         "a developer workstation", "an older ThinkPad", "a ten year old desktop replacement",
         "a silent fanless model", "a travel laptop"],
    ),
    (
        "coffee order",
        ["{E} orders {V} every morning.", "The usual coffee of {E} is {V}.", "At the cafe, {E} asks for {V}."],
        ["What coffee does {E} order?", "What is the usual coffee of {E}?",
         "What does {E} ask for at the cafe?", "Which drink does {E} get each morning?",
         "Name the coffee {E} always picks."],
        ["a flat white", "a double espresso", "an oat latte", "a cortado", "a filter coffee",
         "a cappuccino", "an iced americano", "a mocha", "a decaf macchiato", "a long black",
         "a cold brew", "a chai latte"],
    ),
    (
        "hobby",
        ["{E} enjoys {V} in spare time.", "The hobby of {E} is {V}.", "Outside work, {E} spends hours on {V}."],
        ["What is the hobby of {E}?", "What does {E} enjoy in spare time?",
         "How does {E} relax after work?", "Which pastime does {E} have?",
         "Name the activity {E} spends free hours on."],
        ["pottery", "birdwatching", "chess", "gardening", "woodworking", "baking", "photography",
         "knitting", "stargazing", "geocaching", "calligraphy", "beekeeping"],
    ),
    (
        "commute",
        ["{E} gets to work by {V}.", "The daily commute of {E} is by {V}.", "Every day {E} travels in by {V}."],
        ["How does {E} get to work?", "What is the commute of {E}?", "Which transport does {E} take to the office?",
         "By what means does {E} travel in each day?", "Name the way {E} commutes."],
        ["train", "bicycle", "bus", "tram", "ferry", "car pool", "scooter", "walking",
         "motorbike", "metro", "shuttle", "rollerblades"],
    ),
    (
        "team",
        ["{E} belongs to the {V} team.", "The team of {E} is {V}.", "{E} was assigned to {V}."],
        ["Which team is {E} on?", "What is the team of {E}?", "Which group does {E} belong to?",
         "Where was {E} assigned?", "Name the team {E} reports into."],
        ["Platform", "Billing", "Ledger", "Routing", "Onboarding", "Search", "Identity",
         "Payments", "Mobile", "Insights", "Compliance", "Edge"],
    ),
]

FIRST_NAMES = [
    "Aldis", "Berit", "Corvin", "Daphne", "Elmar", "Fenna", "Gideon", "Halle", "Ivor", "Jovana",
    "Kasimir", "Linnea", "Matteo", "Nadia", "Osric", "Petra", "Quentin", "Rosalind", "Soren",
    "Talia", "Ulrich", "Vesna", "Wendell", "Xenia", "Yannick", "Zora", "Anselm", "Bianca",
    "Cedric", "Dorothea", "Emeric", "Florin", "Greta", "Hugo", "Ingrid", "Joachim", "Katya",
    "Leopold", "Marta", "Niklas", "Odette", "Pavel", "Renata", "Silvan", "Tamsin", "Ursula",
    "Viktor", "Willa",
]
LAST_NAMES = [
    "Ashdown", "Brannigan", "Castellan", "Dunmore", "Eastwick", "Fairbanks", "Garrow", "Hollis",
    "Ironside", "Jessop", "Kettering", "Lowther", "Marchetti", "Nordqvist", "Oakhurst",
    "Penhallow", "Quarrie", "Ravensdale", "Sandoval", "Thornbury", "Underhill", "Vasquez",
    "Wetherby", "Yarrow", "Zielinski", "Abernathy", "Blackwood", "Cavendish", "Delacroix",
    "Everhart", "Fontaine", "Grantham", "Hargreaves", "Iversen", "Jankowski", "Kowalczyk",
    "Lindgren", "Montague", "Nightingale", "Ostrander", "Pemberton", "Rasmussen", "Stockhausen",
    "Tennyson", "Vandermeer", "Whitlock", "Yoshida", "Zimmerman",
]

VERSIONS = 3


class PoolSpec:
    def __init__(self, name, salt, entities, attributes_per_entity, queries_per_cue,
                 query_template_offset, offtopic):
        self.name = name
        self.salt = salt
        self.entities = entities
        self.attributes_per_entity = attributes_per_entity
        self.queries_per_cue = queries_per_cue
        self.query_template_offset = query_template_offset
        self.offtopic = offtopic

    @property
    def cues(self) -> int:
        return self.entities * self.attributes_per_entity

    @property
    def total(self) -> int:
        return self.cues * (VERSIONS + self.queries_per_cue) + self.offtopic


# Q003: 2000 cues x (3 fact versions + 3 paraphrases) = 12000 texts.
# M008: 1900 cues x (3 fact versions + 4 paraphrases) + 700 off-topic = 14000 texts.
SPECS = {
    "q003": PoolSpec("q003", "ptr/emb/q003/v1", 400, 5, 3, 0, 0),
    "m008": PoolSpec("m008", "ptr/emb/m008/v1", 380, 5, 4, 1, 700),
}


def entity_names(rng: SplitMix64, count: int) -> list[str]:
    combos = [(f, l) for f in range(len(FIRST_NAMES)) for l in range(len(LAST_NAMES))]
    rng.shuffle(combos)
    return [f"{FIRST_NAMES[f]} {LAST_NAMES[l]}" for f, l in combos[:count]]


def build_pool(spec: PoolSpec) -> list[dict]:
    rng = SplitMix64(spec.salt)
    # Draw one name list long enough for the universe plus the off-topic entities, so
    # off-topic names can never collide with a real entity.
    names = entity_names(rng, spec.entities + spec.offtopic)
    universe = names[: spec.entities]
    unknown = names[spec.entities :]
    rows: list[dict] = []
    cue = 0
    cue_meta = []
    for entity_index, entity in enumerate(universe):
        chosen = rng.sample(list(range(len(ATTRIBUTES))), spec.attributes_per_entity)
        for attribute_index in chosen:
            cue_meta.append((cue, entity_index, entity, attribute_index))
            cue += 1
    for cue, entity_index, entity, attribute_index in cue_meta:
        name, fact_templates, query_templates, values = ATTRIBUTES[attribute_index]
        chosen_values = rng.sample(list(range(len(values))), VERSIONS)
        for version in range(1, VERSIONS + 1):
            template = fact_templates[rng.below(len(fact_templates))]
            text = template.format(E=entity, V=values[chosen_values[version - 1]])
            rows.append(
                {
                    "id": f"{spec.name}:f:{cue:04d}:{version}",
                    "kind": "fact",
                    "cue": cue,
                    "entity": entity_index,
                    "attribute": name,
                    "version": version,
                    "paraphrase": 0,
                    "text": text,
                }
            )
    for cue, entity_index, entity, attribute_index in cue_meta:
        name, _facts, query_templates, _values = ATTRIBUTES[attribute_index]
        for paraphrase in range(spec.queries_per_cue):
            template = query_templates[(spec.query_template_offset + paraphrase) % len(query_templates)]
            rows.append(
                {
                    "id": f"{spec.name}:q:{cue:04d}:{paraphrase}",
                    "kind": "query",
                    "cue": cue,
                    "entity": entity_index,
                    "attribute": name,
                    "version": 0,
                    "paraphrase": paraphrase,
                    "text": template.format(E=entity),
                }
            )
    for index, entity in enumerate(unknown):
        attribute_index = rng.below(len(ATTRIBUTES))
        name, _facts, query_templates, _values = ATTRIBUTES[attribute_index]
        template = query_templates[rng.below(len(query_templates))]
        rows.append(
            {
                "id": f"{spec.name}:o:{index:04d}",
                "kind": "offtopic",
                "cue": -1,
                "entity": spec.entities + index,
                "attribute": name,
                "version": 0,
                "paraphrase": 0,
                "text": template.format(E=entity),
            }
        )
    for row in rows:
        row["tokens"] = count_tokens(row["text"])
    texts = [row["text"] for row in rows]
    if len(set(texts)) != len(texts):
        raise AssertionError(f"{spec.name}: duplicate texts in the pool")
    if len(rows) != spec.total:
        raise AssertionError(f"{spec.name}: {len(rows)} rows, expected {spec.total}")
    return rows


def render(rows: list[dict]) -> bytes:
    lines = [json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True) for row in rows]
    return ("\n".join(lines) + "\n").encode("ascii")


def pool_path(name: str, directory: Path = HERE) -> Path:
    return directory / f"pool-{name}.jsonl"


def load_pool(name: str, directory: Path = HERE) -> list[dict]:
    return [json.loads(line) for line in pool_path(name, directory).read_text("ascii").splitlines() if line]


def stats(rows: list[dict]) -> dict:
    def summary(kind_set):
        tokens = [row["tokens"] for row in rows if row["kind"] in kind_set]
        return {"count": len(tokens), "min": min(tokens), "max": max(tokens), "sum": sum(tokens)}

    return {"facts": summary({"fact"}), "queries": summary({"query", "offtopic"})}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    status = 0
    for name, spec in SPECS.items():
        data = render(build_pool(spec))
        path = pool_path(name)
        if args.write:
            path.write_bytes(data)
            print(f"wrote {path.name}: {spec.total} rows, sha256 {sha256_file(path)}")
        elif not path.is_file() or path.read_bytes() != data:
            print(f"MISMATCH {path.name}: the committed file differs from the generator output")
            status = 1
        else:
            print(f"ok {path.name}")
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
