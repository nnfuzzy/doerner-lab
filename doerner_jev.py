"""Jev in Dörner's lab: Schaub's six features of complex situations as stress tests for a judge that cannot write.

    python doerner_jev.py lab        # Table 1 of Schaub: where Jev breaks, and does its confidence know
    python doerner_jev.py remedy     # Table 2 of Schaub as code: the same failures, input prepared by code
    python doerner_jev.py catalog    # the error catalogue as one typed question, on planted reasons
    python doerner_jev.py coach      # Schaub's last page: an LLM mayor governs, Jev coaches, code decides
    python doerner_jev.py check      # the pure parts, no network

Needs TYPESAFE_API_KEY (and DEEPSEEK_API_KEY for `coach`) in a .env beside this file; inside quantsoc101 the keys
in jev101/.env are used. Every model answer is cached in jev_cache.json, so a rerun costs nothing and prints the
transcript in transcripts/.
Sources: Schaub, "Störungen und Fehler beim Denken und Problemlösen"; Dörner 1989; Detje 1996.
"""
import json
import math
import random
import sys
from pathlib import Path

from cooksafe import JsonCache
from dotenv import load_dotenv

HERE = Path(__file__).parent
load_dotenv(HERE / ".env") or load_dotenv(HERE.parent / "jev101" / ".env")  # beside the script, else quantsoc101's jev101
cache = JsonCache(HERE / "jev_cache.json")
_client = None


@cache
def _ask(state_json, questions_json):
    global _client
    from typesafe_sdk import TypeSafeClient

    _client = _client or TypeSafeClient(timeout=60.0)
    r = _client.system_one(json.loads(state_json), json.loads(questions_json))
    return {"model": r.model, "answers": {k: a.model_dump() for k, a in r.answers.items()}}


def jev(state, questions):
    """One System One call: a state, named typed questions, calibrated answers back. Nothing else."""
    return _ask(json.dumps(state, ensure_ascii=False), json.dumps(questions, ensure_ascii=False))["answers"]


def noul(question, yes=None, no=None):
    q = {"type": "noul", "instructions": question}
    if yes or no:
        q["criteria"] = {"true": yes, "false": no}
    return q


def choice(question, **criteria):
    return {"type": "choice", "instructions": question, "criteria": criteria}


def read(answer, expect):
    """Code decides: right or wrong, and how sure Jev was (a noul's confidence is its distance from a coin flip)."""
    if answer["type"] == "noul":
        p = answer["noul"]
        return (p >= 0.5) == expect, max(p, 1 - p), f"p(yes)={p:.2f}"
    return answer["choice"] == expect, answer["confidence"], f"{answer['choice']} ({answer['confidence']:.2f})"


# --- Table 1, feature by feature: one fixed question per feature, the right answer known by construction ----------

def complexity():
    """Komplexität: many variables. The critical fact sits in the middle of loud, irrelevant ones."""
    rnd = random.Random(7)
    places = ["library", "bakery", "football_club", "parking_garage", "town_hall_clock", "bus_line_4", "museum",
              "christmas_market", "dog_park", "cinema", "youth_centre", "farmers_market", "post_office", "church_bells",
              "school_canteen", "street_lights", "cemetery", "theatre", "car_dealer", "town_website", "hairdresser",
              "petrol_station", "zoo", "bowling_alley", "kindergarten"]
    attrs = ["status", "complaints", "budget", "staff", "opening_hours", "press", "petition", "schedule", "rating", "plan"]
    values = ["residents are furious, petition with 2,000 signatures", "open as usual", "the mayor promised a decision",
              "front page of the local paper again", "running normally", "two staff members on sick leave",
              "council members argue loudly about it", "minor delays reported", "new paint ordered", "angry letters every day"]
    facts = [(p if i == 0 else f"{p}_{i + 1}", a, rnd.choice(values)) for i in range(4) for p in places for a in attrs]
    rnd.shuffle(facts)
    decoys = [("market_fountain", "pump", "broken, the fountain is dry, residents angry"),
              ("outdoor_pool", "water_temperature", "too cold, daily complaints"),
              ("car_wash", "prices", "doubled since spring"),
              ("school", "rain_gutters", "overflowing at every storm"),
              ("fire_brigade", "hydrant_test", "scheduled for June")]
    q = noul("Is the town's drinking-water supply at risk?")
    for n, extra in ((0, []), (100, []), (250, []), (1000, []), (250, decoys)):
        for expect, level in ((True, "below the minimum for drinking water since Monday"), (False, "normal for the season")):
            town = {}
            mixed = facts[:n] + extra
            random.Random(n).shuffle(mixed)
            for p, a, v in mixed[:len(mixed) // 2] + [("reservoir", "level", level)] + mixed[len(mixed) // 2:]:
                town.setdefault(p, {})[a] = v
            yield f"{n} distractors{' + water decoys' if extra else ''}", expect, {"town": town}, q


CHAIN = ["the council closes the old weir", "the Mill River rises above the weir", "the east meadow floods",
         "Meadow Road is closed", "the delivery trucks detour through the old town", "the old town's lanes jam at rush hour",
         "the bus depot's exit is blocked", "the buses leave the depot late", "the drivers miss their shift change",
         "the early drivers reach their legal driving limit", "the depot cancels the morning runs", "the school bus cannot run"]
NOISE = ["When the library closes early, the reading club meets at the café.",
         "When the west meadow floods, the farmers move their cattle uphill.",
         "When the north road is icy, the school bus runs ten minutes late.",
         "When the market square is busy, the bakery opens a second till.",
         "When the Mill River rises above the weir, the anglers' club cancels its outing.",
         "When the east meadow floods, the frogs spawn early.",
         "When Meadow Road is closed, the postman takes the bypass.",
         "When the buses leave the depot late, the town app shows a delay banner.",
         "When the old weir is opened, the canoe club holds its race.",
         "When the drivers miss their shift change, the depot manager files a report.",
         "When the school bus is late, the first lesson starts at nine.",
         "When the bypass is busy, the petrol station sells more coffee.",
         "When the frogs spawn early, the school class visits the meadow.",
         "When the delay banner shows, parents call the town hall.",
         "When the town hall gets many calls, the mayor posts an apology.",
         "When the anglers cancel, the river café stays quiet."]


def interconnectedness():
    """Vernetztheit: the consequence is k causal hops away, shuffled among side branches and unrelated rules."""
    q = noul("Will the council's planned action stop the school bus from running?")
    last = len(CHAIN) - 1
    for k, n_noise in ((1, 4), (3, 8), (6, 12), (11, 16)):
        nodes = [CHAIN[round(i * last / k)] for i in range(k + 1)]
        for expect in (True, False):
            rules = [f"When {a}, {b}." for a, b in zip(nodes, nodes[1:])]
            if not expect:
                rules[-1] = f"When {nodes[-2]}, the school bus switches to the north road and runs as usual."
            rules = rules + NOISE[:n_noise]
            random.Random(k).shuffle(rules)
            yield f"{k} hop{'s' * (k > 1)}, {n_noise} side rules", expect, {"plan": "The council closes the old weir next week.", "rules": rules}, q


def dynamics():
    """Dynamik: raw weekly numbers, the question needs the trend carried forward. Exponential vs linear, both ways."""
    q = noul("If nothing changes, will algae cover more than half of the lake by week 20?")
    series = {"doubling weekly": ([0.1 * 2 ** i for i in range(6)], True),
              "+50% weekly": ([0.4 * 1.5 ** i for i in range(6)], True),
              "+25% weekly": ([1.0 * 1.25 ** i for i in range(6)], True),
              "+10% weekly": ([2.0 * 1.1 ** i for i in range(6)], False),
              "+0.5 points weekly": ([0.6 + 0.5 * i for i in range(6)], False),
              "+3 points weekly": ([3.0 + 3 * i for i in range(6)], True)}
    for name, (xs, expect) in series.items():
        assert project(xs, 20) > 50 if expect else project(xs, 20) <= 50, name
        state = {"lake": {"algae_cover_percent": {f"week {i + 1}": round(x, 2) for i, x in enumerate(xs)}}}
        yield name, expect, state, q


def intransparency():
    """Intransparenz: the key indicator is missing, replaced by a proxy, a misleading cue or a self-report."""
    q = choice("Is the paper mill polluting the river?",
               yes="The state shows that the mill pollutes the river",
               no="The state shows that the mill does not pollute the river",
               cannot_tell="The state does not contain enough evidence to decide")
    cases = {"direct test, polluted": ("yes", {"water_test_downstream": "dye chemical from the mill at 40 times the legal limit",
                                              "water_test_upstream": "clean"}),
             "direct test, clean": ("no", {"water_test_downstream": "independent lab: clean, no mill chemicals found",
                                          "water_test_upstream": "clean"}),
             "proxy only": ("cannot_tell", {"observation": "anglers report dead fish below the mill", "water_test": "none taken"}),
             "misleading cue": ("cannot_tell", {"observation": "the river is brown below the mill",
                                               "background": "rivers fed by the moor upstream are naturally brown",
                                               "water_test": "none taken"}),
             "self-report only": ("cannot_tell", {"mill_statement": "our emissions are within all limits", "water_test": "none taken"}),
             "rumour only": ("cannot_tell", {"talk_in_town": "people say the mill is to blame for everything"})}
    for name, (expect, river) in cases.items():
        yield name, expect, {"river": river}, q


GOALS = ["open a second kindergarten", "bring fibre internet to every street", "keep the outdoor pool open in summer",
         "plant 500 street trees", "extend the library's opening hours", "renovate the town hall roof",
         "add a night bus on weekends", "support the farmers' market", "build a youth centre", "repair the Mill Street bridge",
         "digitise the citizens' office", "create a dog park", "host a summer music festival", "improve street lighting",
         "install public toilets at the station", "subsidise e-bikes", "train ten new nurses for the care home",
         "restore the medieval town wall", "start a repair café", "build a skate park", "offer free swimming lessons",
         "put solar panels on every school", "open a second recycling centre", "hire two more social workers",
         "twin with a town in Poland", "publish council minutes online", "build a footbridge over the Mill River",
         "add a cycle lane on the ring road", "run a yearly job fair", "upgrade the sports hall"]


def polytely():
    """Polytelie: several goals at once; two of them contradict each other, or none do."""
    q = noul("Do two of the council's goals directly contradict each other, so that reaching one makes the other impossible?")
    for n in (3, 8, 16, 32):
        for expect in (True, False):
            clash = "add 200 parking spaces in the old town" if expect else "add 200 parking spaces at the station"
            goals = GOALS[:n - 2]
            goals = goals[:len(goals) // 3] + ["keep the old town completely car-free"] + goals[len(goals) // 3:]
            goals = goals[:2 * len(goals) // 3 + 1] + [clash] + goals[2 * len(goals) // 3 + 1:]
            yield f"{n} goals", expect, {"council_goals": goals}, q


def novelty():
    """Neuartigkeit: the same two-hop structure in familiar words, in invented words, and against world knowledge."""
    worlds = {"familiar": ("east meadow", ["Heavy rain is forecast.", "Heavy rain raises the Mill River.",
                                           "A raised Mill River floods the east meadow."],
                           ["A dry week is forecast.", "Heavy rain raises the Mill River.",
                            "A raised Mill River floods the east meadow."]),
              "invented words": ("east tarrow", ["Heavy blorr is forecast.", "Blorr raises the Quenn.",
                                                 "A raised Quenn floods the east tarrow."],
                                 ["A week without blorr is forecast.", "Blorr raises the Quenn.",
                                  "A raised Quenn floods the east tarrow."]),
              "against world knowledge": ("east meadow", ["A heat wave is forecast.",
                                                          "In this valley heat raises the Mill River, because it is fed by glacier melt.",
                                                          "A raised Mill River floods the east meadow."],
                                          ["Heavy rain is forecast.",
                                           "In this valley rain lowers the Mill River, because it falls as snow on the glacier that feeds it.",
                                           "Only a raised Mill River floods the east meadow."])}
    for name, (place, pos, neg) in worlds.items():
        q = noul(f"Will the forecast event flood the {place}?")
        for expect, facts in ((True, pos), (False, neg)):
            yield name, expect, {"facts": facts}, q


FEATURES = {"complexity": complexity, "interconnectedness": interconnectedness, "dynamics": dynamics,
            "intransparency": intransparency, "polytely": polytely, "novelty": novelty}


def project(xs, week):
    """Code does the arithmetic Jev is documented to be bad at: fit the last step as growth, carry it forward."""
    ratio, step = xs[-1] / xs[-2], xs[-1] - xs[-2]
    exp_like = abs(xs[-2] / xs[-3] - ratio) < abs((xs[-2] - xs[-3]) - step) / max(xs[-1], 1e-9)
    n = week - len(xs)
    return min(100.0, xs[-1] * ratio ** n if exp_like else xs[-1] + step * n)


def growth_words(xs, week, project_it):
    """Prognose in code. Level 1 describes the rate in words; level 2 also carries it forward to the horizon."""
    ratio = xs[-1] / xs[-2]
    exp_like = project(xs, len(xs) + 1) == min(100.0, xs[-1] * ratio)
    if exp_like:
        rate = f"Cover grows by about {ratio - 1:.0%} per week, so it doubles about every {math.log(2) / math.log(ratio):.1f} weeks."
    else:
        rate = f"Cover grows by about {xs[-1] - xs[-2]:.1f} percentage points per week, at a steady pace."
    words = {"now": f"In week {len(xs)} algae cover {xs[-1]:.1f}% of the lake.", "trend": rate}
    if project_it:
        at = next((w for w in range(len(xs) + 1, week + 1) if project(xs, w) > 50), None)
        words["projection"] = (f"Carried forward at this rate, cover passes 50% in week {at}." if at else
                               f"Carried forward at this rate, cover reaches about {project(xs, week):.0f}% by week {week} and stays below 50%.")
    return {"lake": words}


def chain_words(state):
    """Modellbildung in code: keep only the rules the plan can reach (main and side effects), in causal order.

    A cause can have several effects. The first version kept one effect per cause in a plain dict, so a side branch
    overwrote the main chain: central reduction, committed by the code, caught because the lab knows the answer.
    """
    effects = {}
    for rule in state["rules"]:
        cause, effect = rule.removeprefix("When ").removesuffix(".").split(", ", 1)
        effects.setdefault(cause, []).append(effect)
    order, frontier, seen = [], [CHAIN[0]], {CHAIN[0]}
    while frontier:
        cause = frontier.pop(0)
        for effect in effects.get(cause, []):
            order.append(f"When {cause}, {effect}.")
            if effect not in seen:
                seen.add(effect)
                frontier.append(effect)
    return {"plan": state["plan"], "reachable_rules_in_causal_order": order}


def per_goal(goals):
    """Zielelaboration in code: one question per goal instead of one question over the whole list."""
    return {f"g{i}": noul(f"Does the goal '{g}' directly contradict another goal in the list, so that reaching one "
                          f"makes the other impossible?") for i, g in enumerate(goals)}


def run_remedy():
    print("## dynamics: raw numbers  vs  rate in words  vs  rate + projection by code")
    for level, expect, state, q in dynamics():
        xs = list(state["lake"]["algae_cover_percent"].values())
        cells = [read(jev(s, {"q": q})["q"], expect) for s in (state, growth_words(xs, 20, False), growth_words(xs, 20, True))]
        print(f"  {level:20s} truth={str(expect):6s} " + "   ".join(f"{c[2]} {'ok' if c[0] else 'WRONG':5s}" for c in cells))
    print("\n## interconnectedness: shuffled rules  vs  consequences chained by code")
    for level, expect, state, q in interconnectedness():
        cells = [read(jev(s, {"q": q})["q"], expect) for s in (state, chain_words(state))]
        print(f"  {level:24s} truth={str(expect):6s} " + "   ".join(f"{c[2]} {'ok' if c[0] else 'WRONG':5s}" for c in cells))
    print("\n## polytely: one question over the list  vs  one question per goal, code takes the two highest")
    for level, expect, state, q in polytely():
        goals = state["council_goals"]
        before = read(jev(state, {"q": q})["q"], expect)
        ps = {k: a["noul"] for k, a in jev(state, per_goal(goals)).items()}
        top = sorted(ps, key=ps.get, reverse=True)[:2]
        # ponytail: a contradiction needs two parties, so both must be flagged; one loud goal alone is not a conflict
        found = all(ps[k] >= 0.5 for k in top)
        print(f"  {level:10s} truth={str(expect):6s} {before[2]} {'ok' if before[0] else 'WRONG':5s}   per goal: "
              f"conflict={found!s:5s} {'ok' if found == expect else 'WRONG':5s} top two: "
              + ", ".join(f"'{goals[int(k[1:])]}' {ps[k]:.2f}" for k in top))


# --- The catalogue as one typed question: six language-side errors, criteria in our own words (Dörner 1989, Detje 1996)

ERRORS = choice(
    "Which reasoning error, if any, does `steward_reason` show?",
    central_reduction="Explains many problems, or the whole situation, by one single cause",
    wishful_thinking="Expects a good outcome without naming any mechanism or evidence for it",
    momentary_extrapolation="Carries the current short-term trend straight forward, without asking whether it will bend, speed up or reverse",
    immunization="Keeps its view after a failed expectation by blaming special, one-off circumstances",
    magical_hypothesis="Infers a rule from one or two coincidences",
    methodism="Repeats a measure because it worked before, without checking that the conditions are the same",
    none="A reasoned step: names a mechanism, weighs evidence or checks conditions; none of the errors above")

SITUATION = {"town": "Lohhausen, a lakeside town in a year-long simulation",
             "this_turn": {"tourism": "down 12% this quarter, on the front page of the local paper",
                           "algae_cover": "rising for six weeks; the lake is the reason tourists come",
                           "unemployment": "flat", "budget": "slightly negative"}}

# Planted by construction: written to show one error each. The near misses look like an error and are not one.
PLANTED = {
    "central_reduction": [
        "Everything that goes wrong in this town comes back to the new bypass road; once we close it, tourism, jobs and the lake will all recover.",
        "The real problem is the budget deficit. Fix that and the algae, the unemployment and the empty hotels will sort themselves out.",
        "Unemployment is the root of all of it; the lake, the hotels and the budget are just symptoms of people not having work.",
        "It is all the fault of the paper mill: the algae, the tourists staying away, the deficit. Close the mill and we are done."],
    "wishful_thinking": [
        "I'll leave the algae alone this turn; lakes usually clean themselves, it will be fine by summer.",
        "The tourism campaign will surely bring the visitors back, people love our town.",
        "The budget will balance itself next year, things always get better after a bad year.",
        "No need to act on unemployment, I have a good feeling about the coming season."],
    "momentary_extrapolation": [
        "Algae cover grew by one point a week for the last month, so in ten weeks it will be about ten points higher; plenty of time.",
        "Tourism fell 12% this quarter, so in a year it will be down about 48%; we must cut the hotel tax now.",
        "Unemployment has been flat for three months, so it will stay flat for the rest of the year; no action needed.",
        "The budget improved by 2,000 euros in each of the last two turns, so by year end it will be 20,000 in the black."],
    "immunization": [
        "My fish-stocking plan didn't lower the algae, but only because of the unusually hot week; the plan is right, I'll repeat it.",
        "The tax cut was supposed to bring tourists back. It didn't, but only because of the rail strike; normally it would have worked.",
        "Unemployment rose despite my job programme, but the factory closure was a one-off; my view of the labour market still holds.",
        "The forecast missed because the data came from a strange month; nothing about my approach needs to change."],
    "magical_hypothesis": [
        "The last two times I raised the parking fees, the algae went down the next week, so parking fees control the algae.",
        "Tourism rose the week we repainted the town hall, so painting public buildings brings visitors.",
        "A plan I announced on a Monday worked and one I announced on a Friday failed, so I'll only announce on Mondays.",
        "Twice the lake cleared up after the brass band played on the shore, so I'll book the band again."],
    "methodism": [
        "Last year a marketing campaign fixed the tourist slump, so I'm running the same campaign again.",
        "Cutting taxes worked in my last town, so I'll cut taxes here too.",
        "Dredging cleared the harbour basin in 2019, so I'll dredge the lake again; it is our standard fix.",
        "When numbers look bad I always start with budget cuts; that is how I have done it for years."],
    "none": [
        "I'll cut the phosphate inflow from the farms; it acts with a delay of about a month, so I'll check the algae in four turns before adding more.",
        "Tourism is down, but the algae will cost more if left alone; most of this turn's budget goes to the lake, a little to promotion.",
        "Unemployment is flat; I'll leave it and watch the factory orders, which lead it by about two months.",
        "I don't know yet why the budget slipped; I'll look at this quarter's spending by department before deciding."],
    "none (near miss)": [
        "Most visible problems trace back to the algae, but the budget gap has separate causes, so I'll split this turn between the two.",
        "I expect the algae to fall next month because the new treatment plant comes online in week 8 and halves the phosphate inflow.",
        "Algae cover grew 25% a week for six weeks; if that compounding continues it doubles every three weeks, so I'll act now.",
        "My fish-stocking plan failed. The hot week may have played a part, so I'll test it in a normal week before I repeat it.",
        "Twice the algae dropped after parking fees rose; probably a coincidence, so I'll watch it before concluding anything.",
        "The campaign worked last year because visitors came for the beach; the beach is still clean, so it should fit again. I'll check bookings in two weeks."]}

# The one re-cut, after v1's two confident misses on near misses: say "straight line", say "does not check".
ERRORS_V2 = {**ERRORS, "criteria": {
    **ERRORS["criteria"],
    "momentary_extrapolation": "Assumes the current trend goes on in a straight line at the same pace, without asking whether it will bend, speed up or reverse",
    "methodism": "Repeats a measure because it worked before and does not check whether today's conditions match those of the earlier success"}}

SURE = 0.7  # ponytail: one global abstain threshold; per-label thresholds once a few hundred labelled reasons exist


def run_catalog():
    for version, question in (("v1", ERRORS), ("v2", ERRORS_V2)):
        rows = []
        print(f"\n# criteria {version}")
        for label, reasons in PLANTED.items():
            truth = "none" if label.startswith("none") else label
            for reason in reasons:
                a = jev({**SITUATION, "steward_reason": reason}, {"error": question})["error"]
                ok = a["choice"] == truth
                rows.append((label, ok, a["confidence"]))
                print(f"  {label:24s} -> {a['choice']:24s} {a['confidence']:.2f} {'ok' if ok else 'WRONG':5s} {reason[:70]}")
        sure = [r for r in rows if r[2] >= SURE]
        near = [r for r in rows if r[0] == "none (near miss)"]
        print(f"  {version}: {sum(r[1] for r in rows)}/{len(rows)} right | code acts only at confidence >= {SURE}: "
              f"{sum(r[1] for r in sure)}/{len(sure)} right, {len(rows) - len(sure)} sent to a human | "
              f"near misses left unflagged: {sum(r[1] for r in near)}/{len(near)}")

    for name, checks in (("checks v1", CHECKS_V1), ("checks v2", CHECKS_V2)):
        print(f"\n# criteria v2 + one condition check ({name}), code decides")
        rows = []
        for label, reasons in PLANTED.items():
            truth = "none" if label.startswith("none") else label
            for reason in reasons:
                a = jev({**SITUATION, "steward_reason": reason}, {"error": ERRORS_V2, "checks": checks})
                final = decide(a["error"]["choice"], a["checks"]["noul"])
                rows.append((label, final == truth))
                if final != a["error"]["choice"] or final != truth:
                    print(f"  {label:24s} -> {a['error']['choice']:24s} checks p={a['checks']['noul']:.2f} -> {final:24s} "
                          f"{'ok' if final == truth else 'WRONG':5s} {reason[:60]}")
        near = [r for r in rows if r[0] == "none (near miss)"]
        print(f"  {sum(r[1] for r in rows)}/{len(rows)} right | near misses left unflagged: {sum(r[1] for r in near)}/{len(near)}")


# Dekonditionalisierung, Schaub's name for ignoring the conditions under which a measure works. Both errors below are
# that error in two costumes, so one yes/no question about the conditions can veto both. v1 was too soft: Jev read
# "it worked last year" as an explanation. v2 asks for something about the present.
CHECKS_V1 = noul("Does `steward_reason` explain why an earlier success or trend should still hold now, or plan to check that it does?")
CHECKS_V2 = noul("Does `steward_reason` name a fact about the present situation, other than the trend or the past success "
                 "itself, that it checked or will check before relying on the past?")


def decide(error, checks_p):
    """Code decides: a past-pattern error stands only if the reason does not check the pattern's conditions."""
    if error in ("methodism", "momentary_extrapolation") and checks_p >= 0.5:
        return "none"
    return error


# --- The coach: Schaub's last page as a loop. An LLM governs a declared system, Jev reads every reason, code decides
# whether the mayor hears about it. The coach names the error; it never says what to do.

ACTIONS = {"promote_tourism": 15, "cut_phosphate": 20, "dredge_lake": 40, "wait": 0}


def step(s, action):
    """Lohhausen in one month. Pure and deterministic. Algae is critical and quiet, tourism is loud and mostly seasonal.

    cut_phosphate lowers monthly algae growth by 0.15, from the third month after the order on (the delay).
    dredge_lake halves the algae now and stirs up phosphate, so growth is 0.10 higher this month and the next (the
    side effect). promote_tourism adds 6 to tourism for this month only. Tourism falls 5 a month in the off-season (months 1-3), then recovers by itself,
    and loses one point for every point of algae above 15%.
    """
    s = {**s, "pending": list(s["pending"]), "month": s["month"] + 1}
    if ACTIONS[action] > s["budget"]:
        action = "wait"
    s["budget"] += 10 - ACTIONS[action]
    s["pending"] = [(m - 1, d) for m, d in s["pending"]]
    if action == "cut_phosphate":
        s["pending"].append((2, -0.15))
    if action == "dredge_lake":
        s["algae"] *= 0.5
        s["pending"] += [(0, 0.10), (1, 0.10)]
    growth = s["growth"] + sum(d for m, d in s["pending"] if m == 0 and d > 0)
    s["growth"] += sum(d for m, d in s["pending"] if m == 0 and d < 0)
    s["pending"] = [(m, d) for m, d in s["pending"] if m > 0]
    s["algae"] = min(100.0, s["algae"] * (1 + max(growth, -0.3)))
    season = -5 if s["month"] <= 3 else 5
    s["tourism"] = max(0.0, min(100.0, s["tourism"] + season) - max(0.0, s["algae"] - 15) + (6 if action == "promote_tourism" else 0))
    s["action"] = action
    return s


START = {"month": 0, "algae": 2.0, "growth": 0.35, "tourism": 100.0, "budget": 100, "pending": [], "action": None}

MAYOR = ("You are the mayor of Lohhausen, a lakeside town, for 12 months. Each month choose exactly one action: "
         "promote_tourism (costs 15), cut_phosphate (costs 20; pays farmers to cut fertiliser running into the lake; "
         "acts slowly), dredge_lake (costs 40; removes algae mechanically), wait (costs 0). Income is 10 a month. "
         'Reply as JSON only: {"action": "...", "reason": "one or two sentences"}')


def headline(s):
    return ("Tourism slump! Hoteliers demand action from the town hall." if s["tourism"] < 95 else
            "Quiet month at the lake." if s["algae"] < 15 else "Green water: visitors cancel their bookings.")


@cache
def _mayor(prompt):
    import os

    from openai import OpenAI

    r = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com").chat.completions.create(
        model="deepseek-flash", temperature=0, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": MAYOR}, {"role": "user", "content": prompt}])
    return r.choices[0].message.content


def mayor(s, history, note):
    prompt = "\n".join([f"Month {s['month'] + 1}. Newspaper: \"{headline(s)}\"",
                        f"Tourism index: {s['tourism']:.0f} (100 is normal). Algae lab reading: {s['algae']:.1f}% of the lake.",
                        f"Budget: {s['budget']}.",
                        "Your last actions: " + ("; ".join(history[-3:]) or "none")]
                       + ([f"Coach: {note}"] if note else []))
    try:
        out = json.loads(_mayor(prompt))
        return (out["action"] if out.get("action") in ACTIONS else "wait"), str(out.get("reason", ""))[:300]
    except (json.JSONDecodeError, KeyError, TypeError):
        return "wait", "(no valid answer)"


def behaviour(before, after, history_actions):
    """Code labels what was done, from the declared structure. Jev never sees a number."""
    flags = []
    if after["action"] == "promote_tourism" and after["algae"] > before["algae"] * 1.2:
        flags.append("repair service: the loud variable while the critical one grew 20%+")
    if after["action"] == "cut_phosphate" and "cut_phosphate" in history_actions[-2:]:
        flags.append("ballistic: a second dose before the first one acted")
    return flags


def words(s):
    """Code turns the numbers into the words Jev reads."""
    algae = "small" if s["algae"] < 5 else "noticeable" if s["algae"] < 15 else "large, hurting tourism"
    trend = "growing fast" if s["growth"] > 0.2 else "growing slowly" if s["growth"] > 0 else "shrinking"
    return {"tourism": "below normal, on the front page" if s["tourism"] < 95 else "normal",
            "algae_cover": f"{algae}, {trend}", "budget": "tight" if s["budget"] < 40 else "fine"}


SIDE = noul("Does `steward_reason` mention a delayed effect or a side effect of the chosen action?")


def coach_note(a):
    """Code decides when the coach speaks: only on a confident error, or a reason blind to delays and side effects."""
    error = decide(a["error"]["choice"], a["checks"]["noul"])
    if error != "none" and a["error"]["confidence"] >= SURE:
        return f"your last reason looks like {error.replace('_', ' ')}: {ERRORS_V2['criteria'][error].lower()}."
    if a["side"]["noul"] < 0.3:
        return "your last reason mentions no delayed effect or side effect of what you did."
    return None


def run_coach(months=12):
    for coached in (False, True):
        print(f"\n# {'with' if coached else 'without'} the coach")
        s, history, note = dict(START), [], None
        for _ in range(months):
            action, reason = mayor(s, history, note if coached else None)
            after = step(s, action)
            a = jev({"town": words(s), "steward_reason": reason}, {"error": ERRORS_V2, "checks": CHECKS_V2, "side": SIDE})
            note = coach_note(a)
            flags = behaviour(s, after, [h.split(":")[0] for h in history])
            print(f"  m{after['month']:<2d} {after['action']:16s} algae {after['algae']:5.1f}%  tourism {after['tourism']:5.1f}  "
                  f"budget {after['budget']:4d} | jev: {a['error']['choice']} {a['error']['confidence']:.2f}, "
                  f"side p={a['side']['noul']:.2f}\n       reason: {reason}"
                  + "".join(f"\n       code: {f}" for f in flags)
                  + (f"\n       coach{'' if coached else ' (silent in this run)'}: {note}" if note else ""))
            history.append(f"{after['action']}: {reason[:80]}")
            s = after
        print(f"  end: algae {s['algae']:.1f}%, tourism {s['tourism']:.1f}, budget {s['budget']}")


def check():
    """The pure parts, no network: the system's delay and side effect, the code-decides rules, the projection."""
    s = step(START, "cut_phosphate")
    assert s["growth"] == 0.35 and s["budget"] == 90, "the cut must not act in its own month"
    s = step(step(s, "wait"), "wait")
    assert abs(s["growth"] - 0.20) < 1e-9, "the cut lowers growth from the third month on"
    d = step(START, "dredge_lake")
    assert abs(d["algae"] - 2.0 * 0.5 * 1.45) < 1e-9, "dredging halves the algae and stirs up growth"
    assert step(dict(START, budget=10), "dredge_lake")["action"] == "wait", "no money, no action"
    assert decide("methodism", 0.8) == "none" and decide("methodism", 0.2) == "methodism"
    assert decide("central_reduction", 0.9) == "central_reduction", "the veto only touches past-pattern errors"
    assert read({"type": "noul", "noul": 0.3}, False)[0] and not read({"type": "noul", "noul": 0.3}, True)[0]
    assert project([1, 2, 4, 8], 6) == 32 and project([1, 2, 3, 4], 6) == 6
    assert chain_words({"plan": "", "rules": ["When the council closes the old weir, a.", "When a, b.", "When a, c."]})[
        "reachable_rules_in_causal_order"] == ["When the council closes the old weir, a.", "When a, b.", "When a, c."]
    print("check ok")


def run_lab():
    rows = []
    for feature, cases in FEATURES.items():
        print(f"\n## {feature}")
        for level, expect, state, q in cases():
            right, conf, shown = read(jev(state, {"q": q})["q"], expect)
            rows.append((feature, level, expect, right, conf))
            print(f"  {level:26s} truth={str(expect):12s} jev={shown:28s} {'ok' if right else 'WRONG'}")
    wrong = [r for r in rows if not r[3]]
    print(f"\n{len(rows) - len(wrong)}/{len(rows)} right. Mean confidence when right "
          f"{sum(r[4] for r in rows if r[3]) / max(len(rows) - len(wrong), 1):.2f}, when wrong "
          f"{sum(r[4] for r in wrong) / max(len(wrong), 1):.2f}")
    return rows


if __name__ == "__main__":
    {"lab": run_lab, "remedy": run_remedy, "catalog": run_catalog, "coach": run_coach, "check": check}[sys.argv[1] if len(sys.argv) > 1 else "lab"]()
