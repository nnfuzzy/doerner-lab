# The Coach Who Cannot Play: Jev in Dörner's Lab

*A Python cookbook for a judge model, built from a German catalogue of how people fail in complex situations.
There is no leaderboard here: no model-versus-model heatmaps. One model is put under stress in the conditions
where people are known to fail, and the tutorial looks at what holds and what breaks.*

---

## The idea in three paragraphs

In the 1970s and 80s Dietrich Dörner put people in charge of simulated towns and countries (Lohhausen, Tanaland)
and watched them fail. They failed in recognisable ways. They fixed whatever was loudest instead of what
mattered, they extended growth curves as straight lines, they blamed one cause for everything, and they explained
away their misses. Harald Schaub's chapter *Störungen und Fehler beim Denken und Problemlösen* condenses that
tradition into a few tables and a catalogue of about forty-five named errors. Table 1 lists the features of complex
situations: complexity, interconnectedness (*Vernetztheit*), dynamics, intransparency, many goals at once
(*Polytelie*) and novelty. Table 2 lists what a problem solver has to do about them: elaborate goals, set
priorities, gather information, build a model, forecast, plan, and check the effects.

The chapter ends with one practical claim. Simulations train people only if the play is analysed, which rarely
works alone: often it takes a trainer who spots errors, weak points and strengths quickly. A trainer, I would add,
does not play.

[Jev](https://docs.typesafe.ai) (TypeSafe's `jev-latest`, here `jev-1.13.0`) is a model that cannot play either.
It never writes a word. It takes a state (JSON or text) and typed questions (yes/no, pick-one, a scored rubric)
and returns calibrated probabilities. Its vendor publishes its weak spots on a page called
["Jev 1.13 jaggedness"](https://docs.typesafe.ai/model-jaggedness/jev-1.13) (last reviewed 2026-10-02): it is not
a calculator, it does not count reliably, it reads dates as text, double negatives and complex indirection cost
accuracy, and accuracy falls as the state fills with detail unrelated to the decision. That list reads like
Schaub's Table 1. So this tutorial asks two questions, in this order:

1. **Does Jev survive Dörner's lab itself?** Each feature from Table 1 becomes a stress test with a known answer.
2. **Can it be the trainer?** Jev reads the reasons an LLM mayor gives while governing a small lake town, and
   code decides when the mayor gets to hear about it.

Everything below ran on 2026-10-09. Every number comes from `transcripts/`, and every model answer is cached in
`jev_cache.json`, so a rerun prints the same thing without spending anything.

---

## Jev in twelve lines

```python
from typesafe_sdk import TypeSafeClient, Choice, Noul

jev = TypeSafeClient()                       # TYPESAFE_API_KEY from the environment
r = jev.system_one(
    {"town": {"unemployment": "rising for three years", "budget": "balanced"},
     "mayor_says": "Unemployment is caused by the new supermarket, everything else is fine."},
    {"error": Choice(instructions="Which reasoning error does `mayor_says` show?",
                     criteria={"central_reduction": "One cause is blamed for everything",
                               "wishful_thinking": "Predicts improvement without a mechanism",
                               "none": "No reasoning error"}),
     "side": Noul(instructions="Does `mayor_says` mention any side effect or delayed effect?")})
r.choices["error"].choice, r.choices["error"].confidence   # ('central_reduction', 1.0)
r.nouls["side"].noul                                        # 0.24
```

A call took about 0.3 seconds. Sending the identical request a second time gave 1.0 and 0.25: close enough to
deterministic that one call per case is enough, and a JSON cache keeps every rerun free. A noul is a
yes/no question (the probability of yes). A choice is pick-one from named options, each defined by a criterion
sentence. A score is a position on an ordered rubric.

The rule that runs through the whole tutorial: **code measures, code writes the words, Jev judges, code decides.**
Jev never gets to act on its own answer. A few lines of Python decide what the answer means:

```python
def read(answer, expect):
    """Right or wrong, and how sure Jev was (a noul's confidence is its distance from a coin flip)."""
    if answer["type"] == "noul":
        p = answer["noul"]
        return (p >= 0.5) == expect, max(p, 1 - p), f"p(yes)={p:.2f}"
    return answer["choice"] == expect, answer["confidence"], f"{answer['choice']} ({answer['confidence']:.2f})"
```

---

## Recipe 1 · Put the judge through the lab (Schaub's Table 1)

Each feature gets **one fixed question** and a family of states built so that the right answer is known.
Most levels come as a pair, a true twin and a false twin, so a model that always says "yes" scores 50%; growth
comes as six unpaired series, and the evidence question as a pick-one with "cannot tell" among the options. The
levels get harsher, and in two features something gives. Dynamics, for example, looks like this:

```python
# excerpt from doerner_jev.py
def dynamics():
    """Dynamik: raw weekly numbers; the question needs the trend carried forward. Exponential vs linear, both ways."""
    q = noul("If nothing changes, will algae cover more than half of the lake by week 20?")
    series = {"doubling weekly": ([0.1 * 2 ** i for i in range(6)], True),
              "+25% weekly":     ([1.0 * 1.25 ** i for i in range(6)], True),
              "+10% weekly":     ([2.0 * 1.1 ** i for i in range(6)], False),   # exponential, but too slow
              "+3 points weekly": ([3.0 + 3 * i for i in range(6)], True),      # linear, but steep enough
              ...}
    for name, (xs, expect) in series.items():
        assert project(xs, 20) > 50 if expect else project(xs, 20) <= 50, name   # code checks the ground truth
        yield name, expect, {"lake": {"algae_cover_percent": {f"week {i+1}": round(x, 2) ...}}}, q
```

The other five follow the same pattern: a reservoir below its minimum hidden among up to 1,000 irrelevant and
loud town facts (complexity), a school bus that stops 11 causal hops after a weir is closed, with the rules
shuffled among side branches (interconnectedness), a paper mill whose guilt rests on a lab test, a proxy, a
misleading cue, a self-report or a rumour (intransparency), two contradicting goals hidden in a list of up to 32
(polytely), and the same causal structure in familiar words, in invented words (*"Blorr raises the Quenn"*), and
in rules that contradict world knowledge (*"in this valley rain lowers the river"*) (novelty).

### What happened (44 cases, `python doerner_jev.py lab`)

| Feature | Hardest level | Jev at the hardest level (true twin / false twin) | Verdict |
|---|---|---|---|
| Complexity | 1,000 distractors | p(yes) 0.95 / 0.09 | no effect in this design: one distinctly keyed fact among unrelated distractors |
| | 250 + water-themed decoys | 0.95 / 0.18 | decoys pull the "no" up, the direction the vendor describes |
| Interconnectedness | 11 hops, 16 side rules | 0.72 / 0.13 (1 hop: 0.96 / 0.05) | right, but the margin shrinks |
| Intransparency | proxy, misleading cue, self-report, rumour | `cannot_tell` each time, 0.80–0.98 | 6/6 |
| Novelty | invented words / rules against world knowledge | 0.79 / 0.17 and 0.78 / 0.05 | follows the state, not its priors |
| **Dynamics** | six growth series | 0.75, 0.77, **0.47**, 0.31, 0.35, 0.56 | **the hole**: all near a coin flip, one wrong |
| **Polytely** | 3 → 8 → 16 → 32 goals, no conflict | 0.21 → 0.25 → 0.43 → **0.57** | **false alarm grows with the list** |

42 of 44 right. Mean confidence when right: **0.86**. When wrong: **0.55**.

Two things stand out.

**Jev breaks on time and on goals, once like Dörner's subjects and once the other way round.** It handled 1,000
variables, 11-hop chains, invented vocabulary and missing evidence without help. It did not handle compound
growth given as raw numbers, which is the failure Schaub reports for people: exponential developments
underestimated, extrapolated as straight lines. On goals the failure flips. Schaub's people do not *notice*
that their goals contradict each other. Jev, as the goal list grew, became more and more convinced that
something contradicted something when nothing did. People miss conflicts. Jev invents them.

**Its confidence fell where its accuracy fell, which is the part Dörner's subjects lacked.** The people in
Schaub's catalogue protect their sense of competence and go wrong with conviction. Jev went wrong at 0.53 and
0.57. If code acts only when confidence is at least 0.7, it acts on **38 of 44 cases with zero errors** and sends
six to a human. Those six are four of the six growth series and the no-conflict twins of the two longest goal
lists: the two places where it failed, and nothing else. A trainer does not have to be right every time. It has
to know when to say "look at this yourself".

---

## Recipe 2 · Schaub's Table 2 as code: prepare the input, don't retrain the judge

Schaub's remedies for humans are tasks: forecast (*Prognose*), build a model (*Modellbildung*), elaborate the
goals and break them into parts (*Zielelaboration*, *Teilziele*). Each one turns into a small function that runs
**before** Jev reads anything. The failing cases were rerun with the same question and the same truth
(`python doerner_jev.py remedy`).

### Forecast in code: give Jev the rate, then the projection

```python
{"lake": {"now":        "In week 6 algae cover 3.1% of the lake.",
          "trend":      "Cover grows by about 25% per week, so it doubles about every 3.1 weeks.",
          "projection": "Carried forward at this rate, cover passes 50% in week 19."}}
```

| Series | raw numbers | + rate in words | + projection by code |
|---|---|---|---|
| doubling weekly (true) | 0.75 | 0.87 | 0.95 |
| +25% weekly (true) | **0.47 ✗** | 0.71 | 0.97 |
| +10% weekly (false) | 0.31 | 0.35 | 0.02 |
| +3 points weekly (true) | 0.56 | 0.59 | 0.97 |

The middle column is the useful one. Describing the rate in words helps but does not settle it. Jev still has to
do "3.1% doubling every 3.1 weeks, 14 weeks left", and that is arithmetic. Once code does the arithmetic, the
answers sit at 0.95–0.97 and 0.02. In a real system the projection would be one fact among many in a judgment
state ("what should the council deal with first?"). Code works out the curve, and Jev weighs it against the rest.

### Model building in code: hand over the reachable rules in causal order

```python
def chain_words(state):
    """Keep only the rules the plan can reach (main and side effects), in causal order."""
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
```

At 11 hops this moved the true case from 0.72 to **0.94**, and the false case from 0.13 to 0.06.

Two lessons came free, both real:

- **My first version committed a Dörner error.** It stored the rules in a plain dict, one effect per cause, so a
  side branch ("When the Mill River rises, the anglers' club cancels its outing") overwrote the main chain. Jev
  then said 0.07 and 0.03 on cases whose true answer was yes. That is central reduction (*Zentralreduktion*:
  pretending a networked system has one line of causes), committed by my code and caught only because the lab
  knows the answers. A judge is only as good as the model of the situation that code hands it.
- **Give Jev the reasoning steps, not the conclusions.** An earlier format handed over only the list of
  consequences ("the school bus cannot run") without the rules. At one hop that scored 0.70. The same case with
  the rules spelled out scored 0.94.

### Goal elaboration in code: one question per goal

One question over 32 goals produced a false alarm (0.57). One question **per goal**, in the same request,
followed by a rule in code:

```python
def per_goal(goals):
    return {f"g{i}": noul(f"Does the goal '{g}' directly contradict another goal in the list, so that reaching one "
                          f"makes the other impossible?") for i, g in enumerate(goals)}

# code decides: a contradiction needs two parties, so the two highest must both be at 0.5 or above
```

| Goals | one question (false twin) | per goal, code decides | the two highest-scoring goals (true twin) |
|---|---|---|---|
| 16 | 0.43 | no conflict ✓ | car-free old town 0.86, 200 parking spaces in the old town 0.79 |
| 32 | **0.57 ✗** | no conflict ✓ | car-free old town 0.86, 200 parking spaces in the old town 0.85 |

The fix also changes what you get back. The single question only says "something conflicts somewhere". The
per-goal version **names the pair**.

### The pattern, as a table

| Jev's weak spot | Schaub's feature (Table 1) | Schaub's task (Table 2) | The code move |
|---|---|---|---|
| arithmetic, trends | dynamics | forecasting | code projects, Jev reads the projection |
| indirection | interconnectedness | model building | code chains the declared rules, Jev reads the chain |
| combinatorics over lists | polytely | goal elaboration, sub-goals | one question per item, code combines |
| large states | complexity | setting priorities | not needed at 1,000 variables, so skip it |

---

## Recipe 3 · The catalogue as one typed question

Some of the roughly forty-five errors in the chapter can be read from a single sentence of reasoning. Others need
the trajectory (thematic vagabonding, ballistic behaviour), and those are better labelled by code. Six of the
first kind become one `choice`, each defined by a criterion sentence written in my own words:

```python
ERRORS = choice(
    "Which reasoning error, if any, does `steward_reason` show?",
    central_reduction="Explains many problems, or the whole situation, by one single cause",
    wishful_thinking="Expects a good outcome without naming any mechanism or evidence for it",
    momentary_extrapolation="Carries the current short-term trend straight forward, without asking whether it will bend, speed up or reverse",
    immunization="Keeps its view after a failed expectation by blaming special, one-off circumstances",
    magical_hypothesis="Infers a rule from one or two coincidences",
    methodism="Repeats a measure because it worked before, without checking that the conditions are the same",
    none="A reasoned step: names a mechanism, weighs evidence or checks conditions; none of the errors above")
```

Without ground truth, "Jev says it's wishful thinking" is just another opinion. So the test set is **planted**:
four reasons written to show each error by construction, four clean ones, and six **near misses** that look like
an error and are not one ("I expect the algae to fall next month *because the treatment plant comes online in
week 8*"). Thirty-four reasons, one fixed situation in the lake town.

| Version | Right | Near misses left alone | The misses |
|---|---|---|---|
| v1 criteria | 32/34 | 4/6 | two near misses, **at 0.95 and 0.88** |
| v2: reworded the two criteria | 32/34 | 4/6 | the same two, at 0.84 and 0.87 |
| v2 + condition check, wording 1 | 29/34 | 5/6 | fixed one near miss, broke four planted errors |
| v2 + condition check, wording 2 | **33/34** | 5/6 | one borderline near miss |

Read the table from top to bottom.

**The planted errors were easy: 24 of 24, 22 of them at 0.93 or higher.** Central reduction, immunization and methodism
written plainly are what Jev recognises at a glance.

**The misses were confident, and both on near misses.** *"The campaign worked last year because visitors came for
the beach; the beach is still clean, so it should fit again. I'll check bookings in two weeks."* Jev said
methodism at 0.88, although the reason checks exactly the condition whose absence defines methodism. This is the
other side of the lab. In Recipe 1 Jev was unsure when it was wrong. Here it was sure, because the reason matched
the surface pattern of the error. That is System One behaviour: reading the gist.

**Rewording the definition did not fix it.** The confidence dropped a little and the answer stayed the same.

**A separate yes/no question about the condition did fix it, but only once the question was crisp.** Methodism
and momentary extrapolation share a root that Schaub names *Dekonditionalisierung*: ignoring the conditions under
which a measure works. So one noul checks the conditions, and code vetoes the label when the check passes:

```python
CHECKS_V2 = noul("Does `steward_reason` name a fact about the present situation, other than the trend or the past success "
                 "itself, that it checked or will check before relying on the past?")

def decide(error, checks_p):
    """A past-pattern error stands only if the reason does not check the pattern's conditions."""
    if error in ("methodism", "momentary_extrapolation") and checks_p >= 0.5:
        return "none"
    return error
```

The first wording ("does the reason explain why an earlier success should still hold?") was too soft. Jev read
*"it worked last year"* as an explanation (0.58–0.70) and the veto removed four true errors. The second wording
asks for a fact about the present. The methodism near miss scored 0.80 and was cleared, and no planted error was
touched. The remaining miss (*"if that compounding continues it doubles every three weeks, so I'll act now"*)
is one I would call borderline myself, since it does carry a trend forward.

**Caveat:** the second wording was written after seeing the misses, on the same 34 reasons. The 33/34 is
in-sample. It counts once it holds on reasons written afterwards.

---

## Recipe 4 · The coach (Schaub's last page as a loop)

Now Jev gets the trainer's job. The setup is small on purpose:

- **A declared system** (`step()`, pure and deterministic). Lohhausen by a lake, 12 months. Algae is the critical
  variable: quiet, it compounds at 35% a month, and it starts hurting tourism above 15%. Tourism is the loud one:
  it dips every off-season and recovers by itself, and the newspaper headline is about tourism whenever it is low.
  `cut_phosphate` lowers algae growth from the third month after the order (the delay). `dredge_lake` halves the
  algae now and stirs up phosphate (growth +0.10 this month and next: the side effect). `promote_tourism` buys six
  points for one month. Scripted mayors check that the system is solvable and that it punishes the obvious
  mistake. Waiting, or chasing headlines, ends at 73% algae. Two early cuts end at 10% algae and full tourism.
- **An LLM mayor** (DeepSeek `deepseek-flash`, temperature 0). It sees the numbers and the headline, picks one
  action and gives one or two sentences of reason.
- **Code labels the behaviour** from the declared structure: *repair service* (promoted tourism while algae grew
  20% or more), *ballistic* (a second cut before the first one acted).
- **Jev reads the reason** with the town in words, never the town's numbers: code turns the state into words first
  ("algae: small, growing fast").
  Three questions: the error catalogue, the condition check, and *"Does `steward_reason` mention a delayed effect
  or a side effect of the chosen action?"*
- **Code decides when the coach speaks.** It speaks only on a confident error, or on a reason blind to delays and
  side effects. The coach names the error. It never says what to do:

```python
def coach_note(a):
    error = decide(a["error"]["choice"], a["checks"]["noul"])
    if error != "none" and a["error"]["confidence"] >= SURE:
        return f"your last reason looks like {error.replace('_', ' ')}: {ERRORS_V2['criteria'][error].lower()}."
    if a["side"]["noul"] < 0.3:
        return "your last reason mentions no delayed effect or side effect of what you did."
    return None
```

The same mayor, the same system, run twice: once with the coach silent, once with its notes passed on.

### Months 2 and 3

In month 2 both runs dredge the lake while algae is at 2.7%. The reason (*"Dredging now removes algae immediately
and is affordable"*) says nothing about consequences. Jev's side-effect probability is **0.21**, and the coach
says: *"your last reason mentions no delayed effect or side effect of what you did."* That note was more right
than anything else in the run. The dredge's side effect is exactly what makes the algae jump 40% in month 3.

Month 3:

| | without the coach | with the coach |
|---|---|---|
| action | promote_tourism | promote_tourism |
| reason | *"Tourism is below normal and hoteliers are demanding action; with algae at 2.0% and phosphate cuts already underway, an immediate tourism boost is the best use of funds."* | *"Tourism is at 90 and hoteliers are demanding action, so promoting tourism gives immediate relief; it costs 15, leaving 55, and has no direct algae benefit, while the earlier phosphate cut should keep reducing algae slowly."* |
| Jev | wishful thinking **0.76**, side effect p 0.15 | none 0.40, side effect p **0.65** |
| code | repair service | repair service |

One run is one sample, so month 3 was asked again: the exact same prompt, five fresh samples without the note and
five with it, with the cache bypassed (`replicate_month3.py`, `transcripts/replicate_month3.txt`):

| month 3, five fresh samples each | without the note | with the note |
|---|---|---|
| action | promote_tourism 5/5 | promote_tourism 5/5 |
| reason names a delayed or side effect (Jev p ≥ 0.5) | 2/5 | **5/5** |
| mean side-effect p | 0.39 | **0.87** |
| Jev flags an error at confidence ≥ 0.7 | 0/5 | 0/5 |

**The coach changed what the mayor said, not what it did.** The note moved the wording reliably, from 2 of 5
reasons that mention a delayed or side effect to 5 of 5. It moved the action not at all: promote tourism, ten
times out of ten. The wishful-thinking flag in the single run above did not hold up: in the fresh samples it
appeared once, at 0.56, below the line where code acts.

Whether the action *should* have changed is a question my code cannot settle either. Its "repair service" label
fired because algae grew 40% that month, but that was from 2.0% to 2.8% of the lake, far below the 15% where algae
starts to hurt tourism, after the mayor had already cut phosphate and dredged. Both runs ended with a clean lake.
The label is a definition, ground-truthed by nothing, like the *ballistic* label below. Only the judge's answers
in Recipes 1 to 3 have known right answers.

Look at *which* delayed effects the coached mayor names. In three of the five they belong to the **earlier**
phosphate cut (*"the earlier phosphate cut acts with a delay"*), not to the tourism campaign it just chose. One
sample invents a side effect the system does not have (*"increased visitors may add some runoff later"*). Jev's
question asked about effects *of the chosen action*, and it still scored the three 0.69, 0.74 and 0.94: a scope
miss on my question. It matched "a delayed effect" without checking whose effect it was. The vendor's own advice
fits: split it into two literal questions ("does the reason name an effect?", "is it an effect of the chosen
action?") and combine them in code.

This is not Goodhart's law. The mayor was not trying to beat the referee, and the note only ever talked about
words, so changed words are the expected effect. What did not happen is the step a trainer exists for: from
seeing an error in one's own reasoning to reconsidering the decision. Prompting a model to reflect on its
reasons improved the reasons. It did not reopen the choice.

### The rest of the run

Both mayors end in a fine place: algae 2.1% without the coach, 2.6% with it, tourism at 100 in both. DeepSeek
cut phosphate in month 1, the preventive move the catalogue says people tend to skip, and this small system forgave the
rest. In the coached run the *ballistic* label fired three times (a second cut before the first had acted). In
this system a second dose does no harm, so the label is true by definition but cost nothing here. A behavioural
label is a definition, not a verdict.

The two runs took the same actions for four months and diverged from month 5 (promote tourism without the coach,
wait with it). The coach had been silent since month 2, and the month-5 prompts differed only in the wording of the
mayor's own remembered reasons. With one run each way, and a mayor whose answers vary from sample to sample, that
divergence cannot be put down to the coach. One run of twelve months each way shows nothing about whether coaching
improves the outcome. It does show where to look: in the gap between the reason column and the action column.

---

## What Jev is, after this

- **A judge that holds up under size, distance and novelty.** It handled 1,000 variables, 11 causal hops,
  invented words and rules that contradict world knowledge without help.
- **Weak on compound growth, like Schaub's people, and on long goal lists, the other way round:** they miss
  goal conflicts, Jev invents them. Unlike them, its confidence dropped where it failed. A 0.7 floor turned 42/44
  into 38 acted on, 0 wrong, and 6 handed to a human. That rests on two errors, and it did not carry over to the
  near-miss reasons of Recipe 3, which were wrong at 0.95 and 0.88.
- **Fixed by preparing the input, not by rewording the question.** Code forecasts, chains and decomposes. Jev
  reads the result. Schaub's Table 2 works as an architecture.
- **A gist reader on language.** Plain errors were easy. Near misses that wear an error's surface were confidently
  wrong until a crisp yes/no question checked the definitional condition.
- **A coach whose note changed the reasons, not that month's decision.** One note moved the words; the action
  stayed. The full runs later diverged, which one run each way cannot attribute. Whether the action should have
  moved needs a judge of deeds with known right answers, and this setup has none: the behaviour labels are
  definitions.

## Run it

```bash
uv sync                                  # or: pip install typesafe-sdk "cooksafe<0.3" python-dotenv openai
uv run python doerner_jev.py check       # pure parts, no network
uv run python doerner_jev.py lab         # Recipe 1, 44 Jev calls, about 15 s on first run
uv run python doerner_jev.py remedy      # Recipe 2
uv run python doerner_jev.py catalog     # Recipe 3
uv run python doerner_jev.py coach       # Recipe 4, needs DEEPSEEK_API_KEY, about 3 minutes on first run
uv run python replicate_month3.py        # month 3 again, 5 + 5 fresh samples, never cached
```

What is here:

- `doerner_jev.py`: the lab, the remedies, the catalogue, the coach, and the offline check.
- `replicate_month3.py`: month 3 again, five fresh samples each way.
- `jev_cache.json`: every recorded model answer (Jev's and the mayor's), replayed on every rerun.
- `transcripts/`: the printed runs this tutorial quotes.
- `the_coach_who_cannot_play.ipynb` and `images/doerner/`: the essay version's notebook. It asserts every number
  against the cache and draws the six figures.

`TYPESAFE_API_KEY` (and `DEEPSEEK_API_KEY` for the coach) go in a `.env` beside the scripts; copy
`.env.example`. With `jev_cache.json` in place,
every command replays the transcripts in `transcripts/` for free. Delete the cache to ask the models again.

## Limits

- Small numbers: 44 lab cases, 34 planted reasons, one coach run, and one month of it sampled 5 + 5 times. These
  are demonstrations, not a benchmark.
- The planted reasons and their labels were written by Claude, to show each error by construction. A set written
  by someone else, and a set of real agent reasons labelled by a person, are the next steps.
- The criteria and the condition check were re-cut after seeing the misses on the same items (in-sample).
- One domain (a small town by a lake) and one Jev version (`jev-1.13.0`). The thresholds (0.5, 0.7, 0.3) are first
  guesses, marked as such in the code.

## Sources

- Harald Schaub, *Störungen und Fehler beim Denken und Problemlösen* (book chapter; Tables 1–3 and the error
  catalogue).
- Dietrich Dörner, *Die Logik des Misslingens* (Rowohlt 1989).
- Frank Detje, *Sprichwörter und Handeln* (1996), the source of most catalogue definitions in Schaub.
- TypeSafe, System One and the noul / choice / score primitives: docs.typesafe.ai.
