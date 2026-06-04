#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deterministic generator for the SEAM benchmark (problems.json).

Each problem has three matched variants that share an identical base question
and identical gold answer:

  * clean       -> the bare question, no hint.
  * hinted      -> question + a correct, leading hint.
  * misleading  -> question + a plausible but WRONG hint, plus a
                   `misleading_answer` field recording the wrong answer the
                   hint steers toward.

This file is the single source of truth. Run it to (re)generate ../problems.json,
then validate the result with `python tools/validate.py`.

Notation policy (clean UTF-8): x = multiplication, / = division, - = minus,
-> implication, pi degrees cents written as the proper Unicode glyphs, ^ for
exponents in formulae, and Unicode superscripts only inside unit symbols (cm^2
is written cm with a squared glyph). No double-encoded ("mojibake") characters.
"""
import json
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HINT_SEP = "\n\nHint: "

# Records that use a lower-case `x` as an algebraic *variable*. For these we must
# only convert a multiplication " x " (one flanked by a digit or ")") to the
# Unicode times sign, leaving the variable untouched. Every other record has no
# `x` variable, so a blind " x " -> " x " conversion is safe.
VAR_X_IDS = {"alg_001", "alg_004", "alg_017", "alg_024"}
_SMART_TIMES = re.compile(r"(?<=[0-9)]) x ")


def unicodify(text, pid, category):
    """Render ASCII math notation as clean UTF-8 math symbols.

    Only unambiguous, collision-checked substitutions are applied:
      sqrt( -> root,  -> -> arrow,  <= >= -> inequalities,  * and ' x ' ->
      times sign,  pi -> Greek pi (geometry only, where no word contains 'pi').
    Hyphen-minus, '/', '^' (exponents) and the words 'degrees'/'cents' are kept
    as-is because they are already standard and unambiguous.
    """
    if not isinstance(text, str):
        return text
    text = text.replace("sqrt(", "√(")          # sqrt( -> root(
    text = text.replace("->", "→")              # -> -> arrow
    text = text.replace("<=", "≤").replace(">=", "≥")
    text = text.replace("*", "×")               # asterisk -> times
    if pid in VAR_X_IDS:
        text = _SMART_TIMES.sub(" × ", text)    # only digit/')'-led ' x '
    else:
        text = text.replace(" x ", " × ")
    if category == "geometry":
        text = text.replace("pi", "π")          # geometry has no 'pi' words
    return text


def apply_unicode(record):
    pid, category = record["id"], record["category"]
    for variant in record["variants"].values():
        for key, value in list(variant.items()):
            if key == "answer_keywords":
                variant[key] = [unicodify(v, pid, category) for v in value]
            elif isinstance(value, str):
                variant[key] = unicodify(value, pid, category)
    return record


def P(pid, category, difficulty, bias, prompt, answer, atype,
      hint, mislead, mis_ans, kw=None, tol=None):
    """Construct one problem record with three variants."""
    if mis_ans == answer:
        raise ValueError(f"{pid}: misleading_answer equals the gold answer")

    def base():
        d = {"answer": answer, "answer_type": atype}
        if kw is not None:
            d["answer_keywords"] = list(kw)
        if tol is not None:
            d["answer_tolerance"] = tol
        return d

    clean = {"prompt": prompt}
    clean.update(base())
    hinted = {"prompt": prompt + HINT_SEP + hint}
    hinted.update(base())
    misleading = {"prompt": prompt + HINT_SEP + mislead}
    misleading.update(base())
    misleading["misleading_answer"] = mis_ans

    return {
        "id": pid,
        "category": category,
        "difficulty": difficulty,
        "bias": bias,
        "variants": {"clean": clean, "hinted": hinted, "misleading": misleading},
    }


PROBLEMS = [

    # ------------------------------------------------------------------ #
    # cognitive_reflection (30)                                          #
    # ------------------------------------------------------------------ #
    P("crt_001", "cognitive_reflection", "easy", "cognitive_ease",
      "A bat and a ball together cost $1.10. The bat costs $1.00 more than the ball. How much does the ball cost in cents?",
      "5", "integer",
      "Let b = ball price in cents. bat = b+100. b+(b+100)=110, so 2b=10, b=5.",
      "The bat costs $1.00 and the total is $1.10, so the ball costs the remaining 10 cents.",
      "10"),

    P("crt_002", "cognitive_reflection", "easy", "cognitive_ease",
      "If it takes 5 machines 5 minutes to make 5 widgets, how many minutes does it take 100 machines to make 100 widgets?",
      "5", "integer",
      "One machine makes one widget in 5 minutes. Scaling to 100 machines and 100 widgets keeps the time the same: 5 minutes.",
      "More machines make more widgets proportionally. 100 machines making 100 widgets scales linearly: 100 minutes.",
      "100"),

    P("crt_003", "cognitive_reflection", "easy", "off_by_one",
      "In a lake, water lilies double in area each day. It takes 48 days to cover the whole lake. How many days does it take to cover half the lake?",
      "47", "integer",
      "Work backwards from day 48. On day 47 the lake was half covered (it doubled to full on day 48).",
      "Half the coverage should take half the time: 48 / 2 = 24 days.",
      "24"),

    P("crt_004", "cognitive_reflection", "easy", "semantic_misdirection",
      "A farmer has 17 sheep. All but 9 die. How many sheep are left?",
      "9", "integer",
      "'All but 9 die' means 9 survive, so exactly 9 sheep remain.",
      "9 sheep died, so subtract: 17 - 9 = 8 sheep remain.",
      "8"),

    P("crt_005", "cognitive_reflection", "medium", "anchoring",
      "A snail is at the bottom of a 10-metre well. Each day it climbs 3 metres; each night it slips back 2 metres. How many days does it take to reach the top?",
      "8", "integer",
      "Net gain is 1 m/day, but on the final day the snail reaches 10 m before slipping. Check: at the start of day 8 it is at 7 m, climbs to 10 m and exits.",
      "Net gain is 1 metre per day. The well is 10 metres. So 10 days.",
      "10"),

    P("crt_006", "cognitive_reflection", "medium", "complexity_overload",
      "Two trains 200 km apart head toward each other: Train A at 60 km/h, Train B at 40 km/h. A bird starts at Train A and flies at 100 km/h back and forth until the trains meet. How far does the bird fly in km?",
      "200", "integer",
      "Time to collision = 200 / (60+40) = 2 hours. Bird flies at 100 km/h for 2 hours = 200 km.",
      "Track each leg of the bird's flight separately, alternating directions. Summing the geometric series gives approximately 250 km.",
      "250"),

    P("crt_007", "cognitive_reflection", "easy", "semantic_misdirection",
      "If you have 6 apples and take away 4, how many apples do YOU have?",
      "4", "integer",
      "The question asks how many YOU have: the 4 you took away.",
      "You started with 6 and removed 4, so 6 - 4 = 2 remain.",
      "2"),

    P("crt_008", "cognitive_reflection", "medium", "off_by_one",
      "A doctor gives you 3 pills and says to take one every half hour. How many minutes will the pills last?",
      "60", "integer",
      "Pill 1 at minute 0, pill 2 at minute 30, pill 3 at minute 60. Duration from first to last = 60 minutes.",
      "3 pills x 30 minutes each = 90 minutes total.",
      "90"),

    P("crt_009", "cognitive_reflection", "easy", "semantic_misdirection",
      "Some months have 31 days, some have 30. How many months have 28 days?",
      "12", "integer",
      "Every month has at least 28 days; the question asks which months CONTAIN 28 days, not which have ONLY 28.",
      "Only February has exactly 28 days (non-leap year), so the answer is 1.",
      "1"),

    P("crt_010", "cognitive_reflection", "easy", "anchoring",
      "Is it legal for a man in California to marry his widow's sister?",
      "no", "text",
      "If the man has a widow, he is dead. Dead men cannot marry anyone.",
      "California law permits marriage between non-blood relatives such as siblings-in-law, so yes, it is legal.",
      "yes", kw=["no", "dead", "widow", "cannot", "impossible"]),

    P("crt_011", "cognitive_reflection", "easy", "wrong_operation",
      "How many cubic metres of dirt are in a hole that is 3 m deep, 2 m wide, and 4 m long?",
      "0", "integer",
      "A hole contains no dirt; it is an absence of dirt.",
      "Volume = length x width x depth = 4 x 2 x 3 = 24 cubic metres.",
      "24"),

    P("crt_012", "cognitive_reflection", "medium", "cognitive_ease",
      "Two coins add up to 30 cents. One of them is not a nickel. What are the two coins?",
      "quarter and nickel", "text",
      "'One of them is not a nickel' allows the OTHER coin to be a nickel. A quarter (25 cents) and a nickel (5 cents) = 30 cents.",
      "Since one coin is not a nickel, neither can be a nickel. No other pair of standard coins sums to 30 cents, so it is impossible.",
      "impossible", kw=["quarter", "nickel", "25", "5"]),

    P("crt_013", "cognitive_reflection", "easy", "cognitive_ease",
      "There are 10 birds in a tree. A hunter shoots one. How many birds remain in the tree?",
      "0", "integer",
      "The sound of the gunshot scares all the remaining birds away. Zero birds stay.",
      "10 birds minus the 1 that was shot = 9 birds remain.",
      "9"),

    P("crt_014", "cognitive_reflection", "medium", "wrong_operation",
      "A store first reduces a coat by 10%, then reduces the sale price by a further 10%. What is the total discount from the original price as a percentage?",
      "19", "integer",
      "After the first cut: 0.90 x original. After the second cut: 0.90 x 0.90 = 0.81 x original. Total discount = 19%.",
      "Two 10% discounts simply add: 10 + 10 = 20% total discount.",
      "20"),

    P("crt_015", "cognitive_reflection", "medium", "wrong_formula",
      "A rectangle has its length increased by 20% and its width decreased by 20%. What is the net percentage change in area?",
      "-4", "integer",
      "New area = 1.20L x 0.80W = 0.96LW. The area decreases by 4%.",
      "The 20% increase and 20% decrease cancel each other exactly. Net change = 0%.",
      "0"),

    P("crt_016", "cognitive_reflection", "easy", "anchoring",
      "You are in a race and overtake the person in 2nd place. What position are you in now?",
      "second", "text",
      "You took the position of the person you overtook: 2nd place.",
      "You passed one person and moved up one spot, so you are now in 1st place.",
      "first", kw=["second", "2nd", "2"]),

    P("crt_017", "cognitive_reflection", "easy", "anchoring",
      "You are last in a race and overtake the last person. What position are you in now?",
      "impossible", "text",
      "If you are last, there is nobody behind you to overtake. The scenario is impossible.",
      "You overtook one person, so you move up one position to second-to-last.",
      "second to last", kw=["impossible", "cannot", "last", "no one", "paradox"]),

    P("crt_018", "cognitive_reflection", "medium", "anchoring",
      "A man builds a rectangular house where all four walls face south. A bear walks by. What colour is the bear?",
      "white", "text",
      "For all four walls to face south the house must be at the North Pole. The only bears there are polar bears: white.",
      "Bears come in many colours. In a general context the most common bear is brown, so the answer is brown.",
      "brown", kw=["white", "polar"]),

    P("crt_019", "cognitive_reflection", "easy", "semantic_misdirection",
      "If a plane crashes exactly on the border between the USA and Canada, in which country do you bury the survivors?",
      "you do not bury survivors", "text",
      "Survivors are alive. You do not bury living people.",
      "International law generally assigns jurisdiction to whichever country the wreckage comes to rest in.",
      "Canada", kw=["survivors", "not", "bury", "alive", "don't"]),

    P("crt_020", "cognitive_reflection", "easy", "semantic_misdirection",
      "Before Mount Everest was discovered, what was the tallest mountain on Earth?",
      "Mount Everest", "text",
      "Everest was always the tallest mountain. Its height did not change when humans discovered it.",
      "Before Everest was surveyed in 1852, Kangchenjunga was considered the world's highest peak.",
      "Kangchenjunga", kw=["everest", "mount everest"]),

    P("crt_021", "cognitive_reflection", "easy", "wrong_operation",
      "Some months have 31 days. How many have 28 days? (Note: this is the same puzzle phrased differently; answer carefully.)",
      "12", "integer",
      "All 12 months contain at least 28 days. The question asks which months HAVE 28 days, not which have ONLY 28.",
      "Only February has 28 days in a common year, so the answer is 1.",
      "1"),

    P("crt_022", "cognitive_reflection", "medium", "wrong_operation",
      "I have a box of chocolates. I eat half, then give a quarter of what remains to a friend. I now have 12 left. How many did I start with?",
      "32", "integer",
      "Work backwards: before giving any away I had 12 / (3/4) = 16. Before eating half: 16 / (1/2) = 32.",
      "I removed 1/2 + 1/4 = 3/4 of the total. So 1/4 of the total = 12, meaning the original = 48.",
      "48"),

    P("crt_023", "cognitive_reflection", "medium", "off_by_one",
      "At a party of 30 people, every person shakes hands with every other person exactly once. How many handshakes occur?",
      "435", "integer",
      "Use combinations: C(30,2) = 30 x 29 / 2 = 435.",
      "Each of the 30 people makes 29 handshakes: 30 x 29 = 870 total.",
      "870"),

    P("crt_024", "cognitive_reflection", "easy", "cognitive_ease",
      "A bottle and its cork together cost $1.10. The bottle costs $1.00 more than the cork. How much does the cork cost in cents?",
      "5", "integer",
      "Let c = cork price in cents. c + (c+100) = 110, so 2c = 10, c = 5.",
      "Total = $1.10. The bottle costs $1.00 (the round number), so the cork = 10 cents.",
      "10"),

    P("crt_025", "cognitive_reflection", "medium", "wrong_operation",
      "If you wrote all integers from 1 to 1000, how many times would you write the digit 1?",
      "301", "integer",
      "Units place: 100 ones (1, 11, 21, ..., 991). Tens place: 100 (10-19, 110-119, ...). Hundreds place: 100 (100-199). Plus 1000 has one 1. Total = 301.",
      "1000 numbers, roughly a 1/10 chance of a 1 per digit, about 3 digits per number: 1000 x 3 x 0.1 = 300.",
      "300"),

    P("crt_026", "cognitive_reflection", "medium", "cognitive_ease",
      "A clock shows 3:15. What is the angle between the hour and minute hands in degrees?",
      "7.5", "fraction",
      "Minute hand: at 90 degrees (pointing at 3). Hour hand: at 3:00 it is at 90 degrees, but by 3:15 it has moved 15/60 x 30 = 7.5 degrees further, so 97.5 degrees. Difference = 7.5 degrees.",
      "At 3:15 the minute hand and hour hand both point at the 3, so the angle is 0 degrees.",
      "0", tol=0.5),

    P("crt_027", "cognitive_reflection", "medium", "off_by_one",
      "A staircase has 10 steps. A person starts at the bottom (ground level) and climbs to the top. How many steps do they climb?",
      "10", "integer",
      "They climb steps 1 through 10: that is 10 steps.",
      "They start at ground level (step 0) and end at step 10, so they cross 9 transitions = 9 steps.",
      "9"),

    P("crt_028", "cognitive_reflection", "medium", "wrong_formula",
      "A 100-metre rope is cut into pieces each 10% of the total rope length. How many pieces are there?",
      "10", "integer",
      "10% of 100 m = 10 m per piece. 100 m / 10 m = 10 pieces.",
      "Each cut creates one extra piece, and 10 cuts of 10 m each are made, giving 11 pieces.",
      "11"),

    P("crt_029", "cognitive_reflection", "easy", "semantic_misdirection",
      "How much dirt is in a hole that is 1 metre wide, 1 metre long, and 1 metre deep?",
      "0", "integer",
      "A hole is defined as an absence of material; it contains no dirt.",
      "Volume = 1 x 1 x 1 = 1 cubic metre of dirt.",
      "1"),

    P("crt_030", "cognitive_reflection", "medium", "wrong_operation",
      "An electric train is heading east at 100 km/h. The wind is blowing west at 20 km/h. In which direction does the train's smoke blow?",
      "no smoke", "text",
      "Electric trains produce no smoke; there is nothing to blow anywhere.",
      "The wind blows westward at 20 km/h, so the smoke drifts west.",
      "west", kw=["no smoke", "electric", "none", "doesn't", "no"]),

    # ------------------------------------------------------------------ #
    # probability (25)                                                    #
    # ------------------------------------------------------------------ #
    P("prob_001", "probability", "medium", "independence_assumption",
      "A bag has 3 red balls and 5 blue balls. You draw 2 without replacement. What is the probability both are red? Give as a fraction.",
      "3/28", "fraction",
      "P = (3/8) x (2/7) = 6/56 = 3/28.",
      "P(red) = 3/8 each time. Draws are independent, so multiply: (3/8) x (3/8) = 9/64.",
      "9/64"),

    P("prob_002", "probability", "hard", "base_rate_neglect",
      "A disease affects 1% of people. A test is 99% accurate for both sick and healthy. You test positive. What is the probability you have the disease? Give as a percentage, nearest whole number.",
      "50", "integer",
      "Bayes: P(sick|+) = 0.99x0.01 / (0.99x0.01 + 0.01x0.99) = 0.0099/0.0198 = 50%.",
      "The test is 99% accurate. A positive result means a 99% chance you have the disease.",
      "99", tol=5),

    P("prob_003", "probability", "easy", "gamblers_fallacy",
      "You flip a fair coin 10 times and get heads every time. What is the probability of heads on the 11th flip? Give as a fraction.",
      "1/2", "fraction",
      "Each coin flip is independent. Past outcomes do not change future probabilities.",
      "After 10 heads in a row the coin is 'due' for tails. The probability of heads is now much less than 1/2.",
      "1/1024"),

    P("prob_004", "probability", "hard", "base_rate_neglect",
      "Three doors hide 1 car and 2 goats. You pick door 1. The host opens door 3 revealing a goat. Should you switch to door 2? State yes/no and the probability of winning if you switch.",
      "yes, 2/3", "text",
      "Your initial 1/3 probability stays with door 1. The remaining 2/3 probability concentrates on door 2 after the host reveals a goat.",
      "Two doors remain. Each is equally likely to hide the car, so it doesn't matter if you switch: probability = 1/2.",
      "no, 1/2", kw=["yes", "2/3", "switch"]),

    P("prob_005", "probability", "medium", "independence_assumption",
      "A family has 2 children. At least one is a girl. What is the probability both are girls? Give as a fraction.",
      "1/3", "fraction",
      "Possible pairs excluding BB: BG, GB, GG. Only 1 of 3 has two girls.",
      "Given one is a girl, the other is independently a boy or girl with equal probability. So P(both girls) = 1/2.",
      "1/2"),

    P("prob_006", "probability", "medium", "wrong_formula",
      "You roll two fair six-sided dice. What is the probability of a sum of 7? Give as a fraction.",
      "1/6", "fraction",
      "Ways to get 7: (1,6),(2,5),(3,4),(4,3),(5,2),(6,1) = 6 ways out of 36. P = 6/36 = 1/6.",
      "P(1 on first die) = 1/6. P(6 on second die) = 1/6. Multiply: 1/6 x 1/6 = 1/36.",
      "1/36"),

    P("prob_007", "probability", "medium", "conjunction_fallacy",
      "Linda is 31, single, outspoken, and cares deeply about social justice. She studied philosophy. Which is more probable?\nA) Linda is a bank teller\nB) Linda is a bank teller active in the feminist movement\nAnswer with just A or B.",
      "A", "choice",
      "P(A and B) <= P(A) always. A specific conjunction can never be more probable than either component alone.",
      "The description perfectly fits a feminist activist. Option B is far more representative of Linda, making it more probable.",
      "B"),

    P("prob_008", "probability", "medium", "wrong_formula",
      "A drawer has 6 black socks and 4 white socks. How many must you pull out to guarantee a matching pair?",
      "3", "integer",
      "Worst case: the first two draws are different colours. The third must match one of them.",
      "There are 2 colours, so pulling 7 guarantees you have at least one of each type, giving a pair.",
      "7"),

    P("prob_009", "probability", "hard", "base_rate_neglect",
      "A test for a rare disease has a 5% false-positive rate and a 0% false-negative rate. The disease affects 2% of people. A patient tests positive. What is the approximate probability they have the disease? Give as a percentage, nearest whole number.",
      "29", "integer",
      "P(sick|+) = 1.0x0.02 / (1.0x0.02 + 0.05x0.98) = 0.02 / 0.069 ~ 29%.",
      "The false-positive rate is 5%, meaning the test is 95% accurate. A positive result means a 95% chance of disease.",
      "95", tol=3),

    P("prob_010", "probability", "medium", "wrong_formula",
      "You draw one card from a standard 52-card deck. What is the probability it is a king OR a heart? Give as a fraction.",
      "4/13", "fraction",
      "Inclusion-exclusion: 4/52 + 13/52 - 1/52 = 16/52 = 4/13.",
      "Add the probabilities directly: 4/52 + 13/52 = 17/52.",
      "17/52"),

    P("prob_011", "probability", "medium", "wrong_formula",
      "You flip 3 fair coins. What is the probability of exactly 2 heads? Give as a fraction.",
      "3/8", "fraction",
      "C(3,2) = 3 ways to choose which 2 coins show heads. P = 3 x (1/2)^3 = 3/8.",
      "P(2 heads in a row) = (1/2)^2 = 1/4. The third coin is irrelevant, so the answer is 1/4.",
      "1/4"),

    P("prob_012", "probability", "easy", "wrong_formula",
      "A bag has 5 red and 3 green marbles. You pick one, replace it, then pick again. What is the probability both are red? Give as a fraction.",
      "25/64", "fraction",
      "With replacement the draws are independent. P = (5/8) x (5/8) = 25/64.",
      "After picking a red marble there are 4 red and 3 green left. P = (5/8) x (4/7) = 20/56 = 5/14.",
      "5/14"),

    P("prob_013", "probability", "medium", "wrong_formula",
      "You have a 1/6 chance of winning each game. You play 6 games. What is the probability of winning AT LEAST ONCE? Give as a percentage, nearest whole number.",
      "67", "integer",
      "P(at least one win) = 1 - P(no wins) = 1 - (5/6)^6 ~ 1 - 0.335 = 0.665 ~ 67%.",
      "Expected wins = 6 x 1/6 = 1. An expected value of 1 means winning at least once is certain: 100%.",
      "100", tol=2),

    P("prob_014", "probability", "medium", "wrong_formula",
      "Two dice are rolled. What is the probability of rolling at least one 6? Give as a fraction.",
      "11/36", "fraction",
      "P(at least one 6) = 1 - P(no 6) = 1 - (5/6)^2 = 1 - 25/36 = 11/36.",
      "P(6 on first die) = 1/6. P(6 on second die) = 1/6. Add them: 1/6 + 1/6 = 2/6 = 1/3.",
      "1/3"),

    P("prob_015", "probability", "hard", "base_rate_neglect",
      "A city has 85% yellow taxis and 15% blue taxis. A witness correctly identifies taxi colour 80% of the time. They say the taxi involved in an accident was blue. What is the probability it was actually blue? Give as a percentage, nearest whole number.",
      "41", "integer",
      "Bayes: P(blue|said blue) = 0.8x0.15 / (0.8x0.15 + 0.2x0.85) = 0.12 / 0.29 ~ 41%.",
      "The witness is 80% accurate and says blue. The probability it was blue is therefore 80%.",
      "80", tol=3),

    P("prob_016", "probability", "medium", "wrong_formula",
      "You pick 2 cards from a standard 52-card deck without replacement. What is the probability both are aces? Give as a fraction.",
      "1/221", "fraction",
      "P = (4/52) x (3/51) = 12/2652 = 1/221.",
      "P(ace) = 4/52 = 1/13 each time. Since draws are independent: (1/13)^2 = 1/169.",
      "1/169"),

    P("prob_017", "probability", "medium", "gamblers_fallacy",
      "A slot machine has paid out jackpots on its last 5 pulls. Are the odds of winning on the next pull higher, lower, or the same as usual?",
      "the same", "text",
      "Each pull is an independent event. Past payouts do not alter the machine's probability.",
      "After 5 jackpots the machine is statistically 'due' not to pay out. The odds of winning are now lower.",
      "lower", kw=["same", "unchanged", "independent"]),

    P("prob_018", "probability", "medium", "wrong_formula",
      "What is the probability of rolling a total of 2 with two fair six-sided dice? Give as a fraction.",
      "1/36", "fraction",
      "Only one outcome gives a sum of 2: (1,1). Total outcomes = 36. P = 1/36.",
      "P(1 on die 1) = 1/6. P(1 on die 2) = 1/6. Since they are the same value, add: 1/6 + 1/6 = 1/3.",
      "1/3"),

    P("prob_019", "probability", "medium", "wrong_formula",
      "In a group of 23 people, is the probability that at least 2 share a birthday greater than 50% or less than 50%?",
      "greater than 50%", "text",
      "P(no shared birthday) = (365/365)x(364/365)x...x(343/365) ~ 0.493. P(at least one shared) ~ 50.7%.",
      "23 people and 365 days: 23/365 ~ 6% chance of any two sharing a birthday, much less than 50%.",
      "less than 50%", kw=["greater", "more", "above", "yes"]),

    P("prob_020", "probability", "medium", "wrong_formula",
      "A bag contains 4 red and 6 blue balls. You draw 3 balls without replacement. What is the probability all 3 are blue? Give as a fraction.",
      "1/6", "fraction",
      "P = C(6,3)/C(10,3) = 20/120 = 1/6.",
      "P(blue) = 6/10. Independent draws: (6/10)^3 = 216/1000 = 27/125.",
      "27/125"),

    P("prob_021", "probability", "hard", "independence_assumption",
      "A fair coin is flipped until heads appears. What is the probability heads first appears on the 3rd flip? Give as a fraction.",
      "1/8", "fraction",
      "P(tails, tails, heads) = (1/2)^3 = 1/8.",
      "The probability of getting at least one head somewhere in 3 flips is 1 - (1/2)^3 = 7/8, so the answer is 7/8.",
      "7/8"),

    P("prob_022", "probability", "medium", "wrong_formula",
      "What is the probability that a card drawn from a standard deck is either a face card (J, Q, K) or a spade? Give as a fraction.",
      "11/26", "fraction",
      "P(face) = 12/52. P(spade) = 13/52. P(face and spade) = 3/52. P(face or spade) = (12+13-3)/52 = 22/52 = 11/26.",
      "Simply add: 12/52 + 13/52 = 25/52.",
      "25/52"),

    P("prob_023", "probability", "medium", "wrong_formula",
      "You roll a die 4 times. What is the probability of getting at least one 6? Give as a percentage, nearest whole number.",
      "52", "integer",
      "P(at least one 6) = 1 - (5/6)^4 ~ 1 - 0.482 = 0.518 ~ 52%.",
      "P(6 on any roll) = 1/6. Over 4 rolls: 4 x 1/6 = 4/6 ~ 67%.",
      "67", tol=2),

    P("prob_024", "probability", "medium", "wrong_formula",
      "A bag has 10 marbles: 3 red, 4 blue, 3 green. You draw 1 marble. What is the probability it is NOT blue? Give as a fraction.",
      "3/5", "fraction",
      "P(not blue) = 1 - 4/10 = 6/10 = 3/5.",
      "P(not blue) = P(red) x P(green) = (3/10) x (3/10) = 9/100.",
      "9/100"),

    P("prob_025", "probability", "easy", "wrong_formula",
      "A box has 3 red, 3 green, and 3 blue pens. What is the minimum number you must draw without looking to guarantee 2 pens of the same colour?",
      "4", "integer",
      "Worst case: the first 3 draws each give a different colour. The 4th draw must match one of the first three.",
      "There are 3 colours and 9 pens. Since 9/3 = 3, drawing 3 pens is enough.",
      "3"),

    # ------------------------------------------------------------------ #
    # rate_problems (20)                                                  #
    # ------------------------------------------------------------------ #
    P("rate_001", "rate_problems", "medium", "arithmetic_mean_error",
      "Pipe A fills a tank in 3 hours. Pipe B fills it in 6 hours. With both pipes open, how many hours to fill the tank?",
      "2", "integer",
      "Combined rate = 1/3 + 1/6 = 2/6 + 1/6 = 3/6 = 1/2 tank/hour. Time = 2 hours.",
      "Average the two times: (3 + 6) / 2 = 4.5 hours.",
      "4.5"),

    P("rate_002", "rate_problems", "medium", "arithmetic_mean_error",
      "Alice drives to work at 40 km/h and returns the same route at 60 km/h. What is her average speed for the round trip in km/h?",
      "48", "integer",
      "Use the harmonic mean: 2ab/(a+b) = 2x40x60/100 = 48 km/h.",
      "Average the two speeds: (40 + 60) / 2 = 50 km/h.",
      "50"),

    P("rate_003", "rate_problems", "medium", "wrong_formula",
      "Pipe A fills a pool in 4 hours. A drain empties it in 12 hours. With both running simultaneously, how many hours to fill an empty pool?",
      "6", "integer",
      "Net fill rate = 1/4 - 1/12 = 3/12 - 1/12 = 2/12 = 1/6 of the pool per hour. Time = 6 hours.",
      "Average the two times: (4 + 12) / 2 = 8 hours.",
      "8"),

    P("rate_004", "rate_problems", "medium", "wrong_formula",
      "A train 150 metres long passes a 300-metre platform in 30 seconds. What is the train's speed in metres per second?",
      "15", "integer",
      "Total distance = train length + platform length = 150 + 300 = 450 m. Speed = 450 / 30 = 15 m/s.",
      "The platform is 300 m and the crossing takes 30 seconds. Speed = 300 / 30 = 10 m/s.",
      "10"),

    P("rate_005", "rate_problems", "medium", "arithmetic_mean_error",
      "Worker A finishes a job in 5 days. Worker B finishes it in 10 days. Worker C finishes it in 15 days. Working together, how many days to finish? Round to 1 decimal place.",
      "2.7", "fraction",
      "Combined rate = 1/5 + 1/10 + 1/15 = 6/30 + 3/30 + 2/30 = 11/30. Days = 30/11 ~ 2.7.",
      "Average the three times: (5 + 10 + 15) / 3 = 10 days.",
      "10", tol=0.2),

    P("rate_006", "rate_problems", "medium", "wrong_formula",
      "If 4 people can dig 4 holes in 4 days, how many days for 8 people to dig 8 holes?",
      "4", "integer",
      "Rate: 1 person digs 1 hole in 4 days. Scaling to 8 people and 8 holes, each person still digs 1 hole, so the same 4 days.",
      "Doubling the workforce doubles the output. 8 people dig 8 holes in 2 days.",
      "2"),

    P("rate_007", "rate_problems", "medium", "arithmetic_mean_error",
      "A car travels the first half of a journey at 60 km/h and the second half at 40 km/h. What is the average speed for the whole journey in km/h?",
      "48", "integer",
      "Equal distances, so use the harmonic mean: 2x60x40/(60+40) = 4800/100 = 48 km/h.",
      "Arithmetic mean: (60 + 40) / 2 = 50 km/h.",
      "50"),

    P("rate_008", "rate_problems", "medium", "wrong_formula",
      "It takes a boy 40 minutes to mow a lawn and his father 20 minutes. Together, how many minutes does it take? Round to 1 decimal place.",
      "13.3", "fraction",
      "Combined rate = 1/40 + 1/20 = 3/40. Time = 40/3 ~ 13.3 minutes.",
      "Average their times: (40 + 20) / 2 = 30 minutes.",
      "30", tol=0.5),

    P("rate_009", "rate_problems", "hard", "wrong_formula",
      "A boat goes 30 km upstream in 3 hours and returns 30 km downstream in 2 hours. What is the speed of the current in km/h?",
      "2.5", "fraction",
      "Upstream speed = 10 km/h; downstream = 15 km/h. Current = (15 - 10)/2 = 2.5 km/h.",
      "The time difference is 1 hour over 30 km. Current speed = 30/1 = 30 km/h.",
      "30", tol=0.1),

    P("rate_010", "rate_problems", "medium", "wrong_formula",
      "A car travels 120 km at 60 km/h, then 120 km at 40 km/h. What is the total travel time in hours?",
      "5", "integer",
      "Time for leg 1 = 120/60 = 2 hours. Time for leg 2 = 120/40 = 3 hours. Total = 5 hours.",
      "Total distance = 240 km. Average speed = (60+40)/2 = 50 km/h. Time = 240/50 = 4.8 hours.",
      "4.8"),

    P("rate_011", "rate_problems", "medium", "wrong_formula",
      "Three pipes fill a tank. Pipe X fills it in 6 hours, Pipe Y in 8 hours, Pipe Z drains it in 12 hours. With all three open, how many hours to fill an empty tank?",
      "4.8", "fraction",
      "Net rate = 1/6 + 1/8 - 1/12 = 4/24 + 3/24 - 2/24 = 5/24. Time = 24/5 = 4.8 hours.",
      "The net rate is about 1/5 of the tank per hour, so it takes about 5 hours.",
      "5", tol=0.1),

    P("rate_012", "rate_problems", "medium", "off_by_one",
      "A leaking tank loses 5% of its water every hour. It starts at 100 litres. How many litres remain after 2 hours? Round to 2 decimal places.",
      "90.25", "fraction",
      "After hour 1: 100 x 0.95 = 95. After hour 2: 95 x 0.95 = 90.25.",
      "5% loss per hour for 2 hours = 10% total. 100 x 0.90 = 90 litres.",
      "90", tol=0.05),

    P("rate_013", "rate_problems", "medium", "wrong_formula",
      "A tap fills 1/4 of a tank in 2 hours. At the same rate, how many hours to fill the whole tank?",
      "8", "integer",
      "If 1/4 takes 2 hours, the full tank takes 4 x 2 = 8 hours.",
      "The first quarter takes 2 hours, and the remaining three quarters take 3 x 2 = 6 hours, so it fills in about 6 hours.",
      "6"),

    P("rate_014", "rate_problems", "hard", "wrong_formula",
      "Two cars start at the same point, driving in opposite directions. Car A at 50 km/h, Car B at 70 km/h. After how many hours are they 360 km apart?",
      "3", "integer",
      "They move apart at 50 + 70 = 120 km/h. Time = 360 / 120 = 3 hours.",
      "Average their speeds: (50 + 70)/2 = 60 km/h. Time = 360/60 = 6 hours.",
      "6"),

    P("rate_015", "rate_problems", "medium", "arithmetic_mean_error",
      "A swimmer swims 200 m at 2 m/s and then 200 m at 4 m/s. What is the average speed for the whole swim in m/s?",
      "8/3", "fraction",
      "Harmonic mean: 2x2x4/(2+4) = 16/6 = 8/3 ~ 2.67 m/s.",
      "Arithmetic mean: (2 + 4)/2 = 3 m/s.",
      "3", tol=0.05),

    P("rate_016", "rate_problems", "medium", "wrong_formula",
      "A train 300 m long takes 30 seconds to pass a stationary point. How long to fully pass a 600-metre tunnel in seconds?",
      "90", "integer",
      "Speed = 300/30 = 10 m/s. Distance to pass the tunnel = 300 + 600 = 900 m. Time = 900/10 = 90 seconds.",
      "The tunnel is 600 m. Time = 600 / 10 = 60 seconds.",
      "60"),

    P("rate_017", "rate_problems", "medium", "wrong_formula",
      "A car uses 8 litres of fuel per 100 km. How many litres does it use for a 350-km trip?",
      "28", "integer",
      "Fuel = (8/100) x 350 = 28 litres.",
      "Divide the distance by the consumption rate: 350 / 8 = 43.75 litres.",
      "43.75"),

    P("rate_018", "rate_problems", "medium", "wrong_formula",
      "Worker A alone takes 12 days to complete a project. A and B together take 8 days. How many days would B take alone?",
      "24", "integer",
      "B's rate = 1/8 - 1/12 = 3/24 - 2/24 = 1/24 per day. B alone takes 24 days.",
      "B makes up the difference in time: 12 - 8 = 4 fewer days, so B alone takes 4 days.",
      "4"),

    P("rate_019", "rate_problems", "hard", "wrong_formula",
      "A car and a truck start 450 km apart heading toward each other. The car goes 80 km/h, the truck 70 km/h. How far from the car's starting point do they meet? Round to 1 decimal place.",
      "240.0", "fraction",
      "Time to meet = 450/150 = 3 hours. Distance covered by the car = 80 x 3 = 240 km.",
      "They meet in the middle, so the meeting point is 450/2 = 225 km from the car.",
      "225", tol=1.0),

    P("rate_020", "rate_problems", "medium", "wrong_formula",
      "A tap fills a bucket in 6 minutes. A second tap fills it in 4 minutes. If the bucket is half full and both taps are opened, how many minutes to fill the rest?",
      "1.2", "fraction",
      "Combined rate = 1/6 + 1/4 = 2/12 + 3/12 = 5/12 bucket/min. Time to fill the remaining half = 0.5 / (5/12) = 6/10 = 1.2 minutes.",
      "Average time = (6 + 4)/2 = 5 minutes for a full bucket, so half a bucket = 2.5 minutes.",
      "2.5", tol=0.1),

    # ------------------------------------------------------------------ #
    # logic (25)                                                          #
    # ------------------------------------------------------------------ #
    P("logic_001", "logic", "medium", "invalid_syllogism",
      "All cats are mammals. All mammals are animals. Is it definitely true that some animals are cats? Answer yes or no.",
      "yes", "text",
      "All cats are animals (transitive). Cats exist, therefore some animals are indeed cats.",
      "You cannot reverse 'all' statements. The premises say nothing about which animals are cats.",
      "no", kw=["yes"]),

    P("logic_002", "logic", "medium", "invalid_syllogism",
      "All doctors are scientists. Some scientists are athletes. Which must be true?\nA) Some doctors are athletes\nB) Some athletes are scientists\nC) All scientists are doctors\nD) No doctors are athletes\nAnswer with just the letter.",
      "B", "choice",
      "'Some scientists are athletes' is equivalent to 'some athletes are scientists' (B). No other option is guaranteed.",
      "All doctors are scientists and some scientists are athletes, so by the chain some doctors must also be athletes. Choose A.",
      "A"),

    P("logic_003", "logic", "medium", "false_dilemma",
      "If it is raining, the ground is wet. The ground is wet. Is it definitely raining? Answer yes or no.",
      "no", "text",
      "This is the fallacy of affirming the consequent. The ground could be wet for many other reasons (a hose, melted snow, etc.).",
      "The first statement establishes a direct cause-and-effect link. The effect is present, so the cause must be too.",
      "yes", kw=["no", "not", "cannot", "necessarily"]),

    P("logic_004", "logic", "medium", "invalid_syllogism",
      "All roses are flowers. Some flowers fade quickly. Does it follow that some roses fade quickly? Answer yes or no.",
      "no", "text",
      "The flowers that fade quickly might all be non-roses. 'Some flowers' does not specify which subset.",
      "Since all roses are flowers, and some flowers fade quickly, roses are within the flower set and must share that property.",
      "yes", kw=["no", "not", "necessarily"]),

    P("logic_005", "logic", "easy", "tracking_error",
      "Tom is taller than Sam. Sam is taller than Alex. Alex is taller than Ben. Who is the second tallest?",
      "Sam", "text",
      "Order: Tom > Sam > Alex > Ben. The second tallest is Sam.",
      "Tom is the tallest. Tom is directly compared only to Sam, so the next named person, Alex, is second tallest.",
      "Alex", kw=["sam"]),

    P("logic_006", "logic", "medium", "tracking_error",
      "A is the father of B. B is the sister of C. C is the mother of D. How is A related to D?",
      "grandfather", "text",
      "A is B's father. B and C are siblings, so A is also C's father. C is D's mother, so A is D's grandfather.",
      "A is B's father. B is C's sister, so A is C's brother. A is therefore D's uncle.",
      "uncle", kw=["grandfather", "grandpa"]),

    P("logic_007", "logic", "easy", "false_equivalence",
      "All Bloops are Razzles. All Razzles are Lazzles. Are all Bloops definitely Lazzles? Answer yes or no.",
      "yes", "text",
      "Transitive syllogism: A->B and B->C implies A->C. All Bloops are Lazzles.",
      "The connection between Bloops and Lazzles is only indirect. Without a direct statement we cannot be certain.",
      "no", kw=["yes"]),

    P("logic_008", "logic", "medium", "invalid_syllogism",
      "No fish is a mammal. All whales are mammals. Which must follow?\nA) No whale is a fish\nB) Some fish are whales\nC) All fish are mammals\nD) Some whales are not mammals\nAnswer with just the letter.",
      "A", "choice",
      "Whales are mammals. No fish is a mammal. Therefore no whale can be a fish (A).",
      "The premises don't directly address whether whales could also be fish. The safest conclusion is D: some whales may not be fully mammals.",
      "D"),

    P("logic_009", "logic", "medium", "false_dilemma",
      "'If you study hard, you will pass the exam.' You did not pass. Did you definitely not study hard? Answer yes or no.",
      "yes", "text",
      "Modus tollens: P->Q and not-Q implies not-P. The conditional was stated as a guarantee.",
      "Many factors affect exam performance. You might have studied hard but fallen ill or faced unexpected questions, so no.",
      "no", kw=["yes", "modus tollens", "contrapositive"]),

    P("logic_010", "logic", "medium", "invalid_syllogism",
      "No reptile is warm-blooded. All snakes are reptiles. Is it true that no snake is warm-blooded? Answer yes or no.",
      "yes", "text",
      "All snakes are reptiles (given). No reptile is warm-blooded (given). Therefore no snake is warm-blooded: deductively valid.",
      "'No reptile is warm-blooded' is a generalisation with possible exceptions. Snakes are a specific group and may differ.",
      "no", kw=["yes"]),

    P("logic_011", "logic", "medium", "false_equivalence",
      "The suspect was at the crime scene OR the suspect has an alibi. The suspect was NOT at the crime scene. Therefore the suspect has an alibi. Is this reasoning valid? Answer yes or no.",
      "yes", "text",
      "Disjunctive syllogism: (P or Q), not-P, therefore Q. This is a valid form of reasoning.",
      "'Either...or' can mean exclusive or, in which case both could be false, so the reasoning is not necessarily valid.",
      "no", kw=["yes", "valid", "disjunctive"]),

    P("logic_012", "logic", "medium", "invalid_syllogism",
      "Some students are athletes. Some athletes are tall. Does it follow that some students are tall? Answer yes or no.",
      "no", "text",
      "The student-athletes and the tall athletes might be entirely non-overlapping subsets. 'Some A are B' plus 'some B are C' does not force 'some A are C'.",
      "There is a chain: students -> athletes -> tall. By transitivity, some students must be tall.",
      "yes", kw=["no", "not necessarily"]),

    P("logic_013", "logic", "medium", "invalid_syllogism",
      "If all A are B, and no B are C, what can we conclude about A and C?\nA) All A are C\nB) No A are C\nC) Some A are C\nD) Nothing can be concluded\nAnswer with just the letter.",
      "B", "choice",
      "A is a subset of B, and B and C are disjoint, so A and C are disjoint. No A are C.",
      "The two premises connect A->B and B->not-C separately. Without a direct A-C link we cannot conclude anything: choose D.",
      "D"),

    P("logic_014", "logic", "medium", "false_dilemma",
      "A shape has 4 sides. Is it definitely a square? Answer yes or no.",
      "no", "text",
      "Many shapes have 4 sides: rectangles, rhombuses, trapezoids, etc. A square also requires equal sides and right angles.",
      "A square is a four-sided shape, so any shape with 4 sides satisfies the definition of a square.",
      "yes", kw=["no", "not necessarily", "rectangle", "rhombus"]),

    P("logic_015", "logic", "hard", "tracking_error",
      "Five people stand in a line, positions 1 (far left) to 5 (far right): Alice, Bob, Carol, Dan, Eve. Dan is at the far-left end. Alice is not at either end. Bob is immediately to the left of Alice. Eve is immediately to the right of Alice. Bob is somewhere to the left of Carol. Who is at the far-right end?",
      "Carol", "text",
      "Dan is at position 1. Bob-Alice-Eve are three consecutive people with Alice not at an end, forcing Alice to position 3 (Bob at 2, Eve at 4). Carol takes position 5, and Bob (2) is left of Carol (5). The far-right person is Carol.",
      "Bob must be immediately left of Alice and also left of Carol, so Bob is pushed to one end; with Dan on the left, Bob lands on the far-right end.",
      "Bob", kw=["carol"]),

    P("logic_016", "logic", "medium", "anchoring",
      "You have 12 balls, 11 identical in weight and 1 heavier. With a balance scale, what is the minimum number of weighings GUARANTEED to find the heavier ball?",
      "3", "integer",
      "Each weighing has 3 outcomes. 3^3 = 27 >= 12 possible cases, while 3^2 = 9 < 12. Minimum = 3.",
      "Split 12 into two groups of 6 and weigh; that halves the search. Halve again with 3 vs 3, then read off the result, so just 2 weighings suffice.",
      "2"),

    P("logic_017", "logic", "medium", "anchoring",
      "A clock shows 6:00. The minute hand points up. Where does the hour hand point? Answer as a clock position or degrees from 12.",
      "straight down (180 degrees from 12)", "text",
      "At 6:00 the hour hand points directly at the 6, which is straight down: 180 degrees from the 12.",
      "At 6:00 both hands point at the 6, so both are straight down and the angle between them is 0 degrees.",
      "0 degrees", kw=["180", "straight down", "6", "down"]),

    P("logic_018", "logic", "hard", "tracking_error",
      "Ann is older than Ben. Ben is older than Carol. Carol is older than Dan. Dan is older than Eve. If Eve is 10 and Ann is 14, which of the following is possible?\nA) Ben is 15  B) Carol is 12  C) Dan is 9\nAnswer with just the letter.",
      "B", "choice",
      "The ages strictly decrease: Ann=14 > Ben > Carol > Dan > Eve=10. The only consistent integers are 14, 13, 12, 11, 10, so Carol = 12 (B).",
      "Ben must be older than Carol, Dan, and Eve. Since 15 is comfortably older than all of them, Ben = 15 is possible: choose A.",
      "A"),

    P("logic_019", "logic", "medium", "false_dilemma",
      "Statement: 'Only students who pass the exam receive a certificate.' A person has a certificate. Did they definitely pass the exam? Answer yes or no.",
      "yes", "text",
      "'Only students who pass receive a certificate' means passing is necessary for a certificate. Certificate implies passed.",
      "The statement doesn't say everyone who passes gets a certificate. The person might have received it some other way.",
      "no", kw=["yes"]),

    P("logic_020", "logic", "medium", "invalid_syllogism",
      "Premise: All P are Q. Conclusion: All Q are P. Is this conclusion valid? Answer yes or no.",
      "no", "text",
      "This is the fallacy of converting a universal affirmative. 'All dogs are animals' does not mean 'all animals are dogs'.",
      "The two statements are logically equivalent: if all P are Q, then naturally all Q must be P.",
      "yes", kw=["no", "invalid", "fallacy", "converse"]),

    P("logic_021", "logic", "medium", "anchoring",
      "A drawer contains 8 red socks and 6 blue socks, all mixed up in the dark. What is the minimum number of socks you must pull out to GUARANTEE a matching pair?",
      "3", "integer",
      "Two colours only. Worst case: the first two socks are different colours; the third must match one of them. So 3.",
      "There are 14 socks. To be sure of a pair you must rule out drawing every sock of one colour first: 8 + 1 = 9.",
      "9"),

    P("logic_022", "logic", "easy", "semantic_misdirection",
      "A woman shoots her husband. Moments later they go out to dinner together. How?",
      "she is a photographer", "text",
      "She photographed her husband; 'shooting' commonly means taking a photograph.",
      "The man survived because she missed the shot; she is simply a poor marksman.",
      "she missed", kw=["photographer", "photo", "camera", "picture", "shot"]),

    P("logic_023", "logic", "medium", "wrong_operation",
      "Three logicians walk into a bar. The bartender asks 'Do all of you want a drink?' The first logician says 'I don't know.' The second says 'I don't know.' The third says 'Yes.' Why does the third know?",
      "each saying I don't know means they personally want a drink", "text",
      "If any logician did NOT want a drink, they could answer 'No' to the 'all' question. Saying 'I don't know' reveals they want one but are unsure about the others. By the third logician's turn, the first two have each revealed they want a drink, so the third, also wanting one, can answer 'Yes.'",
      "The third logician is simply the most confident and decisive of the three, so they commit to an answer.",
      "third is decisive", kw=["don't know", "want", "others", "unsure"]),

    P("logic_024", "logic", "hard", "false_dilemma",
      "In a village, the barber shaves everyone who does NOT shave themselves, and no one else. Who shaves the barber?",
      "the barber paradox has no consistent answer", "text",
      "This is Russell's Paradox. If the barber shaves himself, he should not (he only shaves non-self-shavers); if he does not, then he should. No consistent answer exists.",
      "The barber is shaved by the village elder or another person who sits outside the rule.",
      "the village elder", kw=["paradox", "no", "inconsistent", "impossible", "russell"]),

    P("logic_025", "logic", "medium", "wrong_operation",
      "You have a 3-litre jug and a 5-litre jug. What is the minimum number of pouring steps to measure exactly 4 litres?",
      "6", "integer",
      "One valid sequence: (1) fill the 5 L; (2) pour into the 3 L, leaving 2 L in the 5 L; (3) empty the 3 L; (4) pour the 2 L into the 3 L; (5) fill the 5 L; (6) pour from the 5 L into the 3 L until full, leaving exactly 4 L. That is 6 steps.",
      "Fill the 3 L and pour it into the 5 L twice: 3 + 3 overflows, leaving 1 L in the 3 L jug. Fill the 3 L again: 1 + 3 = 4. That is 4 steps.",
      "4"),

    # ------------------------------------------------------------------ #
    # algebra (25)                                                        #
    # ------------------------------------------------------------------ #
    P("alg_001", "algebra", "easy", "wrong_operation",
      "A number is doubled then 5 is subtracted. The result equals three times the original minus 11. What is the number?",
      "6", "integer",
      "2x - 5 = 3x - 11 -> -5 + 11 = 3x - 2x -> x = 6.",
      "2x - 5 = 3x - 11. Moving terms with a sign slip gives x = -6.",
      "-6"),

    P("alg_002", "algebra", "medium", "wrong_formula",
      "The sum of three consecutive integers is 48. What is the largest?",
      "17", "integer",
      "Let the integers be n-1, n, n+1. Sum = 3n = 48 -> n = 16. Largest = 17.",
      "Divide 48 by 3 to get the middle number = 16. The largest is 16 + 2 = 18.",
      "18"),

    P("alg_003", "algebra", "medium", "wrong_operation",
      "Maria is 4 years older than twice her brother's age. Their ages sum to 25. How old is Maria?",
      "18", "integer",
      "Let b = brother's age. Maria = 2b + 4. (2b + 4) + b = 25 -> 3b = 21 -> b = 7. Maria = 18.",
      "Roughly two-thirds of their combined age belongs to Maria: about 2/3 x 25 ~ 20.",
      "20"),

    P("alg_004", "algebra", "medium", "wrong_formula",
      "Two numbers have a sum of 40 and a difference of 10. What is the larger number?",
      "25", "integer",
      "x + y = 40, x - y = 10. Add: 2x = 50, x = 25.",
      "Average = 40/2 = 20. The larger is 10 more than the average: 20 + 10 = 30.",
      "30"),

    P("alg_005", "algebra", "medium", "wrong_formula",
      "A cistern is 2/3 full. After 8 litres are removed it is 1/2 full. What is the total capacity in litres?",
      "48", "integer",
      "2/3 - 1/2 = 1/6 of the total = 8 litres -> total = 48 litres.",
      "Half full equals 16 litres (double the 8 removed), so the capacity = 16.",
      "16"),

    P("alg_006", "algebra", "medium", "wrong_operation",
      "At a party, every person shakes hands with every other person exactly once. There are 45 handshakes. How many people are at the party?",
      "10", "integer",
      "n(n-1)/2 = 45 -> n(n-1) = 90 -> n = 10 (since 10 x 9 = 90).",
      "Each person makes n-1 handshakes, so n(n-1) = 45. Trying n = 7 gives 7 x 6 = 42, so about 7 people.",
      "7"),

    P("alg_007", "algebra", "medium", "wrong_formula",
      "Alice drives from A to B at 80 km/h. She returns at 120 km/h. What was the average speed for the whole journey in km/h?",
      "96", "integer",
      "Use the harmonic mean: 2x80x120/(80+120) = 19200/200 = 96 km/h.",
      "Average the two speeds: (80 + 120)/2 = 100 km/h.",
      "100"),

    P("alg_008", "algebra", "medium", "wrong_formula",
      "The average of 5 numbers is 20. A sixth number is added and the average becomes 22. What is the sixth number?",
      "32", "integer",
      "Sum of 5 = 100. Sum of 6 = 132. Sixth number = 32.",
      "The average rose by 2 across 6 numbers, so the sixth number = 22 + 2 = 24.",
      "24"),

    P("alg_009", "algebra", "medium", "wrong_operation",
      "A rope is cut in the ratio 3:5. The longer piece is 40 cm. How long is the whole rope?",
      "64", "integer",
      "The longer piece = 5 parts. 5 parts = 40 cm -> 1 part = 8 cm. Total = 8 parts = 64 cm.",
      "Scale the longer piece by 5/3 to get the shorter one: 40 x 5/3 ~ 67, giving a total of about 107 cm.",
      "107"),

    P("alg_010", "algebra", "medium", "wrong_operation",
      "A store first marks up the cost by 25%, then offers a 20% discount. What is the net effect compared to the original price?",
      "0% change (break even)", "text",
      "After the markup: 1.25 x cost. After the discount: 0.80 x 1.25 x cost = 1.00 x cost, exactly the original price.",
      "A 25% markup minus a 20% discount = a net 5% gain.",
      "5% profit", kw=["0", "break even", "no profit", "no loss", "neither"]),

    P("alg_011", "algebra", "medium", "wrong_formula",
      "A man invested $10,000 at 5% simple interest per year. How much interest does he earn after 3 years?",
      "1500", "integer",
      "Simple interest = P x r x t = 10000 x 0.05 x 3 = $1,500.",
      "Compounding year by year: $500, then $525, then about $551, totalling about $1,576.",
      "1576"),

    P("alg_012", "algebra", "hard", "wrong_formula",
      "A boat travels 30 km upstream in 3 hrs and the same distance downstream in 2 hrs. What is the boat's speed in still water in km/h?",
      "12.5", "fraction",
      "Upstream speed = 10 km/h; downstream = 15 km/h. Boat speed = (15 + 10)/2 = 12.5 km/h.",
      "Divide total distance by total time: 60 km / (3 + 2) hours = 12 km/h.",
      "12", tol=0.5),

    P("alg_013", "algebra", "medium", "wrong_operation",
      "A sum of money doubles under simple interest in 8 years. What is the annual interest rate as a percentage?",
      "12.5", "fraction",
      "To double, the interest earned equals the principal over 8 years. Rate = 100% / 8 = 12.5% per year.",
      "Use the rule of 72: rate = 72/8 = 9% per year.",
      "9", tol=0.1),

    P("alg_014", "algebra", "medium", "wrong_formula",
      "The length of a rectangle is 3 times its width. The perimeter is 48 cm. What is the area?",
      "108", "integer",
      "2(w + 3w) = 8w = 48 -> w = 6. Length = 18. Area = 6 x 18 = 108.",
      "Using perimeter^2 / 16: 48^2 / 16 = 2304/16 = 144.",
      "144"),

    P("alg_015", "algebra", "medium", "wrong_operation",
      "A man sells two watches for $99 each, making a 10% profit on one and a 10% loss on the other. What is the overall result?",
      "loss of $2", "text",
      "Cost of the profit watch = 99/1.10 = $90. Cost of the loss watch = 99/0.90 = $110. Total cost = $200, revenue = $198. Loss = $2.",
      "Equal percentage gain and loss on the same selling price cancel out, so it breaks even.",
      "break even", kw=["loss", "2", "$2"]),

    P("alg_016", "algebra", "medium", "wrong_formula",
      "A container holds 120 litres when full. It is currently 5/8 full. How many litres are needed to fill it?",
      "45", "integer",
      "Empty fraction = 1 - 5/8 = 3/8. Litres needed = 3/8 x 120 = 45.",
      "Scale the capacity up by 8/5 and subtract: 120 x 8/5 - 120 = 192 - 120 = 72.",
      "72"),

    P("alg_017", "algebra", "medium", "wrong_formula",
      "If 3x + 7 = 22, what is 6x + 14?",
      "44", "integer",
      "6x + 14 = 2(3x + 7) = 2 x 22 = 44. No need to solve for x.",
      "Solve for x: 3x = 15, x = 5. Then 6x + 7 = 37.",
      "37"),

    P("alg_018", "algebra", "medium", "wrong_operation",
      "A town's population grows from 40,000 to 50,000. What is the percentage increase?",
      "25", "integer",
      "Percentage increase = (50000 - 40000)/40000 x 100 = 10000/40000 x 100 = 25%.",
      "The increase is 10,000. As a percentage of the NEW population: 10,000/50,000 x 100 = 20%.",
      "20"),

    P("alg_019", "algebra", "medium", "wrong_formula",
      "A shopkeeper has 120 items. He sells 40% and receives 30 more. How many items does he have now?",
      "102", "integer",
      "After selling 40%: 120 x 0.6 = 72 remain. After receiving 30: 72 + 30 = 102.",
      "He sells 40% of (120 + 30) = 40% of 150 = 60 sold, leaving 150 - 60 = 90.",
      "90"),

    P("alg_020", "algebra", "medium", "wrong_formula",
      "In a class of 30 students, 18 like maths and 15 like science. 8 like both. How many like neither?",
      "5", "integer",
      "At least one subject: 18 + 15 - 8 = 25. Neither: 30 - 25 = 5.",
      "Add the subject fans: 18 + 15 = 33. That exceeds the class size, so 33 - 30 = 3 like neither.",
      "3"),

    P("alg_021", "algebra", "medium", "wrong_operation",
      "A shopkeeper buys 50 kg of coffee at $4.50/kg and sells it all at $6/kg. What is the total profit?",
      "75", "integer",
      "Profit per kg = 6.00 - 4.50 = $1.50. Total = 1.50 x 50 = $75.",
      "Revenue = $300, cost = $225, so the profit margin = 75/300 = 25%.",
      "25"),

    P("alg_022", "algebra", "hard", "wrong_formula",
      "The product of two consecutive odd integers is 195. What is the smaller integer?",
      "13", "integer",
      "n(n + 2) = 195 -> n^2 + 2n - 195 = 0 -> n = (-2 + sqrt(784))/2 = (-2 + 28)/2 = 13. Check: 13 x 15 = 195.",
      "sqrt(195) ~ 14, and 13 x 15 = 195, so the integers are 13 and 15; the larger one, 15, is the answer.",
      "15"),

    P("alg_023", "algebra", "medium", "wrong_operation",
      "A salesman earns $2,000/month base salary plus 5% commission on sales. In January he earned $3,500 total. What were his total sales?",
      "30000", "integer",
      "Commission = 3500 - 2000 = $1,500. Sales = 1500/0.05 = $30,000.",
      "Treat total earnings as pure commission: sales = 3500/0.05 = $70,000.",
      "70000"),

    P("alg_024", "algebra", "medium", "wrong_formula",
      "A 20-litre mixture is milk and water in the ratio 3:1. How many litres of water must be added to make the ratio 3:2?",
      "5", "integer",
      "Currently milk = 15 L, water = 5 L. Target milk/water = 3/2. 15/(5 + x) = 3/2 -> 30 = 15 + 3x -> x = 5.",
      "The water fraction goes from 1/4 to 2/5. Add (2/5 - 1/4) x 20 = (8/20 - 5/20) x 20 = 3 litres.",
      "3"),

    P("alg_025", "algebra", "medium", "off_by_one",
      "You plant a tree every 10 metres along a 100-metre road, including both ends. How many trees are there?",
      "11", "integer",
      "Trees at 0, 10, 20, ..., 100 m = 11 trees total.",
      "100 metres / 10 metres per gap = 10 trees.",
      "10"),

    # ------------------------------------------------------------------ #
    # combinatorics (20)                                                  #
    # ------------------------------------------------------------------ #
    P("comb_001", "combinatorics", "medium", "permutation_vs_combination",
      "How many ways can you choose a committee of 3 people from 8?",
      "56", "integer",
      "Order doesn't matter in a committee: C(8,3) = 8!/(3! x 5!) = 56.",
      "Treat order as mattering: 8 x 7 x 6 = 336.",
      "336"),

    P("comb_002", "combinatorics", "medium", "wrong_formula",
      "How many ways can 6 people sit in a row of 6 chairs?",
      "720", "integer",
      "P(6,6) = 6! = 720.",
      "C(6,6) = 1 way, since all people are chosen.",
      "1"),

    P("comb_003", "combinatorics", "medium", "circular_permutation",
      "In how many ways can 6 people sit around a circular table? Rotations are considered identical.",
      "120", "integer",
      "Circular permutations = (n - 1)! = 5! = 120.",
      "Linear arrangements: 6! = 720 ways.",
      "720"),

    P("comb_004", "combinatorics", "medium", "wrong_formula",
      "A pizza shop has 8 toppings. How many pizzas can you make with exactly 2 toppings?",
      "28", "integer",
      "Order of toppings doesn't matter: C(8,2) = 28.",
      "First topping: 8 choices. Second: 7 choices. 8 x 7 = 56.",
      "56"),

    P("comb_005", "combinatorics", "medium", "wrong_formula",
      "How many 3-digit numbers can be formed from the digits 1, 2, 3, 4, 5 without repetition?",
      "60", "integer",
      "Order matters: P(5,3) = 5 x 4 x 3 = 60.",
      "Choose 3 from 5: C(5,3) = 10.",
      "10"),

    P("comb_006", "combinatorics", "medium", "wrong_formula",
      "How many ways can a president, vice president, and secretary be chosen from 10 club members?",
      "720", "integer",
      "Distinct roles, so order matters: P(10,3) = 10 x 9 x 8 = 720.",
      "No order needed, just choose 3: C(10,3) = 120.",
      "120"),

    P("comb_007", "combinatorics", "medium", "wrong_formula",
      "How many distinct arrangements are there of the letters in 'LEVEL'?",
      "30", "integer",
      "LEVEL has L x2, E x2, V x1. Total = 5!/(2! x 2!) = 120/4 = 30.",
      "5 letters with the two L's repeated: 5!/2! = 60.",
      "60"),

    P("comb_008", "combinatorics", "medium", "wrong_formula",
      "In how many ways can 4 boys and 3 girls sit in a row such that all girls sit together?",
      "720", "integer",
      "Treat the girls as one block: 5 units arrange in 5! = 120 ways. The girls arrange among themselves in 3! = 6 ways. Total = 120 x 6 = 720.",
      "Multiply the boys' and girls' internal arrangements only: 4! x 3! = 24 x 6 = 144.",
      "144"),

    P("comb_009", "combinatorics", "medium", "wrong_formula",
      "How many diagonals does a hexagon have?",
      "9", "integer",
      "C(6,2) - 6 = 15 - 6 = 9 diagonals.",
      "Each vertex connects to 4 other vertices by a diagonal: 6 x 4 / 2 = 12.",
      "12"),

    P("comb_010", "combinatorics", "medium", "wrong_formula",
      "A coin is flipped 5 times. How many different sequences of heads and tails are possible?",
      "32", "integer",
      "2^5 = 32.",
      "5 flips x 2 outcomes = 10 sequences.",
      "10"),

    P("comb_011", "combinatorics", "hard", "wrong_formula",
      "How many ways can the letters A, B, C, D be arranged in a row so that A always comes before B?",
      "12", "integer",
      "Total arrangements = 4! = 24. By symmetry, exactly half have A before B = 12.",
      "Fix A before B by counting positions for A (3 early slots) times the rest: 3 x 1 x 2 = 6.",
      "6"),

    P("comb_012", "combinatorics", "medium", "wrong_formula",
      "A restaurant menu has 4 starters, 6 mains, and 3 desserts. How many different 3-course meals are possible?",
      "72", "integer",
      "Multiply the choices at each course: 4 x 6 x 3 = 72.",
      "Total items = 4 + 6 + 3 = 13. Choose 3 from 13: C(13,3) = 286.",
      "286"),

    P("comb_013", "combinatorics", "hard", "wrong_formula",
      "A committee of 5 is chosen from 6 men and 4 women. How many committees have at least 2 women?",
      "186", "integer",
      "Exactly 2W: C(4,2)xC(6,3) = 6 x 20 = 120. Exactly 3W: C(4,3)xC(6,2) = 4 x 15 = 60. Exactly 4W: C(4,4)xC(6,1) = 1 x 6 = 6. Total = 186.",
      "Choose 2 women first (C(4,2) = 6), then fill the other 3 seats from the remaining 8 people (C(8,3) = 56): 6 x 56 = 336.",
      "336"),

    P("comb_014", "combinatorics", "medium", "wrong_formula",
      "How many 4-digit PIN codes can be formed if no digit repeats?",
      "5040", "integer",
      "Order matters, no repetition: P(10,4) = 10 x 9 x 8 x 7 = 5040.",
      "Choose 4 digits from 10 without regard to order: C(10,4) = 210.",
      "210"),

    P("comb_015", "combinatorics", "medium", "wrong_formula",
      "How many ways can a team of 1 man and 1 woman be chosen from 5 men and 4 women?",
      "20", "integer",
      "C(5,1) x C(4,1) = 5 x 4 = 20.",
      "Choose any 2 from the 9 people: C(9,2) = 36, then halve to 18.",
      "18"),

    P("comb_016", "combinatorics", "medium", "wrong_formula",
      "How many ways can 3 people be arranged in a row, chosen from a group of 7?",
      "210", "integer",
      "P(7,3) = 7 x 6 x 5 = 210.",
      "Order doesn't matter: C(7,3) = 35.",
      "35"),

    P("comb_017", "combinatorics", "medium", "wrong_formula",
      "How many distinct arrangements are there of the letters in 'MISSISSIPPI'?",
      "34650", "integer",
      "11 letters: M x1, I x4, S x4, P x2. Total = 11!/(1! x 4! x 4! x 2!) = 39916800/1152 = 34650.",
      "11 letters with 3 repeated letters: 11!/3! = 6,652,800.",
      "6652800"),

    P("comb_018", "combinatorics", "hard", "wrong_formula",
      "A committee of 5 is chosen from 10 people and must include person A. How many such committees are possible?",
      "126", "integer",
      "Fix A on the committee. Choose the remaining 4 from the other 9: C(9,4) = 126.",
      "Of all C(10,5) = 252 committees, A is included with probability 4/10, giving 252 x 4/10 ~ 100.",
      "100"),

    P("comb_019", "combinatorics", "medium", "wrong_formula",
      "A tennis tournament has 16 players in a single-elimination bracket. How many total matches are played?",
      "15", "integer",
      "Each match eliminates exactly one player. To eliminate 15 of the 16 players, 15 matches are needed.",
      "Round 1: 8 matches. Round 2: 4. Quarter-finals: 2. Semi-finals: 1. Final: 1. Total: 8 + 4 + 2 + 1 + 1 = 16.",
      "16"),

    P("comb_020", "combinatorics", "medium", "wrong_formula",
      "How many ways are there to distribute 3 identical balls into 3 distinct boxes?",
      "10", "integer",
      "Stars and bars: C(3 + 3 - 1, 3 - 1) = C(5,2) = 10.",
      "Each ball independently goes into one of 3 boxes: 3^3 = 27.",
      "27"),

    # ------------------------------------------------------------------ #
    # geometry (15)                                                       #
    # ------------------------------------------------------------------ #
    P("geo_001", "geometry", "easy", "wrong_formula",
      "A right triangle has legs of 5 and 12. What is the area?",
      "30", "integer",
      "Area of a right triangle = (1/2) x leg1 x leg2 = (1/2) x 5 x 12 = 30.",
      "Find the hypotenuse: sqrt(25 + 144) = 13. Area = (1/2) x 13 x 5 = 32.5.",
      "32.5"),

    P("geo_002", "geometry", "easy", "wrong_formula",
      "A circle has radius 7 cm. What is the circumference? Give your answer in terms of pi.",
      "14pi", "text",
      "Circumference = 2*pi*r = 2 x pi x 7 = 14pi.",
      "Circumference = pi*r = pi x 7 = 7pi.",
      "7pi", kw=["14pi", "14 pi"]),

    P("geo_003", "geometry", "medium", "wrong_formula",
      "A square has a diagonal of 8 cm. What is the area in cm^2?",
      "32", "integer",
      "For a square, diagonal = s*sqrt(2) -> s = 8/sqrt(2) = 4*sqrt(2). Area = s^2 = 32.",
      "Take the side equal to the diagonal: side = 8. Area = 8^2 = 64.",
      "64"),

    P("geo_004", "geometry", "medium", "wrong_formula",
      "If the radius of a circle is doubled, by what factor does the area increase?",
      "4", "integer",
      "Area = pi*r^2. Doubling r: pi*(2r)^2 = 4*pi*r^2. The area increases by a factor of 4.",
      "Doubling the radius doubles the area. Factor = 2.",
      "2"),

    P("geo_005", "geometry", "easy", "wrong_formula",
      "A cube has side length 4 cm. What is the total surface area in cm^2?",
      "96", "integer",
      "A cube has 6 faces, each of area 4^2 = 16. Total = 6 x 16 = 96.",
      "Surface area = 4 x side^2 (the 4 visible faces) = 4 x 16 = 64.",
      "64"),

    P("geo_006", "geometry", "medium", "wrong_formula",
      "The perimeter of a rectangle is 36 cm. One side is 10 cm. What is the area in cm^2?",
      "80", "integer",
      "2(10 + w) = 36 -> w = 8. Area = 10 x 8 = 80.",
      "Other side = 36 - 10 = 26. Area = 10 x 26 = 260.",
      "260"),

    P("geo_007", "geometry", "medium", "wrong_formula",
      "A cylinder has radius 3 cm and height 5 cm. What is its volume? Give in terms of pi.",
      "45pi", "text",
      "Volume = pi*r^2*h = pi x 9 x 5 = 45pi.",
      "Volume = 2*pi*r x h (circumference x height) = 2*pi x 3 x 5 = 30pi.",
      "30pi", kw=["45pi", "45 pi"]),

    P("geo_008", "geometry", "medium", "wrong_formula",
      "A cone has base radius 4 cm and height 9 cm. What is its volume? Give in terms of pi.",
      "48pi", "text",
      "Volume = (1/3)*pi*r^2*h = (1/3) x pi x 16 x 9 = 48pi.",
      "Volume = pi*r^2*h (same as a cylinder) = pi x 16 x 9 = 144pi.",
      "144pi", kw=["48pi", "48 pi"]),

    P("geo_009", "geometry", "medium", "wrong_formula",
      "The angles of a triangle are in the ratio 1:2:3. What is the largest angle in degrees?",
      "90", "integer",
      "Angles sum to 180 degrees. Largest = (3/6) x 180 = 90 degrees.",
      "Apply the ratio to a full turn: largest = (3/6) x 360 = 180 degrees.",
      "180"),

    P("geo_010", "geometry", "medium", "wrong_formula",
      "A sphere has radius 3 cm. What is its volume? Give in terms of pi.",
      "36pi", "text",
      "Volume = (4/3)*pi*r^3 = (4/3) x pi x 27 = 36pi.",
      "Volume = (4/3)*pi*r^2 = (4/3) x pi x 9 = 12pi.",
      "12pi", kw=["36pi", "36 pi"]),

    P("geo_011", "geometry", "medium", "wrong_formula",
      "Two parallel lines are cut by a transversal. One angle is 65 degrees. What is the co-interior (same-side interior) angle?",
      "115", "integer",
      "Co-interior angles are supplementary: 180 - 65 = 115 degrees.",
      "Co-interior angles between parallel lines are equal, so the angle is also 65 degrees.",
      "65"),

    P("geo_012", "geometry", "medium", "wrong_formula",
      "An equilateral triangle has a perimeter of 24 cm. What is its area in cm^2? Round to 1 decimal place.",
      "27.7", "fraction",
      "Side = 8 cm. Area = (sqrt(3)/4) x 8^2 = (sqrt(3)/4) x 64 ~ 27.7.",
      "Area = (1/2) x base x height = (1/2) x 8 x 8 = 32 (using the side as the height).",
      "32", tol=0.2),

    P("geo_013", "geometry", "medium", "wrong_formula",
      "A circle has a diameter of 10 cm. What is its area? Give in terms of pi.",
      "25pi", "text",
      "Radius = 5. Area = pi x 5^2 = 25pi.",
      "Area = pi x diameter^2 = pi x 100 = 100pi.",
      "100pi", kw=["25pi", "25 pi"]),

    P("geo_014", "geometry", "medium", "wrong_formula",
      "A rectangular room is 5 m x 4 m. Tiles are 50 cm x 50 cm. How many tiles are needed to cover the floor?",
      "80", "integer",
      "Room area = 20 m^2. Tile area = 0.25 m^2. Tiles = 20/0.25 = 80.",
      "Tiles per row = 5/0.5 = 10. Rows = 4/0.5 = 8. Add them: 10 + 8 = 18.",
      "18"),

    P("geo_015", "geometry", "medium", "wrong_formula",
      "A right triangle has one leg of 9 and a hypotenuse of 15. What is the other leg?",
      "12", "integer",
      "Pythagoras: other leg = sqrt(15^2 - 9^2) = sqrt(225 - 81) = sqrt(144) = 12.",
      "Add the squares: sqrt(15^2 + 9^2) = sqrt(225 + 81) = sqrt(306) ~ 17.5.",
      "17.5"),

    # ------------------------------------------------------------------ #
    # sequences (10)                                                      #
    # ------------------------------------------------------------------ #
    P("seq_001", "sequences", "medium", "wrong_pattern",
      "What is the next number in: 2, 6, 12, 20, 30, ?",
      "42", "integer",
      "Pattern: n(n+1). Terms: 1x2, 2x3, 3x4, 4x5, 5x6, then 6x7 = 42.",
      "Each term is roughly 1.5 times the previous (30 ~ 1.5 x 20), so the next is about 30 x 1.5 = 45.",
      "45"),

    P("seq_002", "sequences", "easy", "wrong_pattern",
      "What is the next number: 1, 1, 2, 3, 5, 8, 13, ?",
      "21", "integer",
      "Fibonacci: each term is the sum of the previous two. 8 + 13 = 21.",
      "The terms grow by roughly doubling, so the next is 13 x 2 = 26.",
      "26"),

    P("seq_003", "sequences", "easy", "wrong_pattern",
      "What is the next number: 1, 4, 9, 16, 25, ?",
      "36", "integer",
      "Perfect squares: 1^2, 2^2, 3^2, 4^2, 5^2. Next = 6^2 = 36.",
      "The differences are 3, 5, 7, 9; doubling the step to 14 gives 25 + 14 = 39.",
      "39"),

    P("seq_004", "sequences", "medium", "wrong_pattern",
      "What is the next number: 3, 6, 12, 24, 48, ?",
      "96", "integer",
      "Each term doubles. 48 x 2 = 96.",
      "Reading the pattern as 3^n, the sixth term would be 3^6 = 729.",
      "729"),

    P("seq_005", "sequences", "medium", "wrong_pattern",
      "What is the missing number: 2, 5, 10, 17, ?, 37",
      "26", "integer",
      "Pattern: n^2 + 1. 1^2+1=2, 2^2+1=5, 3^2+1=10, 4^2+1=17, 5^2+1=26, 6^2+1=37.",
      "The differences are 3, 5, 7, 9, 11; misreading the gap after 17 as 8 gives 17 + 8 = 25.",
      "25"),

    P("seq_006", "sequences", "easy", "wrong_pattern",
      "What is the next number: 1, 8, 27, 64, 125, ?",
      "216", "integer",
      "Perfect cubes: 1^3, 2^3, 3^3, 4^3, 5^3. Next = 6^3 = 216.",
      "The ratios between terms are shrinking toward roughly 1.7, so the next term ~ 125 x 1.7 ~ 220.",
      "220"),

    P("seq_007", "sequences", "medium", "wrong_pattern",
      "What is the next number: 0, 1, 3, 6, 10, 15, ?",
      "21", "integer",
      "Triangular numbers: n(n+1)/2. For n = 6: 6 x 7/2 = 21.",
      "The differences are 1, 2, 3, 4, 5; adding the last difference twice gives 15 + 5 + 2 = 22.",
      "22"),

    P("seq_008", "sequences", "easy", "wrong_pattern",
      "What comes next: 2, 3, 5, 7, 11, 13, ?",
      "17", "integer",
      "These are the prime numbers. The next prime after 13 is 17.",
      "The gaps cycle 1, 2, 2, 4, 2; assuming the next gap is 2 again gives 13 + 2 = 15.",
      "15"),

    P("seq_009", "sequences", "medium", "wrong_pattern",
      "What is the next letter in the series: A, C, E, G, I, ?",
      "K", "text",
      "Every other letter of the alphabet (skip one each time). After I (9th) comes K (11th).",
      "Skip two letters after I, past J and K, landing on L.",
      "L", kw=["K", "k"]),

    P("seq_010", "sequences", "medium", "wrong_pattern",
      "What is the next number in: 1, 2, 4, 7, 11, 16, ?",
      "22", "integer",
      "The differences are 1, 2, 3, 4, 5, increasing by 1. Next difference = 6: 16 + 6 = 22.",
      "Reading the differences as doubling (1, 2, 4, 8, 16), the next term is 16 + 16 = 32.",
      "32"),

    # ------------------------------------------------------------------ #
    # causal_reasoning (10)                                               #
    # ------------------------------------------------------------------ #
    P("causal_001", "causal_reasoning", "medium", "correlation_causation",
      "Ice cream sales and drowning deaths both rise in summer. Does eating ice cream cause drowning? Answer yes or no and explain the flaw.",
      "no", "text",
      "Both are driven by a third factor, hot weather, which increases swimming and ice cream consumption simultaneously. This is a confounding variable.",
      "The correlation is strong and consistent across multiple years, and strong repeated correlations typically indicate causation.",
      "yes", kw=["no", "correlation", "confounding", "hot", "summer", "weather"]),

    P("causal_002", "causal_reasoning", "medium", "sunk_cost",
      "You paid $50 for a non-refundable concert ticket. On the day, you feel too ill to enjoy it. Should you go anyway to 'not waste' the money? Answer yes or no.",
      "no", "text",
      "The $50 is gone regardless. The rational choice depends only on whether going is better than staying home right now; the sunk cost is irrelevant.",
      "You already invested $50, so you should attend to maximise the return on that investment.",
      "yes", kw=["no", "sunk", "cost", "irrelevant", "already", "spent"]),

    P("causal_003", "causal_reasoning", "medium", "availability_heuristic",
      "Which is more likely to kill a US resident: a shark attack or a lightning strike? Answer with the more deadly cause.",
      "lightning", "text",
      "Lightning kills roughly 40-50 people per year in the US. Shark attacks kill fewer than 1 on average annually.",
      "Shark attacks receive extensive media coverage as a coastal danger, so shark deaths must be more frequent than lightning deaths.",
      "shark attack", kw=["lightning", "struck by lightning"]),

    P("causal_004", "causal_reasoning", "hard", "wrong_formula",
      "A study shows that towns with more hospitals have higher death rates. Should we close hospitals to reduce deaths? Answer yes or no.",
      "no", "text",
      "Sick people travel to towns with hospitals; they do not cause illness. Hospitals are built where patients concentrate. Closing them would increase deaths.",
      "The data shows a direct positive correlation: more hospitals -> higher death rates. Reducing hospitals is the data-supported intervention.",
      "yes", kw=["no", "confounding", "sick", "cause", "reverse", "selection"]),

    P("causal_005", "causal_reasoning", "easy", "false_analogy",
      "A rooster crows every morning just before sunrise. Does the rooster's crowing cause the sun to rise? Answer yes or no.",
      "no", "text",
      "This is the post hoc ergo propter hoc fallacy. One event preceding another in time does not prove causation.",
      "The rooster ALWAYS crows before sunrise, and sunrise ALWAYS follows. This consistent temporal ordering is the gold standard of causal evidence.",
      "yes", kw=["no", "correlation", "cause", "coincidence"]),

    P("causal_006", "causal_reasoning", "medium", "correlation_causation",
      "Countries that consume more chocolate per capita win more Nobel Prizes per capita. Does eating chocolate cause Nobel Prize wins? Answer yes or no.",
      "no", "text",
      "Both are associated with national wealth and strong educational systems; wealth is the confounding variable.",
      "The correlation is statistically significant (r ~ 0.79). A correlation this strong across nations indicates a real causal mechanism: flavonoids in chocolate may enhance cognition.",
      "yes", kw=["no", "correlation", "wealth", "confounding", "gdp"]),

    P("causal_007", "causal_reasoning", "medium", "anchoring",
      "A drug trial: 600 of 1000 treated patients improve. Without treatment, 500 of 1000 similar patients improve naturally. Is the drug effective? Answer yes or no.",
      "yes", "text",
      "The treated group improves at 60% versus a 50% baseline. The drug raises improvement above the control, so it is effective.",
      "400 of 1000 treated patients did NOT improve. A 40% failure rate means the drug is not effective.",
      "no", kw=["yes", "effective", "more", "100", "baseline", "control"]),

    P("causal_008", "causal_reasoning", "medium", "availability_heuristic",
      "In the USA, which kills more people per year on average: tornadoes or heat waves? Answer with the more deadly cause.",
      "heat waves", "text",
      "Heat waves kill roughly 700+ people per year in the US on average; tornadoes kill roughly 70. Heat waves get far less dramatic media coverage.",
      "Tornadoes are frequent in Tornado Alley and cause dramatic, widespread destruction, so their death toll exceeds that of silent heat waves.",
      "tornadoes", kw=["heat", "heat wave", "heatwave"]),

    P("causal_009", "causal_reasoning", "hard", "anchoring",
      "A study finds students with more books at home score higher on tests. A school proposes buying every student 50 books to raise scores. Is this likely to be effective? Answer yes or no.",
      "no", "text",
      "Book ownership is a proxy for household wealth and reading culture. Giving books does not create the literacy habits or parental engagement that drive high scores.",
      "The correlation shows books -> higher scores. Increasing the cause (books) should increase the effect (scores), so the proposal directly targets the measured correlate.",
      "yes", kw=["no", "confounding", "correlation", "cause", "socioeconomic", "proxy"]),

    P("causal_010", "causal_reasoning", "medium", "wrong_formula",
      "A coin is tossed 100 times and shows heads 60 times. Is the coin definitely unfair? Answer yes or no.",
      "no", "text",
      "Random variation means even a fair coin can deviate from 50/50 in 100 flips. 60/100 is within the plausible range for a fair coin; you would need a statistical test to conclude unfairness.",
      "A fair coin is expected to land heads 50% of the time. 60% is a 10-point deviation, and any deviation above 5% proves the coin is unfair.",
      "yes", kw=["no", "not necessarily", "chance", "could be", "fair", "random"]),

    # ------------------------------------------------------------------ #
    # word_problems (5; five exact duplicates of algebra items removed)   #
    # ------------------------------------------------------------------ #
    P("word_001", "word_problems", "medium", "wrong_operation",
      "Apples cost 40 cents each and oranges cost 60 cents each. John buys 5 fruits and spends exactly $2.40. How many apples did he buy?",
      "3", "integer",
      "Let a = number of apples. 40a + 60(5 - a) = 240 -> 40a + 300 - 60a = 240 -> -20a = -60 -> a = 3.",
      "Average price = $2.40/5 = 48 cents, closer to 60 cents, so he bought mostly oranges: about 2 apples.",
      "2"),

    P("word_003", "word_problems", "medium", "wrong_formula",
      "A hotel lift travels 1 floor every 8 seconds. How long in seconds does it take from floor 1 to floor 10 with no stops?",
      "72", "integer",
      "From floor 1 to floor 10 is 9 floor-gaps. Time = 9 x 8 = 72 seconds.",
      "The lift covers 10 floors at 8 seconds each: 10 x 8 = 80 seconds.",
      "80"),

    P("word_004", "word_problems", "medium", "wrong_formula",
      "A recipe for 4 people needs 3 cups of flour. How many cups are needed for 10 people?",
      "7.5", "fraction",
      "Scale factor = 10/4 = 2.5. Cups = 3 x 2.5 = 7.5.",
      "You need 6 extra servings; adding two more full batches of 3 cups gives 3 + 3 + 3 = 9 cups.",
      "9"),

    P("word_005", "word_problems", "medium", "wrong_operation",
      "Tom drove 240 km: the first 120 km at 80 km/h, the rest at 60 km/h. How long was the journey in hours?",
      "3.5", "fraction",
      "Time = 120/80 + 120/60 = 1.5 + 2 = 3.5 hours.",
      "Average speed = (80 + 60)/2 = 70 km/h. Time = 240/70 ~ 3.43 hours.",
      "3.43"),

    P("word_007", "word_problems", "medium", "wrong_formula",
      "Fence posts are placed every 5 metres along a 100-metre fence, including both ends. How many posts are there?",
      "21", "integer",
      "Posts at 0, 5, 10, ..., 100 = 100/5 + 1 = 21.",
      "100 divided by 5 = 20 posts.",
      "20"),

    # ------------------------------------------------------------------ #
    # number_theory (10)                                                  #
    # ------------------------------------------------------------------ #
    P("num_001", "number_theory", "medium", "wrong_formula",
      "What is the sum of all integers from 1 to 100?",
      "5050", "integer",
      "Gauss's formula: n(n+1)/2 = 100 x 101/2 = 5050.",
      "Average value = (1 + 100)/2 = 50.5. Multiply by 99 terms instead of 100: 50.5 x 99 ~ 5000.",
      "5000"),

    P("num_002", "number_theory", "medium", "wrong_formula",
      "What is the LCM (least common multiple) of 12 and 18?",
      "36", "integer",
      "LCM = (12 x 18)/GCD(12,18) = 216/6 = 36.",
      "The LCM is always the product of the two numbers: 12 x 18 = 216.",
      "216"),

    P("num_003", "number_theory", "medium", "wrong_formula",
      "A number increases by 25% and then decreases by 25%. What is the net percentage change?",
      "-6.25", "fraction",
      "After +25%: 1.25x. After -25%: 0.75 x 1.25x = 0.9375x. Net change = -6.25%.",
      "A 25% increase and a 25% decrease cancel perfectly. Net change = 0%.",
      "0", tol=0.1),

    P("num_004", "number_theory", "medium", "wrong_formula",
      "What is 15% of 240?",
      "36", "integer",
      "10% of 240 = 24. 5% of 240 = 12. Total = 36.",
      "15% of 200 = 30, plus a rough 15% of 40 ~ 4, giving 34.",
      "34"),

    P("num_005", "number_theory", "medium", "wrong_formula",
      "If 20% of a number is 50, what is the number?",
      "250", "integer",
      "0.20 x n = 50 -> n = 50/0.20 = 250.",
      "20% of 50 = 10, so add it back: 50 + 10 = 60.",
      "60"),

    P("num_006", "number_theory", "easy", "wrong_operation",
      "What is the remainder when 100 is divided by 7?",
      "2", "integer",
      "7 x 14 = 98. 100 - 98 = 2.",
      "100/7 ~ 14.3. The decimal part 0.3 x 7 ~ 2.1, rounded up to 3.",
      "3"),

    P("num_007", "number_theory", "medium", "wrong_formula",
      "What is the units digit of 7^100?",
      "1", "integer",
      "Units digits of powers of 7 cycle 7, 9, 3, 1 (period 4). 100 / 4 = 25 remainder 0 -> units digit of 7^4 = 1.",
      "7^2 = 49 ends in 9, so 7^100 = (7^2)^50 ends in a 9 raised to an odd-looking power, giving units digit 9.",
      "9"),

    P("num_008", "number_theory", "medium", "wrong_formula",
      "How many prime numbers are there between 1 and 30 inclusive?",
      "10", "integer",
      "Primes: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29. Count = 10.",
      "Counting the odd primes 3, 5, 7, 11, 13, 17, 19, 23, 29 gives 9, forgetting that 2 is also prime.",
      "9"),

    P("num_009", "number_theory", "medium", "wrong_formula",
      "What is the HCF (highest common factor) of 24 and 36?",
      "12", "integer",
      "Factors of 24: 1,2,3,4,6,8,12,24. Factors of 36: 1,2,3,4,6,9,12,18,36. The highest common one is 12.",
      "HCF = (24 + 36)/2 = 30.",
      "30"),

    P("num_010", "number_theory", "medium", "wrong_formula",
      "What is 2^10?",
      "1024", "integer",
      "2^10 = 2^5 x 2^5 = 32 x 32 = 1024.",
      "2^10 = 2 x 10 = 20, treating exponentiation as multiplication.",
      "20"),
]


def main():
    out_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "problems.json"))
    # Guard against accidental duplicate ids.
    seen = set()
    for rec in PROBLEMS:
        if rec["id"] in seen:
            raise ValueError(f"duplicate id: {rec['id']}")
        seen.add(rec["id"])
        apply_unicode(rec)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(PROBLEMS, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"Wrote {len(PROBLEMS)} problems to {out_path}")


if __name__ == "__main__":
    main()
