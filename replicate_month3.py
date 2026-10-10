"""Is the month-3 coach effect more than one sample? Replay months 1-2 from the cache, then ask the mayor the exact
month-3 prompt five times without the coach's note and five times with it, bypassing the cache; Jev reads each reason.

    python replicate_month3.py      # 10 DeepSeek calls (uncached), up to 10 Jev calls
"""
import doerner_jev as d

NOTE = "your last reason mentions no delayed effect or side effect of what you did."
s, history = dict(d.START), []
for _ in range(2):
    action, reason = d.mayor(s, history, None)
    s = d.step(s, action)
    history.append(f"{s['action']}: {reason[:80]}")

cached = d._mayor
d._mayor = cached.__wrapped__  # the same call, no cache: fresh samples
for label, note in (("without the note", None), ("with the note", NOTE)):
    print(f"\n# month 3, {label}")
    for i in range(5):
        action, reason = d.mayor(s, history, note)
        a = d.jev({"town": d.words(s), "steward_reason": reason}, {"error": d.ERRORS_V2, "checks": d.CHECKS_V2, "side": d.SIDE})
        print(f"  {i + 1}. {action:16s} jev: {a['error']['choice']} {a['error']['confidence']:.2f}, side p={a['side']['noul']:.2f}"
              f"\n     {reason}")
d._mayor = cached
