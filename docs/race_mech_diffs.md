# [Race mechanics doc](https://docs.google.com/document/d/15VzW9W2tXBBTibBRbZ8IVpW6HaMX8H0RP03kq6Az7Xg) corrections

| Topic | Doc says | Actually |
|---|---|---|
| Frame | Uma update in gate order | Uma update in **entry** order |
| Skills | Checked in ID order | Checked in **skill list** order, with every uma's priority 1 skills first |
| Pacemaker | Unknown | See below |
| Speed Up | Exits once far enough ahead of the uma behind | Same, but **with no uma behind her** (last place) she stays in Speed Up for the full section |
| Pace Down | 0.945 in mid-race | 0.945 if she **entered** in mid-race; ends if she becomes pacemaker; the pacemaker never paces down |
| Oonige | Not explained | A front runner **with the oonige skill** |
| Spot struggle | One uma hasn't reached section 6 | One uma is in section 6 **or earlier** |
| Spot struggle | Ends at section 9 | The **whole group** ends once any member reaches section 9 |
| Spot struggle | Blocked if a later strategy passes the leader | Blocked if a later strategy passes the leader; **for oonige, a regular front runner counts** |
| Dueling | Gaps < 3.0 m and < 0.25 | Gaps **≤** 3.0 m and **≤** 0.25 |
| Dueling | Candidates for 2 s | Candidates for **31 frames** (2.06 s) |
| Dueling | Exit when > 5 m from every other member | Exit when > 5 m from every other member, **including ones who already left** |
| Conserve power | Gain per second, continuous | Gain in **whole seconds** only; any blocked frame resets the count |
| Conserve power | 1.5 s cooldown after the state ends | 1.5 s cooldown, **restarted on every blocked frame** |
| Conserve power | Accel from the next frame | Accel from the **release frame**, lasting Duration **+ 1 frame** |
| Last spurt | Recalculation changes speed and distance | Recalculation changes **speed only**; the start distance is kept |
| Target speed | Never below min speed | Never below **13 m/s**; min speed only at 0 HP |
| Overtake lane | Early Out ×100, late Out ×1.15 | Early **In** ×100, late **In** ×1.15 |
| Overtake lane | Box ends 3 m past the furthest target | Box ends 3 m past the **nearest** target |
| Blocking | "All" = front + one side | "All" = front + **both** sides |
| Course | Distance correction uses last frame's ratio | Distance correction uses **this** frame's ratio |

## Pacemaker
The front uma of the most forward strategy, re-picked only when the race leader enters a new section (sections 1-10). If
another strategy leads the race at two re-picks (not necessarily in a row; the re-pick at the gate doesn't count), the
role passes to the race leader or, when she is a Sashi or Oikomi, to the best placed uma of a more forward strategy
within `PaseMakerChangeForwardRunningStyleRange`. The count then starts over and she keeps the role until it reaches two
again; a count that reaches two without a change of hands is kept. Once the most forward strategy leads again, it takes
the role back on the first frame after that which is not a re-pick, and the count starts over. 
