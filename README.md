# Sokoban solver — make.fun entry

Push-optimal Sokoban solver for `sokoban-solver-01-run-20261121`.

## Idea

The generator builds every level as a one-wide hallway with one to four
independent rooms (3–4 boxes each), and boxes never leave their room — so
the level's push-optimal count is the sum of the rooms' optima. This
program:

1. detects the hallway (the player's row/column of floor) and splits the
   level into rooms (connected floor components once the hallway is
   removed);
2. solves every room push-optimally with A* over push states
   (boxes + player-reachable region), using a minimum-assignment
   box-to-goal heuristic and deadlock pruning (dead squares, 2×2 frozen
   blocks), with a greedy best-first fallback when the optimal pass
   exhausts its budget;
3. stitches the rooms together with plain walks through the hallway and
   replays the full move string to verify it before writing it out.

All budgets are counted in expanded states, never in wall-clock time, so
two runs write identical bytes.

## Local results (practice set + 10 freshly drawn sets)

- Practice set: **1,000,000** (all 5 instances solved in exactly P*
  pushes: 13, 16, 29, 47, 60) in ~0.4 s.
- 10 further sets drawn from the kit generator with random seeds: all
  **1,000,000** as well (55 instances total, every one at P*).
- House baseline `plain-astar` on the practice set: 200,000 (solves only
  the 3-box instance, empty answers elsewhere).

## Run

```
MAKEFUN_INPUT_DIR=<input> MAKEFUN_OUTPUT_DIR=<out> node main.ts
```

Reads `<input>/<id>/level.txt`, writes `<out>/<id>/solution.txt` (moves
`u`/`d`/`l`/`r`, one letter per step).
