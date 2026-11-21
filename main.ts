// Sokoban solver for make.fun: sokoban-solver-01-run-20261121.
//
// The generator builds every level as a one-wide hallway with one to four
// independent rooms (3-4 boxes each). Boxes never leave their room, so the
// level's push-optimal count is the sum of the rooms' optima. This program
// detects the hallway and the rooms geometrically, solves each room
// push-optimally with A* (minimum-assignment heuristic, deadlock pruning),
// and stitches the rooms together with plain walks through the hallway.
// All budgets are counted in expanded states, never in wall-clock time, so
// two runs write identical bytes.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Deterministic work budgets (expanded states) for one room search.
const ASTAR_BUDGET = 150000;
const GREEDY_BUDGET = 150000;

interface Dir { dr: number; dc: number; ch: string; }

const DIRS: Dir[] = [
  { dr: -1, dc: 0, ch: 'u' },
  { dr: 0, dc: -1, ch: 'l' },
  { dr: 1, dc: 0, ch: 'd' },
  { dr: 0, dc: 1, ch: 'r' },
];

interface Level {
  w: number;
  h: number;
  wall: boolean[];
  goal: boolean[];
  boxes: number[];
  player: number;
}

function parseLevel(text: string): Level {
  const rawLines = text.split('\n');
  while (rawLines.length > 0 && rawLines[rawLines.length - 1].trim() === '') rawLines.pop();
  const h = rawLines.length;
  const w = Math.max(1, ...rawLines.map((l) => l.length));
  const wall = new Array<boolean>(w * h).fill(true);
  const goal = new Array<boolean>(w * h).fill(false);
  const boxes: number[] = [];
  let player = -1;
  for (let r = 0; r < h; r++) {
    const line = rawLines[r];
    for (let c = 0; c < w; c++) {
      const ch = c < line.length ? line[c] : '#';
      const idx = r * w + c;
      if (ch === '#') continue;
      wall[idx] = false;
      if (ch === '.' || ch === '*' || ch === '+') goal[idx] = true;
      if (ch === '$' || ch === '*') boxes.push(idx);
      if (ch === '@' || ch === '+') player = idx;
    }
  }
  boxes.sort((a, b) => a - b);
  return { w, h, wall, goal, boxes, player };
}

// Neighbour of cell in direction d (0..3), or -1 when outside the board.
function stepCell(level: Level, cell: number, d: number): number {
  const r = Math.floor(cell / level.w) + DIRS[d].dr;
  const c = (cell % level.w) + DIRS[d].dc;
  if (r < 0 || r >= level.h || c < 0 || c >= level.w) return -1;
  return r * level.w + c;
}

// Shortest walk (no pushes) from `from` to `to` over floor cells, treating
// the cells in `blocked` as occupied. Returns the move string or null.
function bfsPath(level: Level, from: number, to: number, blocked: Set<number>): string | null {
  if (from === to) return '';
  const prev = new Map<number, { from: number; ch: string }>();
  const queue: number[] = [from];
  prev.set(from, { from: -1, ch: '' });
  let head = 0;
  while (head < queue.length) {
    const cur = queue[head++];
    for (let d = 0; d < 4; d++) {
      const nxt = stepCell(level, cur, d);
      if (nxt === -1 || level.wall[nxt] || blocked.has(nxt) || prev.has(nxt)) continue;
      prev.set(nxt, { from: cur, ch: DIRS[d].ch });
      if (nxt === to) {
        let s = '';
        let node = nxt;
        while (node !== from) {
          const p = prev.get(node)!;
          s = p.ch + s;
          node = p.from;
        }
        return s;
      }
      queue.push(nxt);
    }
  }
  return null;
}

// Replay a move string on a level. Returns whether every box ends on a goal
// and how many of the steps pushed a box.
function replay(level: Level, moves: string): { ok: boolean; pushes: number } {
  const boxes = new Set<number>(level.boxes);
  let player = level.player;
  let pushes = 0;
  for (const ch of moves) {
    const d = DIRS.findIndex((x) => x.ch === ch);
    if (d === -1) return { ok: false, pushes };
    const target = stepCell(level, player, d);
    if (target === -1 || level.wall[target]) return { ok: false, pushes };
    if (boxes.has(target)) {
      const beyond = stepCell(level, target, d);
      if (beyond === -1 || level.wall[beyond] || boxes.has(beyond)) return { ok: false, pushes };
      boxes.delete(target);
      boxes.add(beyond);
      pushes++;
    }
    player = target;
  }
  for (const b of boxes) if (!level.goal[b]) return { ok: false, pushes };
  return { ok: true, pushes };
}

// The hallway is the player's row (or column) of floor running across the
// board. Returns its floor cells, or null when no clean row/column fits.
function findHallway(level: Level): Set<number> | null {
  const pr = Math.floor(level.player / level.w);
  const pc = level.player % level.w;
  let rowFloor = 0;
  for (let c = 0; c < level.w; c++) if (!level.wall[pr * level.w + c]) rowFloor++;
  if (rowFloor >= level.w - 2) {
    const cells = new Set<number>();
    for (let c = 0; c < level.w; c++) if (!level.wall[pr * level.w + c]) cells.add(pr * level.w + c);
    return cells;
  }
  let colFloor = 0;
  for (let r = 0; r < level.h; r++) if (!level.wall[r * level.w + pc]) colFloor++;
  if (colFloor >= level.h - 2) {
    const cells = new Set<number>();
    for (let r = 0; r < level.h; r++) if (!level.wall[r * level.w + pc]) cells.add(r * level.w + pc);
    return cells;
  }
  return null;
}

interface Room {
  door: number; // global cell index of the door (inside the room)
  entry: number; // global cell index of the hallway cell outside the door
  cells: number[]; // global cell indices of the room (includes door, excludes entry)
  boxes: number[]; // global box cells in this room
  goals: number[]; // global goal cells in this room
}

// Split a level into its rooms: connected floor components once the hallway
// cells are removed. Each room must hold as many boxes as goals and touch
// the hallway through exactly one door cell. Returns null when the shape
// does not match (the caller then falls back to a whole-level search).
function findRooms(level: Level): Room[] | null {
  const hallway = findHallway(level);
  if (hallway === null) return null;
  const boxSet = new Set(level.boxes);
  const seen = new Set<number>();
  const rooms: Room[] = [];
  const total = level.w * level.h;
  for (let i = 0; i < total; i++) {
    if (level.wall[i] || hallway.has(i) || seen.has(i)) continue;
    const comp: number[] = [];
    const queue: number[] = [i];
    seen.add(i);
    let head = 0;
    while (head < queue.length) {
      const cur = queue[head++];
      comp.push(cur);
      for (let d = 0; d < 4; d++) {
        const nb = stepCell(level, cur, d);
        if (nb === -1 || level.wall[nb] || hallway.has(nb) || seen.has(nb)) continue;
        seen.add(nb);
        queue.push(nb);
      }
    }
    const boxes = comp.filter((c) => boxSet.has(c));
    const goals = comp.filter((c) => level.goal[c]);
    if (boxes.length === 0 && goals.length === 0) continue;
    if (boxes.length === 0 || goals.length === 0 || boxes.length !== goals.length) return null;
    const doors: { door: number; entry: number }[] = [];
    for (const c of comp) {
      for (let d = 0; d < 4; d++) {
        const nb = stepCell(level, c, d);
        if (nb !== -1 && hallway.has(nb)) doors.push({ door: c, entry: nb });
      }
    }
    if (doors.length !== 1) return null;
    rooms.push({ door: doors[0].door, entry: doors[0].entry, cells: comp, boxes, goals });
  }
  const covered = new Set<number>();
  for (const r of rooms) for (const c of r.cells) covered.add(c);
  for (const b of level.boxes) if (!covered.has(b)) return null;
  for (let i = 0; i < total; i++) if (level.goal[i] && !covered.has(i)) return null;
  rooms.sort((a, b) => a.door - b.door);
  return rooms;
}

interface SearchNode {
  boxes: number[]; // sorted local cell indices
  player: number; // local cell index
  region: bigint; // player-reachable cells from `player` (bitmask over local cells)
  key: string;
  g: number; // pushes so far
  h: number; // heuristic: remaining pushes, lower bound
  parent: SearchNode | null;
  push: { box: number; dir: number } | null; // push that produced this node
}

interface HeapEntry { f: number; h: number; g: number; seq: number; node: SearchNode; }

class Heap {
  private a: HeapEntry[] = [];
  private cmp: (x: HeapEntry, y: HeapEntry) => number;
  constructor(cmp: (x: HeapEntry, y: HeapEntry) => number) {
    this.cmp = cmp;
  }
  get size(): number { return this.a.length; }
  push(e: HeapEntry): void {
    this.a.push(e);
    let i = this.a.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (this.cmp(this.a[i], this.a[p]) < 0) {
        const t = this.a[i]; this.a[i] = this.a[p]; this.a[p] = t;
        i = p;
      } else break;
    }
  }
  pop(): HeapEntry | undefined {
    const top = this.a[0];
    const last = this.a.pop();
    if (this.a.length > 0 && last !== undefined) {
      this.a[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1;
        const r = l + 1;
        let m = i;
        if (l < this.a.length && this.cmp(this.a[l], this.a[m]) < 0) m = l;
        if (r < this.a.length && this.cmp(this.a[r], this.a[m]) < 0) m = r;
        if (m === i) break;
        const t = this.a[i]; this.a[i] = this.a[m]; this.a[m] = t;
        i = m;
      }
    }
    return top;
  }
}

interface SolveRoomOpts {
  level: Level;
  cells: number[]; // search cells (global indices); entry is one of them
  roomCells: Set<number>; // global cells a box may occupy
  boxes: number[]; // global
  goals: number[]; // global
  entry: number; // global, where the player starts
  door: number; // global, or -1 when there is no door
  mustExit: boolean; // true when the player must be able to leave via the door
}

// Push-based A* over one room (or the whole level). Returns the push list
// (local box index + direction) or null when the budget runs out. First an
// optimal A* pass; if it exhausts its budget, a greedy best-first pass that
// still finds a solution when one exists within its own budget.
function solveRoom(opts: SolveRoomOpts): { box: number; dir: number }[] | null {
  const { level } = opts;
  const n = opts.cells.length;
  const localOf = new Map<number, number>();
  opts.cells.forEach((g, i) => localOf.set(g, i));
  const isRoom: boolean[] = opts.cells.map((g) => opts.roomCells.has(g));
  const goalLocal = new Set<number>(opts.goals.map((g) => localOf.get(g)!));
  const boxLocal = opts.boxes.map((g) => localOf.get(g)!).sort((a, b) => a - b);
  const entryLocal = localOf.get(opts.entry)!;
  const doorLocal = opts.door >= 0 ? localOf.get(opts.door)! : -1;

  // Local adjacency: for each local cell, the neighbour local index per
  // direction, or -1 when the neighbour is not a search cell.
  const nbr: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    for (let d = 0; d < 4; d++) {
      const g = stepCell(level, opts.cells[i], d);
      row.push(g === -1 ? -1 : (localOf.get(g) ?? -1));
    }
    nbr.push(row);
  }

  // All-pairs shortest-path distances over the search cells.
  const dist: number[][] = [];
  for (let s = 0; s < n; s++) {
    const row = new Array<number>(n).fill(Infinity);
    row[s] = 0;
    const q: number[] = [s];
    let head = 0;
    while (head < q.length) {
      const u = q[head++];
      for (let d = 0; d < 4; d++) {
        const v = nbr[u][d];
        if (v !== -1 && row[v] === Infinity) {
          row[v] = row[u] + 1;
          q.push(v);
        }
      }
    }
    dist.push(row);
  }

  // A cell is dead when no goal is reachable from it (a box there can never
  // be pushed onto a goal). Goal cells are never dead.
  const dead: boolean[] = [];
  for (let i = 0; i < n; i++) {
    let live = false;
    for (const g of goalLocal) if (dist[i][g] < Infinity) { live = true; break; }
    dead.push(!live);
  }

  // For every local cell, the 2x2 squares it belongs to (four local indices
  // each, -1 for a wall or outside cell). A square whose four cells are all
  // wall-or-box with no goal inside is a frozen deadlock.
  const squares: number[][][] = [];
  for (let i = 0; i < n; i++) {
    const g = opts.cells[i];
    const r = Math.floor(g / level.w);
    const c = g % level.w;
    const list: number[][] = [];
    const corners: [number, number][] = [[r - 1, c - 1], [r - 1, c], [r, c - 1], [r, c]];
    for (const [r0, c0] of corners) {
      const sq: number[] = [];
      for (const [rr, cc] of [[r0, c0], [r0, c0 + 1], [r0 + 1, c0], [r0 + 1, c0 + 1]] as [number, number][]) {
        if (rr < 0 || rr >= level.h || cc < 0 || cc >= level.w) { sq.push(-1); continue; }
        const gg = rr * level.w + cc;
        const li = localOf.get(gg);
        sq.push(li === undefined ? -1 : li);
      }
      list.push(sq);
    }
    squares.push(list);
  }

  // Minimum-assignment heuristic: the cheapest way to match boxes to goals
  // by shortest-path distance. Every push moves one box one cell, so this
  // sum is a lower bound on the pushes still needed (admissible).
  const hCache = new Map<string, number>();
  function heuristic(boxes: number[]): number {
    const key = boxes.join(',');
    const hit = hCache.get(key);
    if (hit !== undefined) return hit;
    let h = 0;
    if (goalLocal.size === boxes.length) {
      const gs = [...goalLocal];
      let best = Infinity;
      const used = new Array<boolean>(gs.length).fill(false);
      const rec = (bi: number, acc: number): void => {
        if (acc >= best) return;
        if (bi === boxes.length) { best = acc; return; }
        for (let gi = 0; gi < gs.length; gi++) {
          if (used[gi]) continue;
          const d = dist[boxes[bi]][gs[gi]];
          if (d === Infinity) continue;
          used[gi] = true;
          rec(bi + 1, acc + d);
          used[gi] = false;
        }
      };
      rec(0, 0);
      h = best;
    } else {
      for (const b of boxes) {
        let m = Infinity;
        for (const g of goalLocal) m = Math.min(m, dist[b][g]);
        if (m === Infinity) { h = Infinity; break; }
        h += m;
      }
    }
    hCache.set(key, h);
    return h;
  }

  // Player-reachable region from `start` without moving a box, as a bitmask.
  function computeRegion(boxes: number[], start: number): bigint {
    const blocked = new Set(boxes);
    const seen = new Array<boolean>(n).fill(false);
    let mask = 0n;
    const q: number[] = [start];
    seen[start] = true;
    mask |= 1n << BigInt(start);
    let head = 0;
    while (head < q.length) {
      const u = q[head++];
      for (let d = 0; d < 4; d++) {
        const v = nbr[u][d];
        if (v === -1 || blocked.has(v) || seen[v]) continue;
        seen[v] = true;
        mask |= 1n << BigInt(v);
        q.push(v);
      }
    }
    return mask;
  }

  function frozen2x2(boxes: number[], moved: number): boolean {
    const boxSet = new Set(boxes);
    for (const sq of squares[moved]) {
      let hasGoal = false;
      let blocked = true;
      for (const cell of sq) {
        if (cell === -1) continue;
        if (goalLocal.has(cell)) hasGoal = true;
        if (!boxSet.has(cell)) blocked = false;
      }
      if (blocked && !hasGoal) return true;
    }
    return false;
  }

  function isGoal(boxes: number[]): boolean {
    for (const b of boxes) if (!goalLocal.has(b)) return false;
    if (opts.mustExit && doorLocal >= 0 && boxes.includes(doorLocal)) return false;
    return true;
  }

  function search(mode: 'astar' | 'greedy', budget: number): SearchNode | null {
    const startBoxes = boxLocal.slice();
    const startH = heuristic(startBoxes);
    if (startH === Infinity) return null;
    for (const b of startBoxes) if (frozen2x2(startBoxes, b)) return null;
    const startRegion = computeRegion(startBoxes, entryLocal);
    const startKey = startBoxes.join(',') + '|' + startRegion.toString();
    const cmp = mode === 'astar'
      ? (x: HeapEntry, y: HeapEntry) => (x.f - y.f) || (x.h - y.h) || (x.seq - y.seq)
      : (x: HeapEntry, y: HeapEntry) => (x.h - y.h) || (y.g - x.g) || (x.seq - y.seq);
    const heap = new Heap(cmp);
    const closed = new Map<string, number>();
    let seq = 0;
    const startNode: SearchNode = {
      boxes: startBoxes, player: entryLocal, region: startRegion, key: startKey,
      g: 0, h: startH, parent: null, push: null,
    };
    heap.push({ f: startH, h: startH, g: 0, seq: seq++, node: startNode });
    closed.set(startKey, 0);
    let expansions = 0;
    while (heap.size > 0) {
      const cur = heap.pop()!;
      const node = cur.node;
      const bestG = closed.get(node.key);
      if (bestG !== undefined && bestG < node.g) continue;
      if (isGoal(node.boxes)) return node;
      if (++expansions > budget) return null;
      const boxSet = new Set(node.boxes);
      for (let bi = 0; bi < node.boxes.length; bi++) {
        const b = node.boxes[bi];
        for (let d = 0; d < 4; d++) {
          const dest = nbr[b][d];
          if (dest === -1 || !isRoom[dest] || boxSet.has(dest)) continue;
          const behind = nbr[b][(d + 2) % 4];
          if (behind === -1 || ((node.region >> BigInt(behind)) & 1n) === 0n) continue;
          const nb = node.boxes.slice();
          nb[bi] = dest;
          nb.sort((a, b2) => a - b2);
          if (dead[dest]) continue;
          if (frozen2x2(nb, dest)) continue;
          const h = heuristic(nb);
          if (h === Infinity) continue;
          const g = node.g + 1;
          const nRegion = computeRegion(nb, b);
          const nKey = nb.join(',') + '|' + nRegion.toString();
          const prevG = closed.get(nKey);
          if (prevG !== undefined && prevG <= g) continue;
          closed.set(nKey, g);
          const nn: SearchNode = {
            boxes: nb, player: b, region: nRegion, key: nKey,
            g, h, parent: node, push: { box: b, dir: d },
          };
          heap.push({ f: mode === 'astar' ? g + h : h, h, g, seq: seq++, node: nn });
        }
      }
    }
    return null;
  }

  const found = search('astar', ASTAR_BUDGET) ?? search('greedy', GREEDY_BUDGET);
  if (found === null) return null;
  const pushes: { box: number; dir: number }[] = [];
  let cur: SearchNode | null = found;
  while (cur !== null && cur.push !== null) {
    pushes.push(cur.push);
    cur = cur.parent;
  }
  pushes.reverse();
  return pushes;
}

// Turn a push list into player moves, walking the player into place before
// each push. `boxes` is the current global box list (updated in place by
// reference through the returned moves only; the caller tracks positions).
function pushesToMoves(
  level: Level,
  searchCells: number[],
  pushes: { box: number; dir: number }[],
  startPlayer: number,
  startBoxes: number[],
): { moves: string; player: number; boxes: number[] } | null {
  let moves = '';
  let player = startPlayer;
  const boxes = startBoxes.slice();
  for (const p of pushes) {
    const boxGlobal = searchCells[p.box];
    const d = DIRS[p.dir];
    const behind = boxGlobal - d.dr * level.w - d.dc;
    const walk = bfsPath(level, player, behind, new Set(boxes));
    if (walk === null) return null;
    moves += walk;
    moves += d.ch;
    const dest = boxGlobal + d.dr * level.w + d.dc;
    for (let i = 0; i < boxes.length; i++) if (boxes[i] === boxGlobal) boxes[i] = dest;
    player = boxGlobal;
  }
  return { moves, player, boxes };
}

function solveLevel(level: Level): string {
  const rooms = findRooms(level);
  if (rooms !== null) {
    // Rooms whose door is itself a goal go last, so a box may finish on the
    // door; every other room must end with its door free to walk out.
    const ordered = rooms.slice().sort((a, b) => {
      const ga = level.goal[a.door] ? 1 : 0;
      const gb = level.goal[b.door] ? 1 : 0;
      return ga - gb || a.door - b.door;
    });
    let moves = '';
    let totalPushes = 0;
    let curPlayer = level.player;
    let curBoxes = level.boxes.slice();
    for (let ri = 0; ri < ordered.length; ri++) {
      const room = ordered[ri];
      const mustExit = ri < ordered.length - 1;
      const walkIn = bfsPath(level, curPlayer, room.entry, new Set(curBoxes));
      if (walkIn === null) return '';
      moves += walkIn;
      curPlayer = room.entry;
      const searchCells = room.cells.concat([room.entry]);
      const pushes = solveRoom({
        level,
        cells: searchCells,
        roomCells: new Set(room.cells),
        boxes: room.boxes,
        goals: room.goals,
        entry: room.entry,
        door: room.door,
        mustExit,
      });
      if (pushes === null) return '';
      const conv = pushesToMoves(level, searchCells, pushes, curPlayer, curBoxes);
      if (conv === null) return '';
      moves += conv.moves;
      totalPushes += pushes.length;
      curPlayer = conv.player;
      curBoxes = conv.boxes;
      if (mustExit) {
        const walkOut = bfsPath(level, curPlayer, room.entry, new Set(curBoxes));
        if (walkOut === null) return '';
        moves += walkOut;
        curPlayer = room.entry;
      }
    }
    const v = replay(level, moves);
    if (!v.ok || v.pushes !== totalPushes) return '';
    return moves;
  }

  // Fallback: treat the whole level as one room.
  const floorCells: number[] = [];
  const allGoals: number[] = [];
  for (let i = 0; i < level.w * level.h; i++) {
    if (!level.wall[i]) floorCells.push(i);
    if (level.goal[i]) allGoals.push(i);
  }
  const pushes = solveRoom({
    level,
    cells: floorCells,
    roomCells: new Set(floorCells),
    boxes: level.boxes,
    goals: allGoals,
    entry: level.player,
    door: -1,
    mustExit: false,
  });
  if (pushes === null) return '';
  const conv = pushesToMoves(level, floorCells, pushes, level.player, level.boxes);
  if (conv === null) return '';
  const v = replay(level, conv.moves);
  return v.ok && v.pushes === pushes.length ? conv.moves : '';
}

for (const id of readdirSync(INPUT_DIR).sort()) {
  let moves = '';
  try {
    const text = readFileSync(join(INPUT_DIR, id, 'level.txt'), 'utf8');
    moves = solveLevel(parseLevel(text));
  } catch {
    moves = '';
  }
  mkdirSync(join(OUTPUT_DIR, id), { recursive: true });
  writeFileSync(join(OUTPUT_DIR, id, 'solution.txt'), moves + '\n');
}
