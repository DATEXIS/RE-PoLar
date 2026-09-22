"""MCTS program discovery, reimplemented from the PoLar paper (no code published).

"Edits over identity" formulation (Option A). The program is IDENTITY by
default; the search places a small set of *edits*, each edit = one contiguous
<=MAX_SEGMENT_LEN segment assigned a non-keep op (skip or repeat) placed
anywhere in [0, D). An edit at layer 30 is therefore ONE action from the
root: there's no left-to-right walk, so every depth is reachable
immediately and budget is spent uniformly across the stack.

    (The earlier left-to-right / identity-TAIL construction could only edit
    layers the tree physically walked to; its frontier plateaued at ~layer 11
    regardless of budget, truncating all supervision to the front third of the
    stack. That is the bug this rewrite fixes.)

The tree (a DAG, see "TRANSPOSITION" below):
  * Node = the SET of edits placed so far, non-overlapping, stored sorted by
    start. The sorted tuple is the node's IDENTITY.
  * Root = identity (no edits).
  * Every node is a complete, evaluable Program: identity everywhere except its
    edits (gaps filled with KEEP, tiled <=MAX_SEGMENT_LEN). Gaps are the
    identity *baseline*, not a truncation, future expansions still add edits at
    any depth.
  * Actions at a node: for every start in [0, D), every length in
    1..min(MAX_SEGMENT_LEN, D-start) whose range [start, start+length) does not
    intersect an ALREADY-PLACED edit, every op in {SKIP, REPEAT} -> place that
    edit. KEEP is never an explicit edit, it is the default. The untried list
    is shuffled with the tree RNG and popped on expansion.

    ANY-ORDER PLACEMENT (replaces an earlier `start >= cursor` rule).
    Under that earlier rule, a node carried a cursor = end index of the last placed edit,
    and actions were restricted to `start >= cursor`, i.e. every edit after the
    first had to go to the RIGHT of the previous one. That was purely a device
    for tree properness (it gave each edit-set exactly one construction path),
    but it made exploration effort land disproportionately on late layers, and
    the effect was large. Measured with the real search under a CONSTANT reward,
    so the numbers are pure search geometry with zero model signal (D=36, B=200,
    lam=5.0, eps=0.1, r<=4, 300 trees):
      - 50.0% of all placed edits anchored in the late third (layers 24-35), vs
        30.4% under a uniform-over-actions baseline.
      - by edit index within a program: 1st edit 37/36/27 (early/mid/late, i.e.
        essentially unbiased), 2nd edit 5/26/68, 3rd edit 0.3/9/91.
      - a node whose cursor sat in the early third expanded 0.67% of its own
        remaining action space; a late-cursor node expanded 2.79%, same effort,
        4.2x different coverage, because the early node still had ~553 legal
        follow-ups and the late one only ~85.
      - net: a SPECIFIC early->early 2-edit program was 4.3x less likely to be
        evaluated than a specific late->late one.
    Cross-checked against real supervision (one full DART-Math difficulty
    tier, 17,494 discovered valid programs): 2nd/3rd-edit positions matched the zero-signal numbers
    above to within ~2 points (68.3 vs 68.4, 88.6 vs 90.7), i.e. WHERE the
    2nd/3rd segment of a discovered program sat was predicted by the ordering
    rule alone, not by the model. (1st-edit positions did NOT match the null,
    51.7% early vs 37.0%, that part was real signal.) Dropping the restriction
    removes the bias: every node's action space now spans the whole stack.

    NOT the same as allowing NESTED/overlapping edits (the literal reading of
    ICML App. B.3, where actions apply to the current PATH rather than the
    original index space, e.g. repeat [2,4) then skip only the second copy of
    layer 3 -> path 0,1,2,3,2,4,5). That is deliberately still forbidden: such
    a path has NO decomposition into contiguous original-index segments, so it
    is not representable as a `Program` at all (`re_polar.router.train.
    program_from_layer_path` raises on it), and the router's output space
    (per-layer boundary logits + per-segment op logits) literally cannot emit
    one. Searching them would spend budget on supervision the router must
    discard. A deliberate scope decision, not an oversight.

  * TRANSPOSITION: without the ordering rule, (A then B) and (B then A) place
    the same edit SET and must not become two nodes, that would split visit
    statistics and waste budget re-evaluating one program under two identities.
    `_nodes` maps the canonical sorted edit tuple -> Node, so both orders reach
    the SAME node object and the tree is a DAG. A node's `untried`/`children`
    are therefore per-EDIT-SET, not per-path, which is correct: the legal
    follow-up edits depend only on which ranges are occupied, not on the order
    they were placed in. Paths stay acyclic (each action adds exactly one edit,
    so edit-count strictly increases), and `update()` backpropagates along the
    chain actually traversed.
  * PROGRESSIVE WIDENING: the root's action set (~276 edits for D=36) dwarfs the
    budget (~200), so without a widening cap the root would never empty its
    untried list, selection would never descend, and the tree would stay
    depth-1, every propose() just a single edit. The paper's programs are multi-edit
    (e.g. skip layer 0 + repeat d,d+1,d+2), so we cap a node's children:

        k(visits) = max(1, ceil(alpha * visits ** beta))

    A node may expand a NEW child only while len(children) < k(visits);
    otherwise selection descends by UCB into an existing child. k(0)=k(1)=... >=1
    so one edit is always allowed (single-edit depth reachability is preserved),
    and k grows sublinearly so budget flows into deeper, multi-edit combinations.

Selection uses

    UCB(child) = R/v + c * sqrt(ln V / v) - lam * executed_len(child) / D

where executed_len = len(program.to_layer_path()) counts *executed* layer
indices (skips subtract, repeats add), so the penalty prefers shorter programs.
The is_valid guard is kept: a program that skips every layer -> empty path ->
reward 0, not proposed (rare with edits-over-identity, but still guarded).

The tree is GPU-free: propose() yields complete programs to evaluate and
update() takes the reward back, batching across many trees happens in
scheduler.py (program-major). Evaluated programs and rewards accumulate in
.evaluated for the supervision output.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from re_polar.core import MAX_SEGMENT_LEN, Op, Program, Segment, is_valid

# An edit = one non-keep op on a contiguous segment [start, start+length).
Edit = Tuple[int, int, Op, int]  # (start, length, op, times); times=1 for SKIP, >=2 for REPEAT

# PoLar's MCTS bounds the repetition count r<=4 (paper Appendix B.2: "block size k and
# repetition count r are bounded by small constants (k, r<=4)"). Our reconstruction
# historically defaulted to 2 (a single 2x loop) -- an UNDER-approximation of the paper's
# search space. `times` here counts TOTAL executions of the segment, so "r repetitions"
# beyond the original pass means times = r+1 -> pass max_repeat_times=5 to match r<=4,
# not 4 (this comment previously said 4, an off-by-one against the `times`
# semantics below; production runs already used 5 and were correct -- this
# comment was the stale part). (The released router/
# parser only decode 2x, which is what earlier misled us into thinking the search was
# 2x-only; r>2 improves the search/oracle until the router grows a times-head.)
DEFAULT_MAX_REPEAT_TIMES = 2

# Interned action universe, keyed by (num_layers, max_repeat_times). Every node's
# `untried` holds REFERENCES into this one list instead of freshly-built tuples.
#
# This matters at production scale. Dropping the `start >= cursor` rule (module
# docstring) means a deep node's action set no longer shrinks -- it stays ~630
# edits instead of decaying toward ~85 -- so a fully-searched tree holds ~190
# nodes x ~630 actions. With fresh 4-tuples that measured 8.9 MB/tree, i.e. 13 GB
# for DART's 1500 concurrent trees and 55 GB for a 6180-input mmlu run, since
# MCTSRunner keeps every tree alive for the whole search. Interning makes each
# untried entry an 8-byte pointer rather than a ~72-byte tuple, cutting it ~10x.
_ACTION_UNIVERSE: Dict[Tuple[int, int], List[Edit]] = {}


def _action_universe(num_layers: int, max_repeat_times: int) -> List[Edit]:
    """Every edit placeable on an EMPTY program, in canonical (start, length, op)
    order -- the superset every node's action list is filtered out of."""
    key = (num_layers, max_repeat_times)
    universe = _ACTION_UNIVERSE.get(key)
    if universe is None:
        universe = []
        for start in range(num_layers):
            for length in range(1, min(MAX_SEGMENT_LEN, num_layers - start) + 1):
                universe.append((start, length, Op.SKIP, 1))
                for times in range(2, max_repeat_times + 1):
                    universe.append((start, length, Op.REPEAT, times))
        _ACTION_UNIVERSE[key] = universe
    return universe


@dataclass
class Node:
    """One edit-SET. `edits` (sorted by start) is the node's identity, see the
    module docstring's TRANSPOSITION note; the same set is one node no matter
    which order its edits were placed in, so a node may have several parents."""

    edits: Tuple[Edit, ...]  # canonical: sorted by start, non-overlapping
    executed_len: int        # len(to_layer_path()) of this node's program
    untried: List[Edit]
    children: Dict[Edit, "Node"] = field(default_factory=dict)
    visits: int = 0
    total_reward: float = 0.0
    # Only set/used by ProgramMCTS(global_selection=True) (see GLOBAL SELECTION
    # MODE below) -- the default hierarchical+DAG mode never populates this
    # (a node there may have several parents, so a single back-pointer would be
    # meaningless; None throughout for that mode).
    parent: Optional["Node"] = None


class ProgramMCTS:
    """One search tree for one input. propose() -> evaluate externally -> update()."""

    def __init__(self, num_layers: int, budget: int, c: float = math.sqrt(2),
                 lam: float = 5.0, seed: int = 0, *,
                 alpha: float = 2.0, beta: float = 0.5,
                 max_repeat_times: int = DEFAULT_MAX_REPEAT_TIMES,
                 epsilon: float = 0.0,
                 global_selection: bool = False,
                 ucb_global_v: bool = True):
        self.num_layers = num_layers
        self.budget = budget
        self.c = c
        self.lam = lam
        self.seed = seed
        self.alpha = alpha  # progressive-widening scale
        self.beta = beta    # progressive-widening exponent (k = alpha * visits**beta)
        self.max_repeat_times = max_repeat_times  # segment exec-count cap; paper bounds r<=4
        # PRE paper Appendix B (arXiv:2507.07996, same MCTS pre-ICML-rename): "the
        # algorithm selects a random unexplored child node with probability 0.1
        # instead of the one with the highest UCB score. This behavior is
        # hard-coded as a fixed conditional in the selection logic." Default is
        # now 0.0: this specific reproduction
        # run deliberately isolates widening+global-V from epsilon's own
        # (separately, already-known-non-paper-literal, see propose()'s legacy
        # branch) exploration mechanism; pass epsilon=0.1 to match the paper's
        # stated value.
        self.epsilon = epsilon
        # UCB's V term (legacy hierarchical mode only -- see _ucb): PoLar states
        # BOTH papers' Appendix B verbatim as "V is the total number of
        # simulations" (ICML B.3) -- a global counter, not each comparison's own
        # parent's visit count. Default True ("stay close to
        # PoLar as much as possible"), superseding an earlier decision to
        # deliberately keep local
        # `parent.visits` instead (standard UCT/UCB1, Kocsis & Szepesvari 2006)
        # -- pass ucb_global_v=False for that textbook-UCT reading instead.
        # No effect under global_selection=True: that mode already always uses
        # global V via its own dedicated _global_ucb, never calls _ucb at all.
        self.ucb_global_v = ucb_global_v
        self.rng = random.Random(seed)
        self._universe = _action_universe(num_layers, max_repeat_times)
        # canonical sorted edit-tuple -> Node (module docstring, TRANSPOSITION)
        self._nodes: Dict[Tuple[Edit, ...], Node] = {}
        self.root = self._get_or_create(())
        self.proposals = 0
        # GLOBAL SELECTION MODE (see _propose_global's docstring for the
        # mechanism). Piloted as the default, then flipped back
        # to hierarchical+widening (global_selection=False) after
        # real-data evidence showed this mode's flat
        # tree-wide argmax produces a median tree-evaluation depth of 10 edits
        # -- prompt-independent, confirmed on complete production data -- vs.
        # PoLar's own "modest programs" framing (ICML Finding 4). Pass
        # global_selection=True to opt back into this mode. Bootstrapped
        # lazily by _propose_global's first call, not here, so a tree that
        # never calls propose() costs nothing extra either way.
        self.global_selection = global_selection
        self._all_nodes: List[Node] = [self.root]      # flat scan pool (global_selection only)
        # paper's literal "V" -- tracked for BOTH modes (see update()); only
        # ever READ by global_selection's _global_ucb or by _ucb when
        # ucb_global_v=True, but cheap enough (one int) to always maintain.
        self._global_visits_total = 0
        # path tuple -> reward, for every program actually evaluated
        self.evaluated: Dict[Tuple[int, ...], float] = {}
        # SEARCH TRAJECTORY: one entry per update() call, in
        # the exact order simulations actually happened -- lets a consumer
        # reconstruct the real tree (parent/child edges + the order nodes were
        # visited) after the fact, for both modes (works uniformly since
        # update() is shared code). Every entry always exists (the identity
        # bootstrap calls update() too, same as any other proposal); a
        # degenerate all-skip retry (program=None) still gets one, with its
        # own (empty) executed path -- it's a real tree node the search
        # actually visited, not a no-op.
        self.trajectory: List[dict] = []

    @property
    def exhausted(self) -> bool:
        return self.proposals >= self.budget

    # -- action / program construction -------------------------------------

    def _actions_for(self, edits: Sequence[Edit]) -> List[Edit]:
        """Every edit placeable given the already-placed `edits`: any (start, length)
        whose range [start, start+length) does not intersect an existing edit's range,
        anywhere in [0, D), NOT restricted to the right of the last edit (module
        docstring, ANY-ORDER PLACEMENT).

        One SKIP plus one REPEAT per allowed loop count (2..max) per (start, length),
        in increasing (start, length) order; the caller shuffles. At the root (no
        edits placed) this is exactly the action set the earlier `_actions_at(0)`
        (before ANY-ORDER PLACEMENT) produced, in the same order, only DEEPER
        nodes see a different (larger) set.
        """
        if not edits:
            return list(self._universe)
        occupied = bytearray(self.num_layers)
        for start, length, _op, _times in edits:
            for i in range(start, start + length):
                occupied[i] = 1
        # filter the interned universe rather than building fresh tuples, so a
        # node's untried list costs one pointer per entry (see _ACTION_UNIVERSE)
        return [a for a in self._universe
                if not any(occupied[i] for i in range(a[0], a[0] + a[1]))]

    def _child_edits(self, edits: Tuple[Edit, ...], action: Edit) -> Tuple[Edit, ...]:
        """Canonical (sorted-by-start) edit tuple for placing `action` on `edits`.

        Starts are unique across a node's edits (they never overlap), so sorting the
        raw tuples orders them by start, this is the node identity key that makes
        (A then B) and (B then A) the same node."""
        return tuple(sorted(edits + (action,)))

    def _get_or_create(self, edits: Tuple[Edit, ...]) -> Node:
        """The node for this exact edit SET, reusing it if some other placement
        order already built it (module docstring, TRANSPOSITION)."""
        node = self._nodes.get(edits)
        if node is None:
            node = Node(edits=edits, executed_len=self._executed_len(edits),
                        untried=self._actions_for(edits))
            self.rng.shuffle(node.untried)
            self._nodes[edits] = node
        return node

    def _executed_len(self, edits: Sequence[Edit]) -> int:
        """len(to_layer_path()) computed from the edits (identity baseline = D).

        Each layer is executed once by default; a SKIP edit removes its layers,
        a REPEAT edit adds (times-1) extra copies. Equals len(to_layer_path())
        for every valid program; 0 for the degenerate all-skip program.
        """
        total = self.num_layers
        for _start, length, op, times in edits:
            if op is Op.SKIP:
                total -= length
            elif op is Op.REPEAT:
                total += (times - 1) * length
        return total

    def _build_program(self, edits: Sequence[Edit]) -> Program:
        """Identity over [0, D) with the edits applied; gaps -> KEEP tiled <=4.

        `edits` must be in increasing-start, non-overlapping order, guaranteed for
        any `Node.edits` by `_child_edits`'s canonical sort, so a single
        left-to-right pass yields a contiguous <=MAX_SEGMENT_LEN-segment cover of
        [0, D). (Callers passing a hand-built list, e.g. tests, must sort it
        themselves.)
        """
        segments: List[Segment] = []
        pos = 0

        def fill_keep(until: int):
            nonlocal pos
            while pos < until:
                size = min(MAX_SEGMENT_LEN, until - pos)
                segments.append(Segment(pos, pos + size, Op.KEEP))
                pos += size

        for start, length, op, times in edits:
            fill_keep(start)
            params = {"times": times} if op is Op.REPEAT else {}
            segments.append(Segment(start, start + length, op, params))
            pos = start + length
        fill_keep(self.num_layers)

        return Program(num_layers=self.num_layers, segments=segments)

    # -- MCTS --------------------------------------------------------------

    def _ucb(self, parent: Node, child: Node) -> float:
        exploit = child.total_reward / child.visits
        v = self._global_visits_total if self.ucb_global_v else parent.visits
        explore = self.c * math.sqrt(math.log(v) / child.visits)
        penalty = self.lam * child.executed_len / self.num_layers
        return exploit + explore - penalty

    def _widening_limit(self, visits: int) -> int:
        """Max children a node with `visits` visits may hold (progressive widening)."""
        return max(1, math.ceil(self.alpha * (visits ** self.beta)))

    def _can_expand(self, node: Node) -> bool:
        """True iff `node` may add a NEW child now (has an untried edit and is
        under its progressive-widening cap)."""
        return bool(node.untried) and len(node.children) < self._widening_limit(node.visits)

    def _expand(self, node: Node, action: Edit) -> Node:
        """Place `action` on `node`, link the edge, return the child (shared by
        normal expansion and the epsilon override).

        The child may already exist, reached earlier via a different placement
        ORDER of the same edit set, in which case this only adds the edge, which
        is the whole point of the transposition table (module docstring)."""
        child = self._get_or_create(self._child_edits(node.edits, action))
        node.children[action] = child
        return child

    # -- GLOBAL SELECTION MODE (piloted as the default, later reverted; pass
    # -- global_selection=True to opt back in -- see the __init__ comment for
    # -- why hierarchical+widening is the default again) ------------------
    #
    # A literal reading of CoLa/PoLar's own Algorithm 1 (PRE paper,
    # arXiv:2507.07996, App. B):
    #
    #   1: Initialize root node P0 = [L1, ..., LN]
    #   2: for N = 1 to number of simulations do
    #   3:   Selection: traverse tree maximizing UCB(P)
    #   4:   Expansion: generate skip/repeat candidates if node unexplored
    #   5:   Simulation: evaluate path accuracy on held-out input(s)
    #   6:   Backpropagation: update Q(P) and v(P) along trajectory
    #   7: end for
    #
    # CORRECTION (caught re-checking sources for this comment): the ICML
    # paper (arXiv:2606.06574) does NOT drop this box, contrary to what this
    # comment claimed until now -- it has its own Appendix B.1-B.4 with an
    # equivalent Algorithm 1, missed earlier by a tooling bug (grep silently
    # treats a file as binary and suppresses matches; the extracted text has
    # 2 stray null bytes -- `grep -a` or a real parser is required). The ICML
    # version's line 3 reads "Selection: traverse tree using UCB to reach A
    # LEAF program" -- it says "leaf" explicitly, which the PRE version above
    # does not. That wording leans toward a literal hierarchical descent to
    # an actual childless leaf, closer to the hierarchical+widening default
    # this module actually implements than to the flat/global "compare every
    # node in the whole tree" reading implemented below.
    # NOT re-implemented as of this writing -- flagged here so this mode is
    # not mistaken for settled fidelity; the
    # pilot results below stand on their own regardless of which reading is
    # more textually faithful, but "more textually faithful" no longer
    # confidently favors this implementation over an unbuilt alternative.
    #
    # Under the (still-implemented) global reading below, UCB never scores anything untried (there is no
    # "node" for an action nobody has applied yet -- Node objects are only
    # ever created already-evaluated, see `update()`'s docstring). Instead:
    # UCB picks ONE already-evaluated node from the WHOLE tree (not just
    # siblings under one shared parent -- "traverse tree maximizing UCB(P)"
    # is read as a flat argmax over every node that still has an untried
    # edit, not a level-by-level descent gated by a separate "can this node
    # still expand" rule). That node is extended by exactly one untried edit
    # (line 4's "if node unexplored" = the picked node hasn't had this
    # specific edit tried before), the resulting program is evaluated (line
    # 5), and the reward backpropagates up that node's OWN ancestor chain
    # (line 6 -- root's own `visits` therefore equals the running total of
    # ALL simulations anywhere in the tree, not a frozen count, since root is
    # an ancestor of every node).
    #
    # This needs the paper's literal global V (`_global_visits_total` below)
    # in the UCB formula, not the local `parent.visits` the legacy
    # (global_selection=False) hierarchical mode uses (that mode
    # deliberately keeps local `parent.visits` for its
    # independent-per-node-bandit framing -- that
    # reasoning doesn't apply here, since this mode's selection isn't
    # hierarchical: every node competes against every other node directly,
    # so they need one SHARED reference point, which is exactly what the
    # paper's own "V is the total number of simulations" already is). No
    # progressive widening: nothing artificially caps how many children a
    # node may have -- a node earns further children purely by continuing to
    # win the global comparison against every other existing node.
    #
    # DELIBERATE SIMPLIFICATION vs. the legacy mode: this builds a genuine
    # single-parent TREE, not the transposition-collapsed DAG the legacy
    # (global_selection=False) hierarchical mode uses (module docstring,
    # "TRANSPOSITION") -- reaching the same edit SET via two different
    # construction orders is NOT deduplicated onto one node here, so this
    # mode may occasionally re-discover an equivalent program under a second
    # node. Harmless for the search's OUTPUT (`self.evaluated` is keyed by
    # the program's actual layer-path, so a duplicate discovery doesn't
    # produce a duplicate `valid_paths()` entry, and the scheduler's
    # cross-tree EvalCache still skips the redundant model call) -- just
    # some search-tree bookkeeping that isn't shared the way it is in the
    # legacy mode. Not fixed here; revisit only if a real run shows this
    # actually costs meaningful budget.
    #
    # STATUS: this is our reconstruction of one plausible literal reading of
    # a 7-line, informally-worded algorithm box with no released code to
    # check against (PoLar's own public repo has no MCTS search code at all,
    # only a consumer of precomputed merged_mcts_samples.json -- confirmed by
    # direct search). Not proven to match what the paper's authors actually
    # ran. Piloted as the default after a CPU pilot battery (fake
    # rewards; no GPU/model in this environment) showed it: crash-free across
    # 180 (reward, seed) combinations, deterministic given a seed, held the
    # root.visits==total-simulations invariant throughout, matched-or-beat
    # the legacy mode's discovery rate on every non-degenerate reward tested
    # (including one legacy found ZERO valid programs on across 10 seeds and
    # this mode found ~half the budget's worth of), and ran at comparable
    # wall-clock cost. NOT yet validated against a real model/reward -- that's a
    # deliberately deferred larger-scale item, not something this pilot
    # covers. The legacy mode (global_selection=False) is fully preserved
    # and untouched by any of this -- pass it explicitly for the old
    # behavior, still exercised by its own tests.

    def _global_ucb(self, node: Node) -> float:
        """UCB for `node` using the paper's literal global V (this mode only):
        `self._global_visits_total`, not `node`'s specific parent's visits."""
        exploit = node.total_reward / node.visits
        explore = self.c * math.sqrt(math.log(self._global_visits_total) / node.visits)
        penalty = self.lam * node.executed_len / self.num_layers
        return exploit + explore - penalty

    def _chain_to(self, node: Node) -> List[Node]:
        """root -> ... -> node, via `Node.parent` (well-defined: this mode's
        tree is single-parent by construction, see the class docstring)."""
        chain = []
        cur: Optional[Node] = node
        while cur is not None:
            chain.append(cur)
            cur = cur.parent
        chain.reverse()
        return chain

    def _propose_global(self) -> Optional[Tuple[Program, List[Node]]]:
        """propose(), global-selection variant. See the block comment above
        this method for the full mechanism and its provenance."""
        if self.exhausted:
            return None

        if self._global_visits_total == 0:
            # Algorithm 1 line 1, "Initialize root node P0": happens BEFORE
            # the numbered simulation loop (line 2), i.e. for free, does not
            # consume budget, mirroring the scheduler's existing identity
            # precompute (scheduler.py's `_evaluate_group(self.identity, ...)`,
            # also outside the search budget). No comparison needed: this is
            # unconditionally the tree's very first evaluation.
            return self._build_program(()), [self.root]

        candidates = [n for n in self._all_nodes if n.untried]
        if not candidates:
            return None  # whole action space exhausted (only reachable for tiny D)

        # epsilon override (same constant/meaning as the default mode's, see
        # __init__'s docstring comment): w.p. epsilon, pick uniformly among
        # every node that still has an untried edit, instead of the argmax.
        if self.epsilon > 0 and self.rng.random() < self.epsilon:
            node = self.rng.choice(candidates)
        else:
            node = max(candidates, key=self._global_ucb)

        self.proposals += 1
        action = node.untried.pop()
        child_edits = self._child_edits(node.edits, action)
        child = Node(edits=child_edits, executed_len=self._executed_len(child_edits),
                     untried=self._actions_for(child_edits), parent=node)
        self.rng.shuffle(child.untried)
        node.children[action] = child
        self._all_nodes.append(child)

        program = self._build_program(child.edits)
        if not is_valid(program):  # degenerate all-skip cover -> empty path
            self.update(self._chain_to(child), 0.0, program=None)
            return self._propose_global() if not self.exhausted else None
        return program, self._chain_to(child)

    def propose(self) -> Optional[Tuple[Program, List[Node]]]:
        """Select + expand -> a complete program to evaluate + its backprop chain.

        The program may repeat an already-evaluated one (the scheduler's cache
        absorbs that); None only when the budget is exhausted.

        global_selection=True (was tried as the default for a while, then
        reverted -- see this class's own block comment above): dispatches to
        `_propose_global()` instead, see that method's block comment. Pass
        global_selection=False for the legacy mode below; every line below
        this check is completely unchanged and untouched by the new mode's
        existence.
        """
        if self.global_selection:
            return self._propose_global()

        if self.exhausted:
            return None
        self.proposals += 1

        node, chain = self.root, [self.root]
        expanded = False
        # selection: descend by UCB while progressive widening blocks a new child here
        while not self._can_expand(node) and node.children:
            # epsilon-greedy override (see __init__ docstring comment): with
            # probability epsilon, force-expand a random untried edit here
            # instead of descending into an existing child via UCB. `self.epsilon
            # > 0` short-circuits before touching the RNG so epsilon=0.0 (default)
            # never consumes a random draw -> bit-identical to before this existed.
            if self.epsilon > 0 and node.untried and self.rng.random() < self.epsilon:
                node = self._expand(node, node.untried.pop())
                chain.append(node)
                expanded = True
                break
            node = max(node.children.values(), key=lambda ch: self._ucb(node, ch))
            chain.append(node)
        # expansion: place one new edit if widening allows (else re-propose a leaf's program)
        if not expanded and self._can_expand(node):
            node = self._expand(node, node.untried.pop())
            chain.append(node)

        program = self._build_program(node.edits)
        if not is_valid(program):  # degenerate all-skip cover -> empty path
            self.update(chain, 0.0, program=None)
            return self.propose() if not self.exhausted else None
        return program, chain

    def update(self, chain: List[Node], reward: float, program: Optional[Program]):
        for node in chain:
            node.visits += 1
            node.total_reward += reward
        if program is not None:
            self.evaluated[tuple(program.to_layer_path())] = reward
        parent = chain[-2] if len(chain) > 1 else None
        # program=None means the degenerate all-skip cover (module docstring,
        # "The is_valid guard"): to_layer_path() RAISES on it by design
        # (Program.to_layer_path, "skips all layers"), so represent it as []
        # directly rather than rebuilding+calling it. A PARENT further up the
        # chain can never itself be this degenerate program -- every node
        # already in `chain` passed is_valid() when IT was first proposed, and
        # a node's edits never change afterward.
        self.trajectory.append({
            "path": program.to_layer_path() if program is not None else [],
            "parent_path": (self._build_program(parent.edits).to_layer_path()
                            if parent is not None else None),
            "reward": reward,
        })
        # the paper's literal "V" (_global_ucb / ucb_global_v): one simulation
        # anywhere in the tree, root's bootstrap included. Tracked regardless
        # of mode (cheap; see __init__) -- only global_selection's
        # _global_ucb and _ucb's ucb_global_v branch ever read it.
        self._global_visits_total += 1

    def valid_paths(self) -> List[List[int]]:
        """All evaluated programs with reward 1, shortest first (label preference)."""
        return [list(p) for p, r in sorted(self.evaluated.items(), key=lambda kv: len(kv[0]))
                if r >= 1.0]

    def invalid_paths(self) -> List[List[int]]:
        return [list(p) for p, r in self.evaluated.items() if r < 1.0]
