# Itty Bitty Architecture

Itty Bitty is a bit-native runtime for experimenting with binary model
representations and bitwise execution. The current architecture is the small
runtime in `src/` plus a narrow set of standalone experiments in
`experiments/`.

## Repository Shape

The repository is split into three layers:

- `src/`: current runtime, CLI, and default tests.
- `experiments/`: active and archived experiments that are intentionally
  outside the default runtime story.
- `notes/`: long-form lab notes, dead ends, and historical design narratives.

That boundary is deliberate. Experimental code can influence the architecture,
but it should not become part of the runtime surface until it has been reduced
to a small API and focused tests.

## Runtime

The keepable runtime is the bit-string execution core:

- `itty-bit-string`: word-backed bit storage and bitwise primitives.
- `itty-bit-string-list`: collections, transpose, condense, weighted condense,
  and vote allocation helpers.
- `itty-bit-string-map`: mmap-backed storage for serialized bit strings.
- `itty-vocabulary`: text/bit-string mapping and context encoding.
- `itty-network`: feed nodes and affinity nodes over bit-string lists.
- `itty-inference`: output selection and vocabulary decoding.
- `itty-exec-buffer`: staged execution buffers for bitwise plans.
- `itty-manager`, `itty-work-queue`, `itty-pipeline`: concurrency and task
  execution scaffolding.
- `itty-position`: locality and Gray-code helpers.
- `itty-model-metrics`: runtime-oriented density and activation metrics.

These modules form the current runtime architecture.

## Topology

The runtime topology is a layered network over bit-string lists.

- an input is an `itty_bit_string_list_t`
- a network is an ordered list of layers
- a layer is an ordered list of nodes
- a node reads the current input list and produces one or more output bit
  strings
- a layer output becomes the input list for the next layer

The current runtime supports two node families:

- feed nodes
- affinity nodes

This topology is list-to-list rather than scalar-to-scalar. A layer does not
produce one number. It produces a new list of bit strings, and later layers
operate on that list.

The active experiments use a different topology. Instead of passing a list of
bit strings through a stack of layers, they keep a collection of stored memory
entries and ask which entry should answer a probe. Here, a probe is the input
bit string being used to look up the best matching stored entry.

The reason for that change is that the experiments are trying to answer a
different question from the runtime. A layered network is good for studying how
bit patterns transform as they move forward through nodes. The current
experiments are focused on a different problem:

- can one example claim a stored memory entry?
- can that claim stay stable when more examples are added?
- can the chosen entry decode its own target cleanly?
- can later updates to one entry's stored bits or committed distance improve
  that entry without silently damaging the others?

Those questions are easier to study when ownership is explicit. The experiments
therefore make the memory entries explicit too. Instead of hiding ownership
inside several transformation stages, they store a set of candidate entries and
measure which one wins, how strong the margin is, and whether its stored decode
state remains valid over time.

That collection is called a route table. The table is just a list of route
entries. Each route entry stores the bits associated with one candidate owner.
In the current experiments, a route entry stores:

- a route key, which is the bit string used to compare the entry against a
  probe
- a decoder target, which is the bit pattern that entry expects to recover
- payload metadata such as payload bit count
- commit state such as the best accepted distance for that entry

When the document says "route", it means one of these stored entries. The
topology question is not "which layer comes next?" It is "which stored entry
should own this probe?"

- a probe bit string addresses a route table
- one route is selected by comparing the probe against each route key
- the selected route evaluates its own stored decoder target
- commit updates quality state for that selected route only

That experimental topology is closer to an associative memory table than to a
stack of dense arithmetic layers.

That does not mean the project is moving away from the runtime's layered
architecture. The point of the experiments is to separate one hard part of the
design, make it legible, and then bring the result back into the runtime in a
smaller form.

If the route-table experiments hold up, they would feed back into the main
architecture as one or more runtime nodes or layers with explicit roles:

- a selection stage that chooses which stored memory entry should answer a
  probe
- a decode stage that reads that entry's stored target state
- a replay or commit stage that checks whether updates preserve prior quality

In other words, the experiments are flattening the problem on purpose. They
pull ownership, decoding, and replay out of the full layer stack so each one
can be tested directly. Once those rules are stable, they can be reintroduced
as ordinary parts of a layered network instead of remaining a separate
experimental world.

## Core Semantics

### Bit String

A bit string is a fixed-width container backed by `size_t` words. It is the
fundamental value type in the runtime.

### Length

Length is storage width in bits.

- `itty_bit_string_get_length()`: storage width in bits.

A zero-valued one-word bit string is a 64-bit container on a 64-bit machine.

### Feed Node

A feed node applies modulation masks to its inputs, then condenses the result
into a bit string by vote-like aggregation. It is the basic transformation node
in the runtime network.

The feed node's trainable state is its list of modulation masks. Each mask is a
bit string with the same length as the input width that node expects.

After condensing, a feed node doubles its output width by concatenating two
halves derived from the condensed bit string. The basic form repeats the same
condensed output in both halves. The rotated form keeps the first half as-is
and rotates the second half by a configured bit offset before concatenation.

This doubling-and-rotation step is one of the ways the runtime introduces
structure beyond a plain linear vote. In practice it plays some of the role an
activation function would play in a more conventional network: it changes how
information is presented to the next layer instead of merely forwarding one
condensed bit string unchanged.

In the current code, the rotation amount is node configuration, not learned
state. The trainable state of a feed node is still its list of modulation
masks. Rotation is a fixed parameter attached to the node when the node is
constructed.

### Affinity Node

An affinity node is a bit-native content-addressing primitive. It scores probe
bit strings against stored trait bit strings using popcount-style similarity,
optionally biased by locality and Gray-position similarity, then condenses the
selected imprints into an output.

The affinity node's stored state is:

- traits: the matchable reference bit strings
- imprints: the bit strings emitted or voted when a trait wins
- probe options: score-length, locality, and Gray-position scoring parameters

### Condense

To condense is to turn a list of bit strings into one bit string by voting per
bit position.

Current condense variants:

- majority condense: set a bit if a strict majority of inputs set it.
- weighted condense: assign each input a vote budget, then set a bit if the
  weighted one-votes cross a strict majority of the total vote budget.

Both are threshold condense strategies. The difference is whether every input
gets one vote or whether different inputs can contribute different vote
weights.

### Segment-Condense

Segment-condense is an experimental decoder family where the output is
interpreted as groups of votes per target bit rather than as one strict
all-bits-must-agree fold. It treats decoding as vote accumulation across
segments.

## Weights And Stored State

This project uses bit-native stored state rather than float matrices.

### Runtime Weights

In the runtime, "weights" means the bit patterns that determine how a node
transforms or selects data.

For feed nodes, the weights are modulation masks.

- each mask is a bit string
- a feed node owns a list of masks
- those masks gate or reshape its input before condense

For affinity nodes, the weights are the stored traits and imprints plus the
probe options used to score them.

- traits determine which stored entry matches a probe
- imprints determine what content that match contributes to the output
- options determine how content, locality, and Gray position contribute to the
  score

### Experimental Weights

In the experiments, the main stored state is:

- route keys in the route-key selector
- payload targets in the self-describing Gray payload experiment
- route keys, payload targets, payload bit counts, and committed distances in
  the combined route-key plus Gray payload experiment

These are still weights in the broad sense: they are the persistent bit
patterns and thresholds that define what the system recognizes and emits.

### Where Weights Come From

The runtime can load model state from files and execute it through the CLI.
The experiments build their stored state directly inside focused tests.

At the current stage:

- runtime execution assumes the node state already exists
- experiment tests construct route keys, payloads, and targets explicitly
- commit records quality state for the selected route after measurement

There is not yet one general-purpose trainer that derives all of these weights
automatically from arbitrary datasets. The experiments are isolating the
subproblems that such a trainer would need to solve.

## Ownership And Decoding

### Ownership

Ownership means that a given example is expected to map to a particular route,
and that later updates should preserve that mapping unless the design
explicitly reallocates it. The important property is stability: new examples
should not silently steal an existing owner's route.

Each route stores a route key, which is the bit string used for addressing.
The route key gives the experiment a separate addressing layer:

- route key: who should own this input?
- decoder state: how should that route reconstruct its target?

### Route Gap

The route gap is the margin between the selected route's similarity score and
the runner-up score. A positive route gap means the route decision is not tied.
Larger gaps indicate stronger ownership separation.

### Route Table

A route table is the structure that stores one entry per route. In the combined
prototype that metadata currently includes:

- route key
- target bit pattern
- payload bit count
- committed distance

The route table is the current experimental topology for memory-like ownership.
One probe selects one route, and only that route's decoder target is measured.

### Replay

Replay means reapplying training or repair to examples that were already
claimed, in order to verify that newly added work does not regress them.

The architectural point is that replay is how ownership stability is checked
over time.

### Replay Safety

Replay safety means a candidate update does not push already-claimed owners
below their allowed baseline.

### Committed Baseline

A committed baseline is the current minimum quality floor that a route owner is
allowed to fall below during replay. In practice this is a stored objective
state such as distance that replay must respect for that route.

The committed baseline is important because it turns replay from "try things and
hope old owners survive" into a concrete acceptance rule.

### Decoder

In this project, a decoder is the mechanism that interprets the selected route's
activation bits as a target bit string. A route may have stable ownership but
still fail because its decoder cannot recover the target cleanly.

### Gray Code

Gray code is a binary encoding where adjacent integer values differ by one bit.
The runtime already exposes Gray helpers in `itty-position`.

### Gray Offset

The Gray offset is the Gray-coded payload start stored in the self-describing
header. Conceptually:

- payload bits live in a window inside a larger activation.
- the start of that window is encoded as a Gray-coded offset in activation bits
  `0..7`.
- the decoder reads that fixed header, decodes `payload_start`, and then reads
  the payload window from that decoded location.

This makes the decoder care both about content and about where that content was
placed inside the activation.

The `0..7` header range is a storage layout choice, not a voting threshold.
The current self-describing Gray payload experiments read all 8 header bits,
decode them as one Gray-coded start position, and then measure exact payload
bit mismatches. They do not use a rule like "5 of 8 header bits must agree".

### Payload Start

The payload start is the bit offset inside an activation where the payload
window begins.

### Self-Describing Header

A self-describing header is a fixed activation region that stores metadata at a
known location. In the self-describing Gray payload prototype, activation bits
`0..7` store `gray(payload_start)`.

### Retuning

Retuning means adjusting the stored or emitted activation so that the fixed
self-describing header and the payload window remain coherent. In a
self-describing payload design, retuning is a decoder-side repair process that
must preserve:

- a valid Gray-coded header
- a valid decoded `payload_start`
- payload bits at the location the header names

### Bundle

A bundle is an experimental batch of related mask flips or bit changes that are
applied together because they represent one coherent repair idea.

### Threshold-Crossing Bundle

A threshold-crossing bundle is a bundle chosen to move vote counts across a
decoder threshold, typically to convert a target-zero or target-one bit from
"almost correct" to "correct".

This terminology belongs to the archived feed-model lab and to thresholded vote
decoders such as condense and segment-condense. The promoted self-describing
Gray payload experiments are not currently driven by that kind of threshold
bundle logic. They mainly measure:

- whether the route selection margin stays positive
- whether the fixed Gray header decodes a valid payload start
- how many payload bits disagree with the target

## Current Experiments

### Feed-Model Lab Archive

`experiments/feed-model-lab/` is an archived experimental tree that is kept out
of the default runtime surface.

It is built only behind `-Dexperimental_feed_model=true`.

### Route-Key Selector

`experiments/route-key-selector/` is the smallest prototype of explicit
associative ownership. It proves:

- routes can be keyed directly,
- probes can select the intended owner route,
- positive route gaps can survive as routes are added,
- ownership can remain stable across an A/B/C/D growth pattern.

The route-key selector test is intentionally narrow. It is testing only
addressing behavior:

- whether the selected route is the expected owner
- whether the selected score beats the runner-up score
- whether the selected gap stays positive as the table grows

It is not testing payload decoding, decoder repair, or commit policy.

### Self-Describing Gray Payload

`experiments/self-describing-gray-payload/` uses a fixed activation header to
store `gray(payload_start)` at a known location. The payload still lives at
`payload_start`, but the start offset is read directly from the activation
header.

This experiment is testing the decoder representation itself.

Its current tests check:

- clean examples A/B/C/D decode the intended `payload_start`
- the decoded header is marked valid when the payload window fits inside the
  activation
- `measured_distance` is zero when the payload matches the target
- payload corruption raises distance without corrupting the decoded start
- header corruption is detected as an invalid header

This is a self-describing decoder experiment. It does not decide route
ownership. It assumes the caller already knows which target it should compare
against.

### Route-Key + Gray Payload

`experiments/route-key-gray-payload/` is the combined direction:

- route key selection provides ownership,
- the selected route reads `payload_start` from the activation header,
- self-describing Gray payload scoring measures decoder distance for that route,
- commit updates the selected route's committed floor without changing route
  ownership.

This experiment is testing the interaction between three concerns:

- route selection
- per-route decoding
- per-route commit state

Its current tests check:

- A/B/C/D still select the intended routes as the table grows
- the selected route keeps a positive route gap
- the selected route can decode a valid `payload_start` from the fixed header
- the measured distance is zero for the clean A/B/C/D examples
- commit records a distance floor for the selected route
- later damage raises measured distance while preserving the previously
  committed floor

This is the smallest current prototype that exercises ownership and decoding in
one path.

This is the current decomposition of the promoted ideas:

- route key: ownership and addressing
- self-describing Gray payload: decoding and alignment within the selected route
- replay: preservation of committed owners over time

## Current Decomposition

The current promoted path separates routing and decoding:

- route key determines ownership,
- route table stores one route entry at a time,
- decoder quality is measured within the owned route,
- replay checks whether committed owners remain stable.

## What The Experiments Are Testing

The experiments are decomposing training into smaller questions.

### Route-Key Question

Can the system allocate stable owners in bit space?

The route-key selector answers that by measuring whether a probe continues to
select the same route with a positive gap as other routes are added.

### Decoder Question

Can one route represent both payload content and payload location in a way that
can be decoded directly from the activation?

The self-describing Gray payload experiment answers that by fixing the header
position and checking whether the decoded start and payload content agree.

### Combined Question

Can stable ownership and per-route decoding coexist without conflating the
two problems?

The combined experiment answers that by selecting one route with the route key,
measuring only that route's self-describing payload distance, and tracking a
committed distance floor per route.

## Toward A General-Purpose Trainer

The codebase now has an execution runtime and three focused experimental
components. A general-purpose trainer would need to connect those pieces into a
full learning loop.

### What Exists Now

- a layered bit-string runtime
- feed and affinity execution nodes
- route-key ownership selection
- a self-describing payload encoding
- per-route measurement and commit bookkeeping

### What Is Still Missing

#### Topology Growth

The project still needs a policy for when to allocate a new route, when to
reuse an existing one, and how route tables connect to larger network
topologies.

#### Weight Update Rules

The project still needs update rules that can derive useful modulation masks,
traits, imprints, route keys, and payload activations from examples rather than
from hand-constructed tests.

#### Decoder Writing

The self-describing payload experiment can measure a payload and decode its
start, but a general trainer still needs a write rule that decides:

- where payload bits should be placed in the activation
- how the header should be set
- how that write should be repaired when the distance is non-zero

#### Replay Policy

The combined prototype tracks committed distance per route. A general trainer
still needs a replay scheduler that decides:

- which previously claimed owners must be revisited
- which candidate updates are allowed
- how much degradation, if any, is acceptable
- when to promote a better measured distance into the committed floor

#### Multi-Layer Credit Assignment

The runtime supports layered execution, but the experiments do not yet define a
general way to assign credit across several trainable stages. A complete
trainer needs a bit-native update path that can move useful information through
more than one ownership or decoder layer.

#### Dataset And Objective Surface

The current experiments are small, explicit test scenarios. A general trainer
still needs:

- a dataset loop
- train/validation splits
- stable objective reporting
- stopping criteria
- regression suites for ownership, decoding, and replay behavior

#### Model Serialization

The runtime can execute model state from files, but the promoted experiments do
not yet define one unified on-disk format for route tables, decoder state,
replay state, and layer topology.

### Practical Next Step

The most direct path from the current code to a broader trainer is:

1. define a trainable route-table format with explicit per-route state
2. define one write-and-repair rule for self-describing payload activations
3. define one replay scheduler with commit acceptance rules
4. connect that loop to a small benchmark dataset
5. expand from one route table to a multi-layer training topology only after
   the single-table trainer is stable

## Build Surface

Meson exposes two experiment toggles:

- `experimental_feed_model=false`: build the archived feed-model lab only when
  explicitly requested.
- `experimental_prototypes=true`: build the small standalone route-key,
  Gray-payload, and combined prototypes.

The default runtime should remain usable even if the feed-model archive is
ignored entirely.

## Notes

Long-form experiment notes live under `notes/`.
