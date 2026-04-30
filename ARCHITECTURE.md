# Itty Bitty Architecture

Itty Bitty is a bit-native runtime for binary model representations and
bitwise execution. The architecture in this repository is the runtime in
`src/` plus the standalone experiments in `experiments/`.

## Repository Shape

The repository is split into three layers:

- `src/`: current runtime, CLI, and default tests.
- `experiments/`: standalone experiments outside the runtime in `src/`.

Experimental code can influence the runtime, but the runtime surface in `src/`
stays separate from the experiment code in `experiments/`.

## Runtime

The runtime is the bit-string execution core:

- `itty-bit-string`: word-backed bit storage and bitwise primitives.
- `itty-bit-string-list`: collections, transpose, condense, weighted condense,
  and vote allocation helpers.
- `itty-bit-string-map`: mmap-backed storage for serialized bit strings.
- `itty-vocabulary`: text/bit-string mapping and context encoding.
- `itty-network`: feed nodes and similarity-matching nodes over bit-string
  lists.
- `itty-inference`: output selection and vocabulary decoding.
- `itty-exec-buffer`: staged execution buffers for bitwise plans.
- `itty-manager`, `itty-work-queue`, `itty-pipeline`: concurrency and task
  execution scaffolding.
- `itty-position`: locality and Gray-code helpers.
- `itty-model-metrics`: runtime-oriented density and activation metrics.

These modules form the runtime architecture.

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
- similarity-matching nodes

This topology is list-to-list rather than scalar-to-scalar. A layer does not
produce one number. It produces a new list of bit strings, and later layers
operate on that list.

The experiments use a different topology. Instead of passing a list of
bit strings through a stack of layers, they keep a collection of stored memory
entries and score each entry against the input bit string. The winning entry is
the one whose stored comparison bits match the input most strongly. After that
selection step, the experiments check whether that same entry can decode its
stored target cleanly and whether its committed quality state remains valid.

This experiment topology is mainly about training questions, not about making
inference itself more complicated. At inference time, the mechanics are simple:
compare the input against the stored entries, choose the best match, and decode
that entry's stored target. The harder questions are about training:

- how to decide when to create a new stored entry
- how to choose the stored comparison bits for that entry
- how to choose the target and payload metadata for that entry
- how to update one entry without damaging the entries that were already working

The reason for that change is that the experiments are testing ownership,
selection, decoding, and post-update rechecking directly. A layered network describes how bit
patterns move forward through nodes. The experiments described here test:

- can one example claim a stored memory entry?
- can that claim stay stable when more examples are added?
- can the chosen entry decode its own target cleanly?
- can later updates to one entry's stored bits or committed distance improve
  that entry without silently damaging the others?

These questions are easier to study when the memory entries are explicit. The
experiments therefore store a set of candidate entries and measure which one
wins, how far ahead it is from the next-best match, and whether its stored decode state remains
valid over time.

The experiment code calls that collection a route table. In plain terms, it is
just a list of stored entries. Each stored entry holds the bits associated with
one candidate owner. In these experiments, each stored entry holds:

- a stored comparison bit string, which is compared against the input bit string
  when choosing an entry
- a decoder target, which is the bit pattern that entry expects to recover
- payload metadata such as payload bit count
- commit state such as the best accepted distance for that entry

The current experiments populate these entries directly in code. The tests
construct small bit strings explicitly and append them to the table one entry
at a time.

In the entry-selection experiment, the stored comparison bits are supplied
directly. In the combined experiment, each appended entry gets:

- comparison bits copied from a hand-constructed input pattern
- a decoder target copied from a hand-constructed payload pattern
- a payload bit count supplied explicitly
- an initial committed distance left unset until a successful commit

The topology question in these experiments is not "which layer comes next?" It
is "which stored entry should own this input bit string?"

- an input bit string addresses a route table
- one stored entry is selected by comparing the input bit string against each stored
  comparison bit string
- the selected entry evaluates its own stored decoder target
- commit updates quality state for that selected entry only

This topology is closer to an associative memory table than to a
stack of dense arithmetic layers.

The runtime remains layered. The experiments isolate ownership, decoding, and
post-update checks so those rules can be stated directly and then reintroduced into the
runtime as runtime nodes or layers with explicit roles:

- a selection stage that chooses which stored memory entry should answer a
  given input bit string
- a decode stage that reads that entry's stored target state
- a commit-check stage that checks whether updates preserve prior quality

The experiments pull ownership, decoding, and post-update checks out of the full layer
stack so each one can be tested directly. The runtime can then express those
same rules inside a layered network.

## Core Semantics

### Bit String

A bit string is a fixed-width container backed by `size_t` words. It is the
fundamental value type in the runtime.

### Length

Length is storage width in bits.

- `itty_bit_string_get_length()`: storage width in bits.

A zero-valued one-word bit string is a 64-bit container on a 64-bit machine.

### Feed Node

A feed node is the basic transformation step in the runtime network.

It works in two stages:

- first it applies a set of bit masks to its input bit strings
- then it combines the masked results into one output bit string by voting bit
  position by bit position

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

### Similarity-Matching Node

An affinity node is the runtime's similarity-matching node. It compares an
input bit string against stored reference bit strings using popcount-style
similarity, optionally biased by locality and Gray-position similarity, then
combines the output patterns attached to the strongest matches.

Its stored state is:

- reference bit strings, called `traits` in the code
- output bit strings attached to those references, called `imprints` in the code
- scoring options: score-length, locality, and Gray-position scoring parameters

### Bitwise Voting

The code uses the word `condense` for turning a list of bit strings into one
bit string by voting at each bit position.

Current condense variants:

- majority condense: set a bit if a strict majority of inputs set it.
- weighted condense: first assign each input bit string a vote count. Then,
  for each bit position, inspect every bit string in the list. If a bit string
  has a `1` at that position, that counts as a vote for setting the output bit
  at that same position. In the weighted form, that vote is not just one yes:
  it contributes the full vote count assigned to that input bit string. If a
  bit string has a `0` at that position, it contributes nothing at that
  position. The output bit is set when the total votes in favor of `1` cross a
  strict majority of all vote counts combined.

Both are threshold-based voting strategies. The difference is whether every input
gets one vote or whether different inputs can contribute different vote
weights.

### Segment Voting Decoder

The code uses the name `segment-condense` for an experimental decoder family in
which the output is interpreted as groups of votes per target bit rather than
as one strict all-bits-must-agree fold. It treats decoding as vote accumulation
across segments.

## Weights And Stored State

This project uses bit-native stored state rather than float matrices.

### Runtime Weights

In the runtime, "weights" means the bit patterns that determine how a node
transforms or selects data.

For feed nodes, the weights are modulation masks.

- each mask is a bit string
- a feed node owns a list of masks
- those masks gate or reshape its input before condense

For similarity-matching nodes, the weights are the stored reference patterns
and output patterns plus the
scoring options used to score them.

- the reference patterns, called `traits` in the code, determine which stored
  entry matches an input bit string
- the output patterns, called `imprints` in the code, determine what content
  that match contributes to the output
- options determine how content, locality, and Gray position contribute to the
  score

### Experimental Weights

In the experiments, the main stored state is:

- stored comparison bit strings in the entry-selection experiment
- payload targets in the self-describing Gray payload experiment
- stored comparison bit strings, payload targets, payload bit counts, and
  committed distances in the combined ownership-plus-Gray-payload experiment

These are weights in the broad sense: they are the persistent bit
patterns and thresholds that define what the system recognizes and emits.

### Where Weights Come From

The runtime can load model state from files and execute it through the CLI.
The experiments build their stored state directly inside tests.

At present:

- runtime execution assumes the node state already exists
- experiment tests construct comparison bit strings, payloads, and targets
  explicitly
- commit records quality state for the selected entry after measurement

More concretely, the current tests populate the stored-entry table by calling
the append functions directly. There is no training procedure yet that:

- decides when to create a new stored entry
- derives the comparison bits from data
- derives the decoder target from data
- fills the payload metadata automatically

There is no general-purpose trainer yet that derives all of these weights
automatically from arbitrary datasets. The experiments are isolating the
subproblems that such a trainer would need to solve.

## Selection And Decoding

### Ownership

Ownership means that a given example is expected to map to a particular stored
entry,
and that later updates should preserve that mapping unless the design
explicitly reallocates it. The important property is stability: new examples
should not silently steal an existing owner's stored entry.

Each stored entry carries comparison bits, which are the bit string used for
addressing. Those comparison bits give the experiment a separate addressing
layer:

- comparison bits: which stored entry should own this input?
- decoder state: how should that stored entry reconstruct its target?

### Score Difference

The code uses the name `route gap` for the difference between the selected
entry's similarity score and the runner-up score. A positive difference means
the decision is not tied. Larger differences indicate stronger ownership
separation.

### Stored Entry Table

A route table is the structure that stores one entry per candidate owner. In the combined
prototype that metadata currently includes:

- comparison bit string
- target bit pattern
- payload bit count
- committed distance

This table is the experimental topology for memory-like ownership.
One input bit string selects one entry, and only that entry's decoder target is
measured.

### Post-Update Checks

Post-update checks mean re-measuring entries that were already claimed after a
new update has been applied elsewhere, in order to verify that the new update
did not damage them.

The architectural point is that these checks are how ownership stability is
measured over time.

### Update Safety

Update safety means a candidate update does not make an already-claimed entry
perform worse than the best accepted result that has already been recorded for
that entry.

### Committed Baseline

A committed baseline is the best accepted result that has already been recorded
for a stored entry. In the current experiments, that is the committed distance
stored for that entry after a successful commit. Later checks compare new
measurements against that stored distance.

The committed baseline is important because it turns later verification from
"try things and hope old owners survive" into a concrete acceptance rule.

### Decoder

In this project, a decoder is the mechanism that interprets the selected
entry's activation bits as a target bit string. An entry may have stable
ownership but
still fail because its decoder cannot recover the target cleanly.

When this document talks about decoder distance, it means the number of bit
positions where the produced bits differ from the expected target bits. A
distance of `0` means an exact match. Larger distances mean more bit
mismatches.

### Gray Code

Gray code is a binary encoding where adjacent integer values differ by one bit.
The runtime already exposes Gray helpers in `itty-position`.

### Gray-Coded Start Position

The code uses the phrase `Gray offset` for the Gray-coded payload start stored in the self-describing
header. Conceptually:

- the activation is larger than the target because the network widens the bit
  width as data moves forward through layers. In the current runtime, feed
  nodes double their output width, so later activations have more bit positions
  available than the payload target itself.
- that extra space then makes it possible to store more than just one payload:
  it leaves room for a fixed header region, for a payload window at a chosen
  start position, and for other non-overlapping payload regions elsewhere in
  the same activation.
- the target only describes the payload bits that the decoder is supposed to
  recover.
- the payload bits therefore live in a window inside that larger activation.
- the start of that window is encoded as a Gray-coded offset in activation bits
  `0..7`.
- the decoder reads that fixed header, decodes `payload_start`, and then reads
  the payload window from that decoded location.

This makes the decoder care both about content and about where that content was
placed inside the activation.

In the current training direction, that start position is hidden state rather
than a supervised target. The decoder uses it to find the payload window. The
trainer does not try to force one specific start position. It only requires the
header to decode to some valid position and then compares the extracted payload
against the expected payload.

The `0..7` header range is a storage layout choice, not a voting threshold.
The current self-describing Gray payload experiments read all 8 header bits,
decode them as one Gray-coded start position, and then measure exact payload
bit mismatches.

### Payload Start Position

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

### Bundled Updates

A bundle is an experimental batch of related mask flips or bit changes that are
applied together because they represent one coherent repair idea.

### Threshold-Crossing Bundled Updates

A threshold-crossing bundled update is a bundle chosen to move vote counts across a
decoder threshold, typically to convert a target-zero or target-one bit from
"almost correct" to "correct".

Threshold-crossing bundled updates belong to thresholded vote decoders such as
bitwise voting and segment voting. The self-describing Gray payload experiments are not
currently driven by that kind of bundle logic. They mainly measure:

- whether the selected entry stays ahead of the runner-up
- whether the fixed Gray header decodes a valid payload start
- how many payload bits disagree with the target

## Current Experiments

### Entry Selection

The entry-selection experiment is the smallest prototype of explicit
associative ownership. It proves:

- stored entries can be selected directly from stored comparison bits,
- input bit strings can select the intended owner entry,
- positive score differences can survive as entries are added,
- ownership can remain stable across an A/B/C/D growth pattern.

This entry-selection test is intentionally narrow. It is testing only
addressing behavior:

- whether the selected entry is the expected owner
- whether the selected score beats the runner-up score
- whether the score difference stays positive as the table grows

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

The important point is that these tests are checking whether the payload can be
recovered from whatever valid start position the activation encodes. They are
not treating the start position itself as a label that must match a separately
supervised target.

This is a self-describing decoder experiment. It does not decide which stored
entry should own an input. It does not decide
ownership. It assumes the caller already knows which target it should compare
against.

### Ownership + Gray Payload

The combined ownership-plus-Gray-payload experiment is the current integrated
direction:

- comparison-bit matching provides ownership,
- the selected entry reads `payload_start` from the activation header,
- self-describing Gray payload scoring measures decoder distance for that entry,
- commit updates the selected entry's committed floor without changing
  ownership.

This experiment is testing the interaction between three concerns:

- stored-entry selection
- per-entry decoding
- per-entry commit state

Its current tests check:

- A/B/C/D still select the intended stored entries as the table grows
- the selected entry stays ahead of the runner-up
- the selected entry can decode a valid `payload_start` from the fixed header
- the measured distance is zero for the clean A/B/C/D examples
- commit records a distance floor for the selected entry
- later damage raises measured distance while preserving the previously
  committed floor

Here too, the decoded start position is treated as internal state. The training
question is whether the selected activation yields a valid extraction and the
right payload bits, not whether it uses one predetermined position.

This prototype exercises ownership and decoding in one path.

The experiments decompose into:

- comparison bits: ownership and addressing
- self-describing Gray payload: decoding and alignment within the selected entry
- post-update checks: preservation of committed owners over time

## Current Decomposition

The experiments separate routing and decoding:

- comparison-bit matching determines ownership,
- the stored-entry table stores one entry per candidate owner,
- decoder quality is measured within the owned entry,
- post-update checks determine whether committed owners remain stable.

## What The Experiments Are Testing

The experiments are decomposing training into smaller questions.

### Ownership Question

Can the system allocate stable owners in bit space?

The entry-selection experiment answers that by measuring whether the same
input bit string continues to select the same stored entry with a positive
score difference as other entries are added.

### Decoder Question

Can one stored entry represent both payload content and payload location in a way that
can be decoded directly from the activation?

The self-describing Gray payload experiment answers that by fixing the header
position and checking whether the decoded start and payload content agree.

### Combined Question

Can stable ownership and per-entry decoding coexist without conflating the
two problems?

The combined experiment answers that by selecting one stored entry with its
stored comparison bits, measuring only that entry's self-describing payload
distance, and tracking a committed distance floor per entry.

That means the training objective is payload recovery, plus the requirement
that the chosen start position be valid. The chosen position itself is not a
supervised output.

## Toward A General-Purpose Trainer

The codebase has an execution runtime and three experimental components. A
general-purpose trainer would need to connect those pieces into a
full learning loop.

### What Exists Now

- a layered bit-string runtime
- feed and similarity-matching execution nodes
- comparison-based ownership selection
- a self-describing payload encoding
- per-entry measurement and commit bookkeeping

### What Is Still Missing

#### Topology Growth

The project needs a policy for when to allocate a new stored entry, when to
reuse an existing one, and how stored-entry tables connect to larger network
topologies.

#### Weight Update Rules

The project needs update rules that can derive useful modulation masks,
reference patterns, output patterns, comparison bit strings, and payload activations from examples rather than
from hand-constructed tests.

#### Decoder Writing

The self-describing payload experiment can measure a payload and decode its
start, but a general trainer needs a write rule that decides:

- where payload bits should be placed in the activation
- how the header should be set
- how that write should be repaired when the distance is non-zero

That write rule does not need to aim for one fixed position. It only needs to
produce an activation whose header points to a valid payload window and whose
extracted payload matches the training target.

#### Post-Update Check Policy

The combined prototype tracks committed distance per stored entry. A general trainer
needs a scheduler for re-checking committed entries that decides:

- which previously claimed owners must be revisited
- which candidate updates are allowed
- how much degradation, if any, is acceptable
- when to promote a better measured distance into the committed floor

#### Multi-Layer Credit Assignment

The runtime supports layered execution, but the experiments do not define a
general way to assign credit across several trainable stages. A complete
trainer needs a bit-native update path that can move useful information through
more than one ownership or decoder layer.

#### Dataset And Objective Surface

The experiments are small, explicit test scenarios. A general trainer needs:

- a dataset loop
- train/validation splits
- stable objective reporting
- stopping criteria
- regression suites for ownership, decoding, and post-update behavior

#### Model Serialization

The runtime can execute model state from files, but the experiments do
not define one unified on-disk format for stored-entry tables, decoder state,
commit-check state, and layer topology.

### Practical Next Step

The most direct path from the current code to a broader trainer is:

1. define a trainable stored-entry-table format with explicit per-entry state
2. define one write-and-repair rule for self-describing payload activations
3. define one re-check scheduler with commit acceptance rules
4. connect that loop to a small benchmark dataset
5. expand from one stored-entry table to a multi-layer training topology only
   after the single-table trainer is stable

## Build Surface

Meson exposes one experiment toggle:

- `experimental_prototypes=true`: build the small standalone
  entry-selection, self-describing-Gray-payload, and combined prototypes.

## Notes

The architecture described here lives in this document and in the code under
`src/` and `experiments/`.
