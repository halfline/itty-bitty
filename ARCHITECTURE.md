# Itty Bitty Architecture

Itty Bitty is an experimental C runtime for exploring neural-network-like
computation over bit strings instead of floating-point tensors. The project is
currently a prototype: the bit-string runtime is the most concrete layer, while
the vocabulary, model loading, network feed, and pipeline layers are still
early scaffolding.

This document describes the architecture as it exists today and outlines
plausible next steps that preserve the project's experimental nature.

## Goals

The central idea is to represent tokens, model parameters, intermediate state,
and outputs as arrays of machine words. Network operations then use cheap
bitwise primitives such as XOR, XNOR, AND, OR, popcount, transposition, and
majority voting.

The intended direction is:

- map text tokens to bit strings,
- load model masks as bit strings,
- feed token bit strings through layers of bitwise nodes,
- select outputs using popcount-based ranking,
- eventually add training and richer block types.

The current code does not implement training. It also does not yet define a
complete model format.

## Build Layout

The project uses Meson and builds:

- `libittybitty`: the runtime library,
- `itty-bitty`: the command-line demo,
- one executable per test source under `src/tests`.

The library sources are listed directly in `meson.build`. There is no generated
code, dependency discovery, or installed public header layout yet.

## Major Components

### Bit Strings

Files:

- `src/itty-bit-string.h`
- `src/itty-bit-string-private.h`
- `src/itty-bit-string.c`

`itty_bit_string_t` is the core data type. It stores:

- `words`: a `size_t *` array,
- `number_of_words`: allocated word count,
- cached `pop_count`,
- cached `bit_length`,
- a mutability mode.

The mutability modes are:

- `READ_WRITE`: owns and frees `words`,
- `READ_ONLY`: borrows `words`,
- `COPY_ON_WRITE`: initially borrows `words`, then copies on append.

This is used heavily by mmap-backed bit strings: mapped model/vocabulary words
can be referenced without copying, but can become owned if appended to later.

Current primitive operations include:

- `exclusive_nor`: XNOR, currently implemented as `~(a ^ b)`,
- `exclusive_or`: XOR,
- `combine`: OR,
- `mask`: AND,
- `get_pop_count`: cached popcount,
- `get_length`: cached significant-bit length,
- `get_bit_capacity`: allocated word capacity expressed in bits,
- `evaluate_similarity`: popcount of XNOR,
- `split`,
- `concatenate`,
- `double`,
- `double_with_rotated_half`,
- `reduce_by_half`,
- `reduce_rotated_by_half`,
- `present`.

`double_with_rotated_half` emits `[X][rotl(X, n)]`, rotating the second half
over the bit string's allocated bit capacity. When followed by the current
AND-based `reduce_by_half`, this behaves like an activation gate:
`X AND rotl(X, n)`. The rotation amount is explicit so it can later become a
model parameter without changing the base `double` primitive.

`reduce_rotated_by_half` is the matching inverse fold for rotated doubles:
`first_half AND rotr(second_half, n)`. This preserves the identity
`reduce_rotated_by_half(double_with_rotated_half(X, n), n) == X`, while plain
`reduce_by_half` remains available when the lossy activation-gate behavior is
desired.

The implementation intentionally still has nuanced behavior around leading zero
words and logical length. Some operations work over allocated words, while
`get_length` derives a significant-bit length. That should not be casually
flattened without first deciding what model semantics require.
Use `get_bit_capacity` when physical storage width is the intended value.

### Bit String Lists

Files:

- `src/itty-bit-string-list.h`
- `src/itty-bit-string-list-private.h`
- `src/itty-bit-string-list.c`

`itty_bit_string_list_t` owns an array of `itty_bit_string_t *`. It is used for:

- token streams,
- node input collections,
- modulation mask collections,
- layer outputs,
- transposed bit matrices.

Important operations:

- append/fetch/iterate,
- element-wise XOR of two lists,
- transpose,
- condense by majority vote,
- weighted condense by vote budget,
- popcount vote allocation,
- popcount vote-mask generation,
- popcount argmax,
- popcount sorting,
- presentation.

`condense` is the key network primitive today. It transposes a list of bit
strings, counts set bits per bit position, and emits a bit when the majority
threshold is reached.

`weighted_condense` generalizes that vote model by giving each input an
integer vote budget. For each output bit, the set-bit votes are summed and
compared with a strict majority threshold over the total vote budget. This is
the synchronous semantic primitive for attention-like value mixing.

`itty_popcount_allocate_votes` turns popcount-like scores into an exact integer
vote budget using largest-remainder apportionment. The derived vote-mask helper
builds bit strings whose popcounts match those budgets. This is intentionally
not a floating-point softmax; it is a discrete normalization primitive for
bit-level voting.

### Mapped Bit String Files

Files:

- `src/itty-bit-string-map.h`
- `src/itty-bit-string-map-private.h`
- `src/itty-bit-string-map.c`

`itty_bit_string_map_file_t` wraps an mmap-backed file and exposes sequential
reads of fixed word-count bit strings.

Current responsibilities:

- open or create a file,
- map non-empty files,
- resize files,
- return borrowed copy-on-write bit strings pointing into the mapping,
- close/unmap on free.

This layer is used by:

- vocabulary bit-string files,
- context files,
- model files.

The map reader is intentionally simple: it advances `current_index` by the
requested word count. Callers decide how many words each record should contain.

### Vocabulary

Files:

- `src/itty-vocabulary.h`
- `src/itty-vocabulary.c`

`itty_vocabulary_t` ties a text vocabulary file to a bit-string file. Each line
in the text file maps to the next one-word bit string in the mapped binary file.

Current operations:

- load text and bit-string pairs,
- translate text to the best matching bit string,
- translate a bit string back to text by exact comparison,
- encode an input string to a binary context file.

Encoding currently performs longest-prefix matching at the current input
position. If no vocabulary entry matches, encoding fails. This avoids silent
skips and avoids infinite loops on unknown input.

The current implementation is O(input length * vocabulary size * token length)
and is suitable only for small experiments. A trie or other prefix index would
be a natural future replacement.

### Network

Files:

- `src/itty-network.h`
- `src/itty-network.c`

The network is a feed-forward structure:

- a network owns layers,
- each layer owns nodes,
- each node is either a feed node or an affinity node.

Feed nodes are created with `itty_network_feed_node_new` and own a list of
modulation masks. Affinity nodes are created with `itty_network_affinity_node_new`
and own trait/imprint lists plus base probe options. Node constructors transfer
ownership of their input lists only on success; if affinity node validation
fails, the caller still owns the trait and imprint lists.

Current feed-node behavior:

1. Start with an input list of bit strings.
2. For each layer:
   - for each feed node:
     - XOR current inputs with the node's modulation masks,
     - condense the modulated inputs by majority vote,
     - double the condensed output,
     - append it to the layer output list.
   - use the layer output list as the next layer's input.
3. Return the final output list.

The feed path returns outputs without printing intermediate state. Inspection
goes through `itty_network_layer_present` and `itty_network_present_plan`.
If a node cannot produce its output, both the synchronous feed and manager feed
return `NULL` rather than silently omitting that node's output.

The shape assumptions are still loose. `itty_bit_string_list_exclusive_or`
processes the shorter of the two lists, so a node with fewer masks than inputs
silently ignores extra inputs, and a node with more masks than inputs silently
ignores extra masks.

Affinity nodes treat the current layer input list as probes. They run
`itty_affinity_probe_list` with the node's traits, imprints, and base options,
then append every affinity output to the layer output list. This makes an
affinity node a contextualizing transform over the current layer, while feed
nodes remain one-output modulation/condense nodes.

### Inference

Files:

- `src/itty-inference.h`
- `src/itty-inference.c`

`itty_inference_run` is the library-level inference contract. It runs the
network with an optional manager, selects the final activation, decodes the
nearest vocabulary token, and returns an `itty_inference_result_t`.

The result exposes:

- the final output list,
- the selected output index,
- the selected activation,
- decoded text,
- nearest-token distance.

The caller still owns the input list. The result owns only outputs created by
the network, so empty-network inference can safely return the caller's input as
the output list without double-freeing it.

### Feed Model

Files:

- `src/itty-feed-model.h`
- `src/itty-feed-model.c`

`itty_feed_model_t` is the first trainable model owner. It is deliberately
feed-only: it owns mutable modulation masks, can build a network from cloned
masks, and can update its masks without transferring ownership to the network.
Each feed-model layer also carries a rotation amount. Forward execution uses
plain `double` when the rotation is zero, otherwise it uses
`double_with_rotated_half`.

The first training rule is intentionally narrow. `itty_feed_model_train_one`
runs the model forward to the final layer's input, expands the target to that
layer's width, and adjusts each final-layer node's masks so its majority vote
moves toward the expanded target. With one layer and one input this reduces to:

```text
mask = input XOR target
```

`itty_feed_model_train_one_with_options` adds the first bit-native learning
rate: `max_flips`, an integer budget for mask-bit flips in one update. The
current budget policy is largest-error-first. For each output bit, the trainer
computes how far the current modulated votes are from the majority threshold
needed for the target bit, sorts the largest deficits/excesses first, and spends
the flip budget there before lower-error bits.

The `_with_stats` training variants fill `itty_feed_model_train_stats_t` with
the number of mask-bit flips performed, the number of candidate output bits that
needed correction, and the largest single-bit vote error seen in that update.
Backward training aggregates those counters across the layers it trains. The
current `max_flips` budget is still applied per trained layer.

The current nearest-vocabulary decoder folds doubled outputs back to vocabulary
width. This is an MVP training loop, not a general multi-layer credit assignment
algorithm. Multi-node final-layer training is supported. Multi-node backward
training is supported only for segment folding; chained reduce remains scoped to
one node per layer because it propagates a single previous desired input.
Backward diagnostics are still scoped to one node per layer.

Feed-model mask shape follows the actual layer input shape:

- layer 0 nodes receive `inputs_per_node` masks,
- layer 1 and later nodes receive `nodes_per_layer` masks,
- every layer still has `nodes_per_layer` nodes.

That lets wider feed models run forward and train the final layer without
pretending every layer consumes the original external input width.

Fresh feed models still start with all-zero masks. `itty_feed_model_randomize_masks`
is an explicit opt-in symmetry breaker for wider models: it rewrites every mask
from a fixed seed and a rational density, `numerator / denominator`. The
generator is internal to the feed model rather than libc `rand()`, so the same
seed and density give reproducible masks across runs.

Final-layer node diagnostics compare each final node's condensed activation to
the expanded final target. In a two-node final layer, zero initialization gives
matching node diagnostics because both nodes begin identically and receive the
same target. Fixed-seed sparse randomization makes those node diagnostics
diverge immediately, which gives multi-node experiments a measurable symmetry
breaker without making runs non-reproducible.

A small final-layer training shape sweep compared zero initialization with
fixed-seed sparse masks at density `1/8`. All tested shapes reached the target.
Random initialization was not universally cheaper for tiny models, but it helped
as width grew. In the 4-layer, 8-node shape, zero initialization reached the
target in 18 steps and 1152 flips; seeded sparse initialization reached it in 15
steps and 886 flips while finishing with higher mask entropy. That is the first
positive signal that width plus reproducible mask diversity can reduce work.

### Model Metrics

Files:

- `src/itty-model-metrics.h`
- `src/itty-model-metrics.c`

`itty_model_metrics_t` does not exist as an owning object yet. The current
metrics surface is a small set of summary functions over bit strings, bit-string
lists, and feed-model masks. A summary records total bits, set bits, unset bits,
set density, and normalized binary entropy.

Entropy uses stored bit capacity as the denominator rather than significant-bit
length. That makes all-zero masks and activations measurable: they report zero
set density and zero entropy instead of disappearing as length-zero values. This
is the right behavior for training diagnostics because a collapsed all-zero mask
is meaningful model state.

Feed-model mask summaries are intentionally read-only. They let the training
loop ask whether masks are balanced, saturated, or collapsed without exposing
the mutable mask arrays.

Activation traces run a network layer by layer and keep one summary per layer.
The trace owns only metrics, not the intermediate bit-string lists, so it is
small enough to use around training loops. This gives the project a first way to
watch entropy drift as activations widen, rotate, fold, or pass through affinity
nodes.

### Training Observation

Files:

- `src/itty-training-observation.h`
- `src/itty-training-observation.c`

`itty_training_observation_t` is a small diagnostic harness around one feed-model
training step. It records:

- mask summaries before and after training,
- activation traces before and after training,
- training stats from the `_with_stats` trainer,
- target distance before and after training, measured by folding the selected
  activation to the target width and XORing it with the target.

This is not a trainer and does not change the learning rule. It is the first
place where the project can ask whether a step helped decode distance while also
watching entropy, density, and vote-error shape.

`itty_training_history_t` repeats the same observed training step up to a maximum
number of steps and stores one compact summary row per step. Each row records the
target distance before and after the step, the train stats, and the before/after
mask and final-activation entropy. The history runner stops early when a step
reaches target distance zero.

The first budgeted dense-target check gives a readable learning-rate curve with
an eight-flip budget:

```text
32 -> 24 -> 16 -> 8 -> 0
```

That makes the integer flip budget visible as a training trajectory instead of a
single update.

`itty_training_optimizer_t` is the first budget-selection abstraction. It does
not mutate the model and does not inspect mask storage directly; it only chooses
training options for the next observed step. The initial policies are:

- fixed budget: always use the configured `max_flips`,
- distance-capped budget: use `min(current_target_distance, max_flips)`,
- distance-fraction budget: use
  `ceil(current_target_distance * numerator / denominator)`, capped by
  `max_flips` when a maximum is configured.

The distance-capped policy keeps sparse corrections from spending a large budget
when only a few bits are wrong, while preserving the same eight-flip dense-target
curve when the remaining distance is larger than the cap.

The first distance-fraction check uses half-distance. On the dense target it
produces a more conservative trajectory:

```text
32 -> 16 -> 8 -> 4 -> 2 -> 1 -> 0
```

On the sparse target it produces:

```text
4 -> 2 -> 1 -> 0
```

History summaries make optimizer comparisons explicit. In the current one-layer
deterministic trainer, fixed and half-distance policies reach the same final
state and spend the same total flips; half-distance only spreads those flips
over more steps. On the dense target fixed budget reaches the target in four
steps while half-distance takes six, and both spend 32 flips. On the sparse
target fixed budget takes one step while half-distance takes three, and both
spend four flips.

The first two-layer backward histories show the same broad optimizer tradeoff
with a deeper model, but the backward pass spends more flips overall. On a
two-layer dense target, fixed budget reaches the target in four steps and
half-distance in six; both spend 48 flips. With a hidden rotation on layer zero,
fixed budget takes five steps and half-distance takes six; both spend 49 flips.
Both rotated and unrotated runs finish with final activation entropy at 1.0.

Three-layer backward histories are the first setup where budget trajectory
changes total work. On a three-layer dense target, fixed budget reaches the
target in eight steps and spends 104 flips. Half-distance takes nine steps but
spends only 96 flips and finishes with higher mask entropy. Adding a hidden
rotation on layer zero keeps the same step and flip counts in this setup. Final
activation entropy still reaches 1.0.

An eight-layer bounded stress check does not converge in the first eight
observed steps. Both fixed and half-distance remain at target distance 32. Fixed
budget spends 368 flips; half-distance spends 640 flips and has higher mask
entropy. A longer manual probe showed fixed budget improving by 128 steps, but
not reaching the target. This is the first clear depth limit for the current
one-node backward rule.

Backward observations now include per-layer summaries. The first eight-layer
step shows the failure shape: layer 0 receives no candidate bits, while candidate
bits grow rapidly toward the output layer:

```text
layer:      0   1   2    3    4     5     6     7
candidates:0  16  88  224  488  1008  2040  4096
flips:     0   8   8    8    8     8     8     8
```

That suggests the current backward rule is not delivering useful corrective
signal all the way to the earliest layer at this depth, while later layers have
very broad error surfaces.

An eight-layer fixed-budget sweep over the first eight observed steps shows that
larger budgets increase work and mask entropy but do not reduce target distance
in that window:

```text
budget:  8   16    32
flips: 368  640  1088
distance:32  32    32
```

So simply increasing the global budget is diagnostic but not a fix for the
missing early-layer signal.

The feed model also has an experimental segment fold for backward training.
Instead of reducing the desired output one layer at a time, the segment fold
keeps the final desired output at full width and traces each final bit backward
through the expansion path for the layer being trained. First halves map
straight through; rotated second halves map through the inverse rotation at that
layer width. Each layer bit is then selected by majority vote over all final
bits that trace back to it.

On the same eight-layer fixed-budget setup, segment folding sends candidate bits
to every layer immediately:

```text
layer:       0   1    2    3    4     5     6     7
candidates:32  64  128  256  512  1024  2048  4096
flips:      8   8    8    8    8     8     8     8
```

With an eight-flip budget, this reaches the dense target in two observed steps
and spends 128 total flips. That makes segment folding the clearest current path
for deep training, while the chained reduce rule remains useful as the baseline.
The same eight-layer setup with a one-bit rotation at every layer also reaches
the target in two steps with 128 flips. A more adversarial probe that varies the
rotation by layer is still harder: with rotations 1 through 8, the first 16
observed steps ended at distance 26 with an eight-flip budget. That suggests
rotation-aware segment folding is useful, but varied rotations may need better
budget selection or a different target-shaping rule.

Additional backward-layer diagnostics showed a sharper failure mode. In the
varied-rotation case, after 128 observed steps every layer's actual condensed
activation exactly matched the segment-fold desired target, but the final folded
output remained 16 bits away. Leaving hidden rotations varied while keeping the
final output layer unrotated reached the target in 23 steps. This suggests the
stall was not hidden-layer training failure; it was a mismatch between rotating
the final output layer and evaluating the final activation with the current
unrotated folding rule.

Segment-fold backward can also train multi-node layers by sending the same
layer desired target to every node and aggregating the node stats into the
layer stats. This proves the wider backward path works, but it is still a crude
rule. A first sweep showed the same-target rule converges for smaller multi-node
models, while random sparse initialization is not yet consistently helpful in
deep backward training. For example, a three-layer, two-node zero-initialized
model reached the dense target in four steps and 192 flips. At eight layers,
two zero-initialized nodes reached in eight steps and 768 flips, but four nodes
did not converge in 32 steps. That points toward node-specific desired targets,
not simply more same-target nodes, as the next likely improvement.

A temporary phased-target experiment gave each node a rotated desired target and
allocated the layer budget toward nodes whose current activations already
matched their phased target better. This did not improve the current benchmark
set. On small shapes it usually increased steps and flips. On the eight-layer,
two-node shape, stride-one phased targets got close but stalled at distance 1
after 32 steps, while the same-target baseline already reached the target in
eight steps. The experiment was removed from the live training path; the next
multi-node target experiment should probably preserve final-output alignment
more directly, such as partitioning bits or using segment-fold disagreement.

The first disagreement-target experiment is available as an opt-in segment-fold
mode for exactly two nodes. Node 0 learns the majority segment target. Node 1
learns the minority bit only where traced final-output votes disagree; where
the traced votes are unanimous, it preserves its current condensed activation.
This does capture a different signal, but the first probe was not broadly
encouraging:

```text
shape      init    rotations  same target        disagreement target
3 x 2      zero    none       4 steps / 192      28 steps / 352
3 x 2      random  varied     13 steps / 375     23 steps / 305
4 x 2      zero    none       4 steps / 256      60 steps / 832
4 x 2      random  varied     24 steps / 757     46 steps / 733
8 x 2      zero    varied     distance 14 / 64   distance 32 / 64
8 x 2      random  varied     distance 32 / 64   distance 32 / 64
```

The disagreement node tends to raise mask entropy, and under random
initialization it can spend fewer flips on smaller rotated cases. But it usually
takes many more steps, and the eight-layer cases get worse. That makes it a
diagnostic target-shaping rule, not a reason to move directly to four nodes.

A related hybrid target rule gives node 0 the original pairwise reduce target
and node 1 the segment-condensed target. This is also available as an opt-in
two-node mode. It keeps node 0 on the local inverse-fold path while node 1 sees
the full final-output trace. The first probe was still weaker than the
same-target baseline:

```text
shape      init    rotations  same target        pairwise + segment
3 x 2      zero    none       4 steps / 192      14 steps / 400
3 x 2      random  varied     13 steps / 375     17 steps / 446
4 x 2      zero    none       4 steps / 256      29 steps / 904
4 x 2      random  varied     24 steps / 757     34 steps / 977
8 x 2      zero    varied     distance 14 / 64   distance 32 / 64
8 x 2      random  varied     distance 32 / 64   distance 32 / 64
```

This hybrid keeps small-case final activation entropy near 1.0, unlike the
minority-disagreement rule, but it is still not a convergence improvement. The
common failure is that splitting node targets by backward target shape alone
adds work without improving the selected final folded output.

A segment-partition target rule keeps both nodes on the segment-fold trace but
deals each traced vote for a layer bit round-robin across the nodes. Each node
then learns the majority target for its own partition. This is closer to the
same-target baseline than the previous splits:

```text
shape      init    rotations  same target        segment partition
3 x 2      zero    none       4 steps / 192      4 steps / 192
3 x 2      zero    varied     6 steps / 226      7 steps / 250
3 x 2      random  varied     13 steps / 375     13 steps / 382
4 x 2      zero    none       4 steps / 256      4 steps / 256
4 x 2      zero    varied     10 steps / 376     13 steps / 460
4 x 2      random  varied     24 steps / 757     25 steps / 790
8 x 2      zero    none       8 steps / 768      8 steps / 768
8 x 2      zero    varied     distance 14 / 64   distance 26 / 64
8 x 2      random  varied     distance 32 / 64   distance 32 / 64
```

This says that preserving final-trace alignment matters: partitioning is far
less destructive than minority-disagreement or pairwise-plus-segment targets.
But partitioning alone still does not explain the hard rotated eight-layer
case, so simply trying four partitioned nodes is not yet justified by the data.

A downstream-mask-aware target rule derives hidden node targets from how the
next layer's current masks would consume that node. For a hidden node output
`x_j` feeding a downstream node with mask `m_j`, the downstream desired
condensed output `d` requests:

```text
x_j = d XOR m_j
```

The rule reduces that requested doubled output back through the hidden node's
own rotation and trains the node toward the majority request from downstream
nodes. Ties preserve the node's current condensed activation. The final layer
still uses the ordinary segment target because there is no downstream mask layer
to invert.

The first probe is mixed but more interesting than the hand-designed splits:

```text
shape      init    rotations  same target        downstream-mask-aware
3 x 2      zero    none       4 steps / 192      6 steps / 216
3 x 2      zero    varied     6 steps / 226      7 steps / 224
3 x 2      random  varied     13 steps / 375     13 steps / 290
4 x 2      zero    none       4 steps / 256      6 steps / 312
4 x 2      zero    varied     10 steps / 376     13 steps / 436
4 x 2      random  varied     24 steps / 757     27 steps / 663
8 x 2      zero    none       8 steps / 768      8 steps / 812
8 x 2      zero    varied     distance 14 / 64   distance 32 / 64
8 x 2      random  varied     distance 32 / 64   distance 32 / 64
```

The promising part is that randomized small rotated shapes spend fewer flips
and reach higher mask entropy. The bad part is that the hard eight-layer varied
rotation case gets worse. This suggests downstream-mask awareness is the right
family, but the simple majority request is too blunt; the next refinement should
carry confidence and majority-margin urgency instead of treating every
downstream request equally.

A first margin-weighted downstream-mask variant tried to do that directly:
wrong downstream bits receive weight based on their majority deficit or excess,
fragile correct bits receive a small stabilizing weight, and comfortable correct
bits stay quiet. That was too aggressive:

```text
shape      init    rotations  equal downstream   margin weighted
3 x 2      zero    none       6 steps / 216      18 steps / 628
3 x 2      zero    varied     7 steps / 224      20 steps / 698
3 x 2      random  varied     13 steps / 290     distance 3 / 64
4 x 2      zero    none       6 steps / 312      14 steps / 572
4 x 2      zero    varied     13 steps / 436     distance 5 / 64
4 x 2      random  varied     27 steps / 663     distance 32 / 64
8 x 2      zero    none       8 steps / 812      distance 24 / 64
8 x 2      zero    varied     distance 32 / 64   distance 32 / 64
8 x 2      random  varied     distance 32 / 64   distance 32 / 64
```

The failure mode is useful: weighting every request by downstream margin makes
the hidden targets churn, spending many more flips and sometimes missing small
targets entirely. A better confidence rule should probably gate ambiguous or
already-good bits out of target generation rather than amplify all wrong bits.

A gated downstream-mask variant tried that narrower rule: equal request votes
are emitted only from downstream bits that are currently wrong or exactly
fragile at the majority boundary. Bits that are comfortably correct send no
request. This avoids some of the weighted churn, but it still does not beat the
equal downstream rule:

```text
shape      init    rotations  equal downstream   gated downstream
3 x 2      zero    none       6 steps / 216      6 steps / 216
3 x 2      zero    varied     7 steps / 224      distance 1 / 64
3 x 2      random  varied     13 steps / 290     12 steps / 570
4 x 2      zero    none       6 steps / 312      6 steps / 312
4 x 2      zero    varied     13 steps / 436     distance 2 / 64
4 x 2      random  varied     27 steps / 663     44 steps / 2811
8 x 2      zero    none       8 steps / 812      8 steps / 792
8 x 2      zero    varied     distance 32 / 64   distance 24 / 64
8 x 2      random  varied     distance 32 / 64   distance 32 / 64
```

The hard varied-rotation case improves over equal downstream in one zero-init
distance metric, but it spends the full budget and remains worse than the
same-target baseline's distance 14. Gating therefore looks like useful
diagnostic pressure, not a production training rule yet.

The feed model now has a suffix-oracle diagnostic for symbolic hidden targets.
For a given layer, node, and backward target rule, it collects the condensed-bit
flips that the rule would propose, applies each proposed flip to a temporary
copy of that node's layer output, runs the frozen suffix, and measures the real
selected folded output distance. This reports how many proposed flips help,
hurt, or do nothing.

On the hard eight-layer, two-node, varied-hidden-rotation case after 64
same-target segment-fold steps, the model is stalled at distance 14. Sampling up
to 256 proposed flips per layer/node showed:

```text
rule                 hidden layers 0-4        layer 5              layer 6
same target          no proposals             no proposals         all neutral
segment partition    all harmful              all harmful          all neutral
equal downstream     mostly none/harmful      all harmful          all neutral
gated downstream     no proposals             mostly none/neutral  all neutral
```

The final layer is different:

```text
layer 7, each node: 256 sampled proposals, 6 helpful, 0 harmful, 250 neutral
```

This strongly suggests the current hidden symbolic targets are not aligned with
the actual selected decoded output once the model reaches this stall. The suffix
still has useful single-bit final-layer moves, but the hidden-layer proposals
are either disconnected from the final error or actively move the selected
decode in the wrong direction. That makes decoder-exact tri-state constraints,
suffix-improvement targets, or accept/reject proposal training higher-value than
more hand-designed full-bitstring hidden targets.

The suffix oracle also supports random one-bit candidates and neutral-cause
classification. Neutral candidates are split into:

```text
same folded output       the selected folded decode did not change
changed folded output    the decode changed but target distance tied
changed selected output  popcount selection changed, but distance tied
```

On a small three-layer rotated case, symbolic final proposals had 5 helpful and
27 neutral candidates out of 32; all 27 neutral candidates left the folded output
unchanged. Random one-bit sampling had 2 helpful and 30 neutral candidates; 22
neutral candidates left the folded output unchanged, while 8 changed the selected
output without changing distance. That gives a quick way to tell whether neutral
means "fold masked it away" or "selection/contrast still tied."

`itty_feed_model_decoder_objective_t` is the first-class score object for the
selected decoder path. It records:

```text
selected_distance
false_negative_count
false_negative_blocker_bits
zero_veto_safety_bits
nearest_wrong_margin
selected_node
selected_popcount
best_decoded_node
best_decoded_distance
```

An oracle-filtered final-layer trainer uses this objective as a training gate. It
builds final condensed proposals, tests them against the frozen suffix, and
trains only the bits that improve the lexicographic decoder objective:

```text
1. reduce selected folded target distance
2. if distance ties, reduce false-negative decoder blockers
3. if still tied, improve target-vs-nearest-wrong margin
4. if still tied, improve zero-veto safety
5. reject decoded-distance regressions
```

The nearest-wrong margin is currently a reserved objective slot because the feed
model trainer receives only a target bit string, not a vocabulary. The measured
parts of the objective are distance, false-negative blockers, and zero-veto
safety.

On the hard eight-layer stalled model:

```text
after same-target segment-fold: distance 14
oracle final step 0:            distance 11, 16 flips
oracle final step 1:            distance  8, 16 flips
oracle final step 2:            distance  8, 16 flips, blockers 948
```

This confirms that the stall was not absolute: final-layer moves can still reduce
the real objective, but symbolic hidden proposals were not finding useful paths
to create them.

The first decoder-exact tri-state final proposals are also available. For final
decoder ones, every final activation ancestor is required to be one. For decoded
zeros, only one zero-veto ancestor is assigned when the folded output is
currently wrong; the rest stay unconstrained. On the hard stalled final layer,
the hit rate matched the old expanded final proposals in the first 256 sampled
bits per node:

```text
old expanded final target:      6 helpful, 0 harmful, 250 neutral
decoder tri-state final target: 6 helpful, 0 harmful, 250 neutral
random one-bit final samples:   1 helpful, 0 harmful, 255 neutral
```

The equal hit rate means the current simple tri-state assignment is not yet a
better final proposal generator, but the contrast with random sampling shows the
decoder-targeted proposals are still much richer than blind flips.

At the distance-8 plateau after two oracle-filtered final-layer steps, the
residual decoded errors split cleanly:

```text
false positives, folded = 1 and target = 0: 0
false negatives, folded = 0 and target = 1: 8
```

That points away from cost-aware zero-veto assignment as the next trainer. The
remaining errors are target-one failures, and the AND decoder needs every traced
ancestor of those decoded bits to become one. A one-bit proposal can therefore be
neutral even when the whole ancestor block would help.

With the blocker-count objective and decoded-bit block proposals, those neutral
scaffolding moves become acceptable as long as decoded distance does not regress.
The same hard case now keeps distance at 8 while blockers fall, then drops once
enough ancestors have been repaired:

```text
oracle final step  1: distance 8, false negatives 8, blockers 956
oracle final step  9: distance 7, false negatives 7, blockers 892
oracle final step 119: distance 6, false negatives 6, blockers 12
oracle final step 120: distance 2, false negatives 2, blockers 4
oracle final step 121: distance 0, false negatives 0, blockers 0
```

Zero-veto safety stays stable in this run, so the extra progress is coming from
target-one scaffolding rather than from trading away target-zero guarantees.

The suffix oracle now classifies candidates with the same decoder objective. Its
original distance-only counters remain, and it also records distance-neutral
proposals that reduce false-negative blockers. Re-running the hard eight-layer
stalled model shows that blocker-aware credit is still concentrated at the final
layer. Aggregating both nodes per layer with 256 sampled proposals per node:

```text
same-target
layer candidates strict_distance_helpful blocker_helpful harmful true_neutral
0     0          0                       0               0       0
1     0          0                       0               0       0
2     0          0                       0               0       0
3     0          0                       0               0       0
4     0          0                       0               0       0
5     0          0                       0               0       0
6     388        0                       0               0       388
7     512        12                      496             0       4
```

```text
segment-partition
layer candidates strict_distance_helpful blocker_helpful harmful true_neutral
0     1          0                       0               1       0
1     6          0                       0               6       0
2     21         0                       0               21      0
3     61         0                       0               61      0
4     156        0                       0               156     0
5     384        0                       0               384     0
6     512        0                       0               0       512
7     512        12                      496             0       4
```

```text
downstream-mask-aware
layer candidates strict_distance_helpful blocker_helpful harmful true_neutral
0     2          0                       0               2       0
1     4          0                       0               4       0
2     8          0                       0               8       0
3     64         0                       0               64      0
4     32         0                       0               32      0
5     450        0                       0               450     0
6     388        0                       0               0       388
7     512        12                      496             0       4
```

Layer 6 remains true-neutral under the blocker-aware objective. That means the
current single-bit hidden proposals still do not reduce selected decoded distance
or false-negative blockers. The next trainer should not be layer-6
suffix-improvement yet; the evidence still points to final-layer scaffold
training, or to hidden block proposals that can move several hidden bits together
before the suffix sees a blocker change.

A follow-up diagnostic projected the final oracle's accepted condensed-bit
repairs back into layer-6 output space. The experiment:

```text
1. Train the hard case to the distance-14 same-target stall.
2. Run the current final oracle proposal pass and record accepted final
   condensed-bit repairs.
3. For each accepted final repair, build the minimal layer-6 output block that
   would cause that same final condensed-bit value through the current final
   masks.
4. Score each projected block with the lexicographic decoder objective.
5. Compare against random layer-6 blocks with the same block-size distribution.
```

The result:

```text
accepted final repairs: 1348
before distance:        14
before blockers:        972

source            blocks  avg size  max size  strict  blocker  objective  harmful  neutral
projected layer6  1348    1.72      2         12      960      376        0        0
random layer6     1348    1.72      2         0       0        33         562      753
```

This changes the interpretation. Layer 6 is not disconnected; the current
symbolic one-bit layer-6 proposals are just not the right proposals. When layer-6
blocks are derived from accepted final condensed repairs, every block improves
the lexicographic objective and most reduce false-negative blockers directly.
The next hidden trainer should therefore use final-repair-derived layer-6 block
targets rather than generic layer-6 one-bit suffix sampling.

`itty_feed_model_train_penultimate_layer_with_final_repairs` is the first trainer
built from this idea. It keeps the final oracle unchanged, but uses its accepted
final condensed repairs as a teacher for the penultimate layer:

```text
1. collect accepted final condensed repairs
2. project each repair through the current final masks into layer-6 output blocks
3. convert each output block back through layer-6 rotation into condensed
   constraints
4. re-expand the condensed constraints and score the realistic paired output
   delta with the lexicographic decoder objective
5. estimate the exact majority-margin mask-flip cost for each block
6. sort blocks by objective class, blocker reduction, full-realization cost,
   and constraint count
7. accept only non-conflicting blocks whose full estimated cost fits the
   remaining layer budget
8. train layer-6 masks only on selected cared-for condensed bits
9. measure requested/realized/lost/extra layer-6 output bits
```

The trainer also runs a condensed-realistic projection oracle before training:

```text
accepted final repair
-> minimal layer-6 output block
-> layer-6 condensed constraints
-> re-expand through layer-6 rotation
-> score the full paired layer-6 output delta
```

This checks whether the projected repair remains useful when the penultimate
layer's unavoidable doubled or rotated-doubled output pair is included. For the
same `N = 128` hard-case run:

```text
condensed-realistic blocks:     128
strict-distance helpful:          6
blocker helpful:                120
other objective helpful:          0
harmful:                          0
neutral:                          2
```

So the projection was not merely optimistic output-block scoring. Almost every
projected repair remains useful after collapsing to condensed constraints and
re-expanding through the layer-6 rotation.

On the hard case with `max_projected_blocks = 128` and `max_layer_flips = 256`:

```text
projected blocks accepted:      98
fully realized blocks:          98
partially realized blocks:       0
unrealized blocks:               0
estimated layer-6 flips:       254
actual layer-6 flips:          254
requested condensed bits:      196
realized condensed bits:       196
requested output bits:         196
realized output bits:          196
lost output bits:                0
extra output bits changed:     196
structural extra output bits:  196
collateral extra output bits:    0
distance:                       14 -> 8
blockers:                      972 -> 776
```

The cost-aware selector changes the failure mode. It accepts fewer projected
blocks, but all selected blocks are fully realized, and every extra output
change is structural: it comes from the doubled or rotated-doubled companion
bits implied by the selected condensed constraints. No collateral output changes
were measured in this run.

The current hard-case baseline progression is:

```text
schedule                         cleanup steps  final flips  hidden flips  total flips
final oracle only                122            1952         0             1952
project layer6, uncapped/noisy   102            1632         256           1888
project layer6, cost-complete    97             1552         254           1806
```

This is the first end-to-end sign that hidden repair projection can absorb work
that the final layer previously had to learn by itself, and that the immediate
bottleneck is now stale teacher signal rather than projection quality or
mask-level realization.

`itty_feed_model_train_penultimate_layer_with_refreshed_final_repairs` addresses
that stale-teacher problem by alternating small repair-projection batches:

```text
repeat:
        collect final oracle repairs on the current model
        project them into layer-6 condensed constraints
        select a cost-aware complete mini-batch
        snapshot layer-6 masks
        train layer-6
        keep the batch only if the decoder objective improves
until the objective reaches zero or a batch is rejected
```

On the same hard case, refreshed batches no longer need final cleanup:

```text
batch  rounds  accepted blocks  hidden flips  cleanup steps  final flips  total flips
8      99      487              1560          0              0            1560
16     50      490              1560          0              0            1560
32     25      493              1560          0              0            1560
64     13      493              1560          0              0            1560
```

Every selected block was fully realized in these runs, with no collateral output
changes. This makes refreshed layer-6 repair projection the current best
trainer for the hard eight-layer case.

The refreshed trainer also records round-level trajectory data:

```text
accepted/reverted
selected node
best decoded node and selected-is-best flag
selected popcount and popcount gap
distance before/after/delta
false-negative blockers before/after/delta
accepted projected blocks
layer flips
```

For the batch-16 hard-case baseline, the first round moves:

```text
distance: 14 -> 8
blockers: 972 -> 956
accepted blocks: 8
layer flips: 32
selected node: 0
best decoded node: 0
selected is best decoded: true
popcount gap: 0
```

The same refreshed layer-6 sweep is less decisive on wider or randomly
initialized variants:

```text
shape              batch  rounds  hidden flips  distance  blockers
4-node zero        32     4       74            32 -> 32  1446 -> 1410
4-node zero        64     19      2898          32 -> 23  1446 -> 676
2-node sparse      32     12      116           32 -> 32  5288 -> 4984
2-node sparse      64     12      255           32 -> 32  5288 -> 4752
4-node sparse      32     9       265           32 -> 32  3056 -> 2884
4-node sparse      64     20      672           32 -> 32  3056 -> 2652
```

So refreshed layer-6 projection is the new hard-case baseline, but it is not yet
a universal trainer. The sparse and 4-node runs often reduce blocker counts
without reducing selected decoded distance. The added selected-vs-best decoded
trajectory did not expose a selection mismatch in these runs: the first round
started with the selected node also being the best decoded node in each targeted
sweep case.

The trainer now includes a layer-5 projection diagnostic: every selected
layer-6 repair block is projected backward through the layer-6 masks into
layer-5 condensed-realistic constraints, then scored with the suffix objective.
For the batch-16 hard-case baseline:

```text
layer-5 projected blocks:      490
strict-distance helpful:        14
blocker helpful:               221
other objective helpful:         0
harmful:                       221
neutral:                        34
```

Pinned-selected-node scoring gives the same counts for this baseline, so the
harmful blocks are not explained by a selected-node switch. Harmful-cause
decomposition shows all 221 harmful cases are selected-distance regressions,
not blocker-only, margin-only, or zero-veto-safety regressions:

```text
harmful distance regressions: 221
harmful blocker regressions:    0
harmful margin regressions:     0
harmful safety regressions:     0
```

The decoded-bit transition matrix for those harmful layer-5 blocks is more
specific:

```text
correct 0 -> false positive:      0
correct 1 -> false negative:    221
false positive -> correct 0:      0
false negative -> correct 1:      0
false positive -> false negative: 0
false negative -> false positive: 0
unchanged wrong:               1585
unchanged correct:            12338
```

So the harmful layer-5 blocks are breaking previously correct one bits. This
motivates a stricter layer-5 rule than the layer-6 rule: if selected distance
does not strictly improve, reject any block that turns a correct decoded bit
wrong.

The targeted sparse and 4-node sweeps show a different layer-5 profile:

```text
shape              batch  prev blocks  prev strict  prev blocker  prev harmful  prev neutral
2-node sparse      32     101          0            29            0             68
2-node sparse      64     190          0            61            0             122
4-node zero        32     18           0            8             0             8
4-node zero        64     308          8            192           2             94
4-node sparse      32     68           0            28            0             38
4-node sparse      64     168          0            71            0             88
```

Most non-baseline layer-5 projections are neutral or blocker-helpful, not
distance-harmful. The hard-case baseline is the odd case where layer-5 recursion
has a large distance-regression tail. A conservative layer-5 micro-trainer should
therefore use only strict-distance or blocker-helpful projected blocks, in tiny
accepted batches, followed by refreshed layer-6 cleanup. A generalized
layer-indexed refreshed repair trainer should wait until that filtered
micro-trainer proves useful.

`itty_feed_model_train_antepenultimate_layer_with_projected_repairs` is the
first such conservative micro-trainer. It:

```text
1. collects final repairs from the current layer-6 teacher
2. projects accepted layer-6 condensed repairs into layer-5 blocks
3. scores each layer-5 block through the full decoder objective
4. keeps only strict-distance-helpful blocks, or blocker-helpful blocks that
   preserve every currently correct decoded bit
5. sorts by strictness, blocker delta, realization cost, and block size
6. applies a tiny complete-block batch
7. reverts the batch unless the full decoder objective improves
8. runs refreshed layer-6 cleanup afterward
```

On the hard case:

```text
layer5 batch  layer5 flips  after layer5  layer6 cleanup flips  total flips
1             0             14            1560                  1560
2             2             13            1552                  1554
4             6             12            1549                  1555
8             14            10            1542                  1556
```

The batch-2 micro-step is the current best two-stage result: it beats refreshed
layer-6 alone by 6 flips (`1554` total versus `1560`). The gain is small but
important: layer 5 can help if the teacher is filtered by decoded-bit
preservation and batch size stays tiny.

A class-separated layer-5 sweep shows where that gain comes from:

```text
mode                accepted strict  accepted blocker  layer5 flips  L6 flips  total
strict only 1       1                0                 2             1552      1554
strict only 2       2                0                 6             1549      1555
strict only 4       4                0                 14            1542      1556
blocker only 1      0                1                 4             1557      1561
blocker only 2      0                2                 8             1554      1562
strict 2 + blocker 2 2              2                 14            1542      1556
```

So the best layer-5 assist is strict-only with one block. Blocker-only repairs
are real, but on this hard case their cost is higher than the layer-6 work they
save. That makes the default optional layer-5 pre-pass:

```text
max_strict_distance_blocks = 1
max_blocker_blocks = 0
```

Blocker-only layer-5 assists remain worth testing on sparse and 4-node cases,
where the observed failure mode is blocker reduction without distance movement.

### Gated OR residuals

The first residual experiment is deliberately asymmetric and no-op by default.
Each feed node now has a residual enable mask and residual mask:

```text
h_j      = condense(inputs XOR masks_j)
skip_j   = inputs[j % input_count] XOR residual_mask_j
merged_j = h_j | (residual_enable_j & skip_j)
output_j = double_or_rotated_double(merged_j)
```

`residual_enable_j` starts at zero, so existing models behave exactly as before
unless a trainer explicitly enables residual bits. The OR merge can add target
one support without directly clearing existing ones, which matches the fragile
AND-decoder failure mode where upstream repairs can break already-correct
target-one decoded bits.

The refreshed layer-6 repair trainer can optionally consider the residual lane
as a second realization path for positive condensed constraints. For each
candidate repair it compares:

```text
normal path:   flip layer-6 majority votes until h_j reaches the target bit
residual path: make skip_j true and enable the residual bit
```

The residual path is selected only when it is strictly cheaper than the normal
majority realization. A tie policy was tested and rejected: allowing equal-cost
residual substitutions caused the first hard-case batch to regress under the
decoder objective and be reverted.

Initial probe results:

```text
case                         total flips  residual blocks  result
hard layer6 only             1560         0                unchanged
hard layer5 strict-1 + L6    1554         0                unchanged
4-node plateau batch         22 -> 18     2                fewer flips, fewer blockers fixed
```

So gated OR residuals are implemented but remain opt-in. On the current hard
2-node case they are visible as candidates but not cheaper than the trusted
layer-6 majority path. On the 4-node plateau they can reduce flip cost, but the
first small probe traded away some blocker reduction, so residuals need a
plateau-specific policy before becoming a default trainer component.

### Segment-Condense Decoder

The original feed-model decoder selects the final output node by popcount and
then repeatedly folds the selected activation with pairwise AND until it reaches
vocabulary width:

```text
wide activation -> reduce_by_half -> ... -> vocabulary-width activation
```

That makes target-one decoded bits fragile, because a single zero anywhere in a
fold ancestry can veto the decoded one.

The alternate decoder mode, `ITTY_FEED_MODEL_DECODER_SEGMENT_CONDENSE`, keeps
the same popcount node selection but replaces repeated pairwise folding with a
segment vote. If the selected activation has `N` vocabulary-width segments, each
vocabulary bit is decoded by majority vote across the corresponding bit in all
segments:

```text
decoded[k] = condense(activation[k],
                      activation[k + vocab_width],
                      activation[k + 2 * vocab_width],
                      ...)
```

This changes blocker accounting from AND ancestry to vote margin:

```text
target 1 and decoded 0:
        blockers = threshold - ones

target 0 and decoded 0:
        safety = max_ones_for_zero - ones
```

The mode is model-local and defaults to the original repeated-AND decoder.
Initial measurement on the same stalled states is encouraging:

```text
shape                 repeated AND distance  segment-condense distance
2-node hard           14                     5
2-node sparse         32                     32
4-node zero           32                     0
4-node sparse         32                     6
```

The segment-condense decoder did not introduce false positives in these probes;
it mainly reduced false negatives by making target-one support majority-based
instead of all-ancestors-required. The next question is whether repair
projection should train under this decoder directly, since its blocker margins
are much less brittle than the AND-fold margins.

Repair projection now trains against the segment-condense objective when the
model decoder is set to `ITTY_FEED_MODEL_DECODER_SEGMENT_CONDENSE`. The
decoder objective exposes segment-native vote fields:

```text
false_negative_vote_deficit
false_positive_vote_excess
target_one_margin
target_zero_safety
```

Final repair block proposals are decoder-aware. Under repeated AND they keep
the old all-ancestor repair/veto behavior; under segment condense they flip only
enough segment votes to cross the majority threshold.

Initial refreshed layer-6 repair projection under the segment decoder solved
the hard case, but still spread sparse repairs too widely:

```text
shape          before  after  layer6 flips  note
2-node hard    5       0      220           solved in one refreshed round
2-node sparse  32      28     443           deficit 1224 -> 472
4-node zero    0       0      0             already decoded correctly
4-node sparse  6       4      737           deficit 124 -> 66
```

The residual OR lane did not help the segment objective in these probes. It was
unused for the hard and 4-node sparse cases, and on 2-node sparse it cut flips
but stalled at a worse distance (`30` instead of `28`). The current segment
path should therefore be tested first without residuals.

Threshold-completion-aware sorting now ranks projected repair candidates by
decoded-distance movement and false-negative-count reduction before aggregate
deficit reduction. This did not change the first sparse sweep, which means the
remaining sparse plateaus are not just an ordering artifact in the layer-6
candidate queue.

The next segment-native repair generator changed the proposal unit from a
single final condensed bit to a decoded-bit vote quota. For each false-negative
decoded bit, it chooses enough zero segment votes to cross that bit's majority
threshold; for each false positive, it chooses only enough one segment votes to
fall back below threshold. This concentrates work on completing decoded votes
instead of reducing many large deficits a little.

With quota-shaped proposals, the segment decoder becomes the strongest current
path. The primary trainer name for this path is:

```c
itty_feed_model_train_segment_condense_quota_repair_projection
```

That name is intentionally explicit: final decoded errors become
segment-condense vote quotas, the cheapest quota votes are selected
deterministically, and those repairs are projected across the current layer
boundary into the previous layer's condensed constraints.

```text
shape          dist    flips  policy
2-node hard    5->0    10     segment-condense quota repair projection
2-node sparse  32->0   664    segment-condense quota repair projection
4-node zero    0->0    0      already decoded
4-node sparse  6->0    189    segment-condense quota repair projection
```

This supersedes the first segment repair table above. The hard-case regression
now locks the stronger `5 -> 0` result with roughly ten layer-6 flips rather
than the older roughly-two-hundred-flip path. The sparse cases are now solvable
by refreshed layer-6 repair projection alone, so they are best understood as a
positive-vote supply problem rather than repeated-AND decoder brittleness.

The key ablation isolates the improvement:

```text
policy                 2-node sparse  4-node sparse
vote-quota first-fit   1186           504
vote-quota cheapest    664            189
```

This is not a decoder change, a residual change, or a layer-5 change. The gain
comes specifically from choosing the cheapest segment votes for each decoded
bit before projecting repairs into layer 6.

A compact segment-training summary tracks the efficiency terms directly:

```text
shape          dist   flips  rounds  blocks  avg quota  max quota  deficit     excess  min zero safety  entropy
2-node hard    5->0   10     1       10      1.00       1          5->0        0->0    64               .2074->.2070
2-node sparse  32->0  664    18      722     32.39      73         1224->0     0->0    78               .5750->.5762
4-node zero    0->0   0      0       0       0.00       0          0->0        0->0    36               .1905->.1905
4-node sparse  6->0   189    2       117     29.65      47         124->0      0->0    74               .5917->.5918
```

The selected-vote distribution by decoded bit summarizes how concentrated the
repairs are:

```text
shape          decoded bits fixed  avg flips/fixed bit  min  max  avg final target-one margin
2-node hard    5                   2.00                 2    2    96.94
2-node sparse  32                  20.75                0    33   7.88
4-node zero    0                   0.00                 0    0    81.81
4-node sparse  6                   31.50                10   76   37.62
```

The sparse solves are not merely barely crossing threshold. The 2-node sparse
case ends with modest positive margin, while 4-node sparse ends with a much
larger average target-one margin. Target-zero safety remains high in both
cases, so a separate post-solve margin pass is not yet justified for these
single-token probes.

Two derived fields make sparse efficiency easier to compare:

```text
shape          quota completion efficiency  vote efficiency
2-node hard    0.5000 decoded bits / flip   0.5000 deficit / flip
2-node sparse  0.0482 decoded bits / flip   1.8434 deficit / flip
4-node sparse  0.0317 decoded bits / flip   0.6561 deficit / flip
```

The sparse runs spend many flips, but most of the work is coherent vote supply:
false-positive excess stays at zero, all accepted blocks fully realize, and
mask entropy rises only modestly. The low decoded-bits-per-flip number is
expected for sparse targets with large initial vote deficits; the vote-efficiency
number is the better signal for whether candidate selection is wasting work.

The layer-6 flip-cost split separates requested condensed votes from the extra
mask flips needed to cross the layer-6 majority threshold:

```text
shape          layer6 flips  direct quota votes  majority support  conflicts  collateral  zero safety  selection
2-node hard    10            10                  0                 0          0           0            0
2-node sparse  664           663                 1                 0          0           0            0
4-node sparse  189           172                 17                0          0           0            0
```

For the current no-residual path, conflict resolution, collateral preservation,
target-zero safety preservation, and selection preservation are filters rather
than active flip-spending categories, so they stay at zero. The extra 4-node
sparse cost is therefore mostly ordinary majority-threshold support: its
requested condensed votes cost more support flips, not conflict repair or
selected-node preservation.

The support-cost histogram records how many selected quota votes cost `0..7`
mask flips, plus an overflow bucket, after compatible batch constraints share
already-requested support:

```text
shape          support-cost histogram
2-node hard    [0,10,0,0,0,0,0,0,0]
2-node sparse  [0,662,1,0,0,0,0,0,0]
4-node sparse  [0,156,15,1,0,0,0,0,0]
```

The segment quota builder now ranks candidate segment votes by projected
layer-6 realization cost before completing a decoded bit's threshold. Equal-cost
vote candidates are resolved by stable structural keys rather than allocator or
qsort behavior: output bit, condensed bit, and original encounter order. The
projected block queue uses the same idea after objective ranking: decoded bit,
final node, assignment count, and original encounter order are the final
tie-breakers. The batch selector still uses marginal cost when accepting
compatible repair candidates, so overlapping quota blocks can share support
flips once they have been proposed. This is a large improvement over first-fit
segment votes: the 4-node sparse run drops from `504` layer-6 flips to `189`,
and most selected quota votes become one-flip repairs.

The current projection implementation is being named around layer boundaries:

```text
project repair from layer L to layer L - 1
```

The strongest implemented case is still final-layer repair projection into the
penultimate layer. The internal projection helper now uses boundary wording, so
the next generalization step is mechanical: lift the segment repair collector
from "final to penultimate" into a layer-indexed collector once there is a
trusted teacher signal for deeper boundaries.

A deterministic shape-matrix smoke test now covers the intended experiment
surface without running an expensive full trainer sweep in unit tests:

```text
layers:    1, 2, 4, 8, 12
nodes:     1, 2, 4, 8
rotations: none, fixed, varied
init:      zero, sparse
targets:   dense, sparse, mixed
```

The smoke pass builds every combination, applies segment-condense decoding, and
checks decoder-objective and node-selection measurements. Targeted training
regressions remain separate so the test suite does not turn the full shape
matrix into a long-running benchmark.

A small multi-example clobber guard trains two inputs toward the same target
through the primary segment trainer and remeasures the first input after the
second update. This is not full multi-token training yet; it is a regression
that catches catastrophic one-example repair clobbering before adding richer
multi-example objectives.

A second clobber diagnostic now uses distinct targets. It trains one input
toward a dense target, then a second input toward a sparse target, and records
the before/after segment objective fields for each example:

```text
initial distance
final distance
flips
rounds
deficit before/after
excess before/after
zero safety
entropy before/after
stop reason
```

The test does not require the first example to remain solved after the second
distinct target update. That regression is the signal being measured. The
assertions only require each example's own bounded training step to be
well-formed and non-regressing.

The first tiny distinct-target probe was too weak: one side did not learn and
the other was already decoded. The current clobber diagnostic therefore chooses
deterministic adversarial seeds where both examples spend accepted flips and
improve their own target before measuring interference:

```text
pair                    A before/after/remeasured  B before/after  clobber  class
dense -> sparse         30 -> 9  -> 29             24 -> 2         20       distance-clobber
sparse -> dense          3 -> 0  -> 17             30 -> 13        17       catastrophic
disjoint sparse          3 -> 0  -> 1               3 -> 1          1       distance-clobber
mixed -> complement     22 -> 9  -> 54             55 -> 10        45       distance-clobber
```

This is the clearest multi-example signal so far: when both examples require
real mask flips, sequential distinct-target training clobbers the first
example immediately. The sparse-then-dense case is the strongest failure:
a solved sparse target becomes far from solved after dense-target training.
The disjoint-sparse case shows a smaller but still real distance-level
regression. Soft clobber, where distance stays fixed but target-one margin or
target-zero safety degrades, has not been the first failure mode in these
adversarial probes; distance-level interference appears first.

A separate bounded training matrix now sits next to the 360-shape smoke matrix:

```text
layers:    2, 4, 8, 12
nodes:     1, 2, 4, 8
rotations: none, varied
init:      zero, sparse
targets:   dense, sparse, mixed
```

Each of the `192` cases gets one small segment-condense quota repair projection
round. The matrix intentionally does not require convergence. It records the
same compact segment summary fields plus a stop reason:

```text
converged
rejected batch
max rounds
no repairs
no progress
```

This makes the matrix a diagnostic surface for interference and shape
sensitivity, not a long-running benchmark or a claim that every shape should
solve immediately.

The first bounded matrix run produced:

```text
runs:        192
improved:    38
converged:   24
rejected:    42
max-rounds: 126
no repairs:   0
no progress:  0
distance:  3607 -> 3522
deficit:   1109853 -> 1108946
flips:     1786
```

The rejected batches now carry a narrower reason and the rejected round records
the proposed/selected block counts, estimated and actual objective deltas,
before/after objective fields, requested and realized constraints, collateral
changes, selected/vote movement flags, and selected/best node movement. The
first rejection breakdown is:

```text
distance regression:          4
false-positive excess:        0
zero-safety regression:       0
target-one margin loss:       0
generic no objective delta:   0
realization mismatch:         0
batch interaction:            0
```

The former no-objective-delta bucket is now split:

```text
true no-op:               9
local activation only:    0
vote movement tied:       0
future-cost improvement:  0
```

A pre-batch predicted-effect filter now rejects candidates that cannot affect
the selected segment decoder surface before batch scoring. The filter asks
whether a candidate is expected to change segment vote counts, reduce
false-negative quota cost, or reduce a cheap weighted estimate of the best
completion cost. The bounded matrix moved many former true no-op rejections
into a more explicit stop reason:

```text
no-effective-candidate runs: 29
filtered no-effect candidates: 232
already satisfied:              0
no majority crossing:           0
unselected node:                0
irrelevant segment:           232
vote tied:                      0
```

The most important read is that repair candidates exist across the matrix
(`no repairs = 0`), but one small refreshed round is usually not enough to move
the decoded bit distance in deeper shapes. Many 8-layer and 12-layer cases
reduce vote deficit while distance stays flat, which means the segment objective
is seeing usable support movement below the decoded threshold. Rejected batches
cluster around zero-initialized dense/mixed cases. The narrower breakdown shows
that most of the former no-delta rejection was not safety loss, false-positive
excess, realization mismatch, tied vote movement, or hidden future-cost
improvement. Most of it was irrelevant segment work that can now be filtered
before batch scoring. The remaining true-noop bucket is smaller and should be
studied separately from the distance-regression bucket, which is the one to
study for true harmful proposals.

Replay scoring now runs around refreshed projected-repair batches. A replay
example stores an input/target pair, and a refreshed batch can be checked
against one or more replay examples before it is accepted. The replay record
tracks:

```text
replay examples scored
solved replay examples before the batch
solved replay examples made unsolved by the batch
replay distance / deficit / excess / margin / safety deltas
decoded-bit transition matrix
```

The first strict replay guard protects the solved example from the
`sparse -> dense` adversarial clobber case:

```text
without replay:
        A sparse: 3 -> 0 -> 17
        B dense: 30 -> 13

with strict replay:
        A sparse: 0 -> 0
        B dense final distance: 30
        replay-unsafe candidates filtered before batching
        stop reason: replay-blocked-all
```

Replay-safe candidate search now happens before batch selection. Each projected
candidate is temporarily applied to the layer, scored against the solved replay
examples, restored, and discarded if it would regress replay distance,
false-positive excess, target-one margin, or target-zero safety. The batch
selector therefore sees only candidates that are safe for the replay set.

That changes the interpretation of the sparse-to-dense adversarial probe. The
old prefix guard rejected and bisected batches after construction. The new
candidate filter reports:

```text
replay guard:
        A sparse: 0 -> 0
        B dense final distance: 30
        stop reason: replay-single-candidate-conflict
        replay rejected batches: 0
        bisections: 0
        replay-safe candidates: 30
        replay-unsafe candidates: 191
        best candidate unsafe: yes
        unsafe correct 0 -> false positive: 191
        unsafe correct 1 -> false negative: 6
        unsafe selected-node switches: 0
        unsafe best-decoded-node switches: 0
```

So the current result is not merely an ordering failure. For that tiny
adversarial pair, some replay-safe dense candidates exist, but the best
individual dense repair candidate damages the solved sparse replay example
under the strict guard. The unsafe-candidate decomposition also shows the main
failure mode clearly: dense-target repairs contaminate sparse target-zero bits.
This is not primarily selected-node churn, best-decoded-node churn, or old
repeated-AND-style target-one destruction.

Adding bounded top-K segment-vote alternatives before replay filtering and
replay-protected zero/one vote penalties slightly broadens the replay-safe
candidate pool for this pair, but does not yet create progress for the dense
example:

```text
top-K replay-aware vote generation:
        A sparse: 0 -> 0
        B dense final distance: 30
        replay-safe candidates: 33
        replay-unsafe candidates: 181
        best candidate unsafe: yes
        replay-safe strict-distance candidates: 0
        replay-safe deficit-helpful candidates: 33
        replay-safe quota-feasible decoded bits: 7
        replay-safe quota-blocked decoded bits: 29
        unsafe correct 0 -> false positive: 181
```

So the candidate set is broader, and the replay penalty moves some candidates
out of the unsafe bucket, but the strongest current dense repair is still in
direct conflict with sparse-zero preservation. Since this is
`replay-single-candidate-conflict` rather than `replay-blocked-all`, final-node
specialization is a next diagnostic rather than an immediate fallback. The safe
candidate split is useful: the safe candidates are not no-ops, but most decoded
bits still do not have enough safe votes to complete a dense quota. That points
to either replay-safe quota assembly/scaffolding or routing dense repairs
through a different final node.

Replay-safe quota-completion scaffolding is now a separate replay policy. When
enabled, replay training only batches decoded bits whose dense quota can be
completed from replay-safe votes:

```text
replay-safe quota scaffold:
        A sparse: 0 -> 0
        B dense distance: 30 -> 30
        B dense deficit: 88 -> 74
        replay-safe candidates: 33
        replay-unsafe candidates: 133
        replay-safe quota-feasible decoded bits: 7
        replay-safe quota-blocked decoded bits: 21
        replay-safe quota-complete candidates: 31
        replay-safe quota-incomplete candidates: 2
        direct protected-zero hits: 0
        replay-reserved zero votes: 0
        realization-collateral false-positive candidates: 133
        intended mask flips scored for replay sensitivity: 235
        replay-safe mask flips: 149
        false-positive-sensitive mask flips: 83
        false-negative-sensitive mask flips: 3
        margin/safety weakening mask flips: 0
        replay-collateral cost: 88064
        decomposed candidates: 166
        unsafe flip fraction: 86 / 235
        one-bad-flip candidates: 86
        mostly-unsafe multi-flip candidates: 8
        alternate unsafe flip fraction: 86 / 235
        alternate replay-collateral cost: 88064
        alternate-better candidates: 0
        stop reason: replay-capacity-conflict
```

This is the first non-clobbering dense movement in the adversarial
`sparse -> dense` pair. It does not cross a decoded threshold yet, but it proves
that safe quota-complete scaffolding can add useful dense support while the
solved sparse replay example remains solved. The remaining stop reason is still
real: after the safe quota-complete moves, the best remaining dense repair is
unsafe for the sparse replay example.

Repeated one-round scaffold tracking shows the mechanism plateaus quickly on
this 2-node state:

```text
round  A distance  B distance  B deficit  safe feasible  safe blocked  direct zero hit  collateral fp  unsafe flip fraction  alt unsafe fraction  alt better  stop
0      0 -> 0      30 -> 30    88 -> 76   6              5             0                39             38 / 84              38 / 84             0           max-rounds
1      0 -> 0      30 -> 30    76 -> 74   1              8             0                46             24 / 77              24 / 77             0           max-rounds
2      0 -> 0      30 -> 30    74 -> 74   0              8             0                48             24 / 74              24 / 74             0           replay-capacity-conflict
```

So the current replay-safe scaffold is real but shallow: it can harvest the
few quota-complete safe dense votes, then safe quota feasibility collapses to
zero before any decoded threshold crosses. For this exact shared selected-node
surface, the result looks capacity-limited rather than merely under-iterated.

The candidate class split is useful. Direct protected-zero hits stay at zero,
but every unsafe candidate falls into the realization-collateral
false-positive bucket. The current sparse-to-dense failure is therefore not
choosing segment votes that directly sit on A's protected zero surface.
Instead, dense repairs that are useful for B change masks in a way that
collateralizes into A false positives. The mask-flip attribution makes that
discrete: individual intended mask flips can now be marked safe,
false-positive-sensitive, false-negative-sensitive, or margin/safety weakening,
and their penalties are folded into candidate effective cost before sorting.
The current repair decomposition says the path is usually not intrinsically
unsafe: `86` candidates have exactly one bad flip, while only `8` multi-flip
candidates are mostly unsafe. However, the alternate realization search does
not find a lower-collateral path for this state (`alt-better = 0`), so those
one-bad-flip repairs do not yet have an available safer input-vote
substitution under the current layer-6 vote geometry.

Bad-flip identity tracking now records each replay-sensitive mask flip by:

```text
layer
node
input index
mask bit index
desired new mask value
bad-use frequency
harmless-use frequency
A decoded bits damaged
B decoded bit helped
```

On the same sparse-to-dense scaffold run, the `86` bad flip occurrences collapse
to a much smaller set:

```text
unique bad flip identities: 15
top bad flip:               L0 N1 I0 bit2 -> 1
top bad frequency:          6
top harmless uses:          0
top A damaged bits:         6
top B helped decoded bit:   2
```

That suggests the replay conflict is not completely diffuse. A small set of
mask-bit identities is reused across many unsafe dense repairs, and the hottest
one is purely bad in this diagnostic window. This makes a mask-bit taboo or
protected-bit penalty a promising next mechanism: after solving replay example
A, mark flips that create A false positives as replay-taboo, then build B quota
candidates around non-taboo realizations instead of discovering the conflict
only after a candidate has already been assembled.

A first taboo-aware quota-construction pass now exists. During segment-vote
candidate generation, each candidate's required layer-6 mask flips are scored
against the solved replay example, and false-positive-sensitive flips add a
`replay_taboo_flip_penalty` before the quota votes are sorted. The sparse-to-
dense adversarial probe shows that this earlier penalty is measurable but not
yet sufficient:

```text
taboo-penalized vote candidates: 222
taboo-sensitive mask flips:      222
taboo penalty total:             227328

overall replay result:
        A sparse: 0 -> 0
        B dense distance: 30 -> 30
        B dense deficit: 88 -> 74
        safe-quota-feasible decoded bits: 7
        safe-quota-blocked decoded bits: 21
        stop reason: replay-capacity-conflict
```

So the taboo signal is not noise, but a soft cost penalty alone does not unlock
new replay-safe quota completions on this state. The next logical replay step is
therefore narrower than global routing but stronger than soft taboo costing:
either reject taboo-dependent votes outright, or test minus-one-bad-flip /
compensating-repair transactions for the many candidates whose unsafe path is
still dominated by a single bad mask flip.

Both of those follow-ups have now been tested.

First, taboo-dependent segment votes are rejected outright during quota
assembly. A vote candidate is now dropped before sorting if its realization path
contains any replay-taboo mask flip, where the taboo map is keyed by:

```text
layer
node
input index
mask bit
desired value
```

The map is built from solved replay examples by testing each possible mask flip
at the current layer and marking:

```text
correct 0 -> false positive  => zero-protection taboo
correct 1 -> false negative  => one-protection taboo
```

That moves the replay filter earlier than full candidate scoring: quota vote
generation now consults the taboo map before assembling repair blocks. On the
same sparse-to-dense probe, the map-based strict rejection changes the replay
surface materially:

```text
A sparse: 0 -> 0
B dense distance: 30 -> 30
B dense deficit: 88 -> 70
safe-quota-feasible decoded bits: 11
safe-quota-blocked decoded bits: 18
false-positive-sensitive mask flips among accepted candidates: 0
all accepted candidate mask flips replay-safe: 295 / 295
stop reason: rejected-local-only
```

So hard taboo rejection does find a cleaner replay-safe path. It removes the old
accepted-candidate false-positive contamination entirely and improves B's replay-
safe deficit reduction (`88 -> 70`), but it still does not cross a decoded
threshold on this tiny sparse-to-dense pair.

Second, a minus-one-bad-flip diagnostic now runs on those taboo-rejected vote
candidates. For each taboo-rejected candidate, the trainer replays the intended
layer-6 mask flips except the single replay-clobbering false-positive flip, then
scores the partial realization. Under the earlier ad hoc taboo scoring, many of
those stripped candidates remained replay-safe and some were still deficit-
helpful. Under the explicit taboo-map rejection, the same diagnostic becomes:

```text
taboo-rejected vote candidates: 266
minus-one-bad candidates tested: 222
minus-one-bad replay-safe:       222
minus-one-bad deficit-helpful:     0
minus-one-bad strict-helpful:      0
```

So the explicit taboo map is stronger than the earlier heuristic. It removes the
old one-bad-flip partials from the candidate frontier entirely and leaves a
smaller, cleaner replay-safe repair surface. The current trainer still does not
cross a decoded threshold, but the remaining failure now looks less like hidden
collateral and more like replay-safe quota insufficiency. The next replay
mechanism is therefore sharper than before:

```text
taboo-map-aware scaffold completion or compensating transactions
```

That is, keep the explicit taboo map and strict taboo rejection, then either:

```text
1. allow taboo-free partial realizations to accumulate until a quota completes
2. or try compensating A-repair transactions for taboo-dependent repairs
```

The first option now has a cleaner foundation, because accepted candidates are
already taboo-free by construction. The second remains the heavier fallback if
that taboo-free scaffold still cannot cross thresholds.

The next audit added the current-example `B` side explicitly, plus a
three-surface propagation trace. For replay-safe sparse-to-dense batches, the
trainer records:

- decoded transitions on `B`,
- per-candidate net sign for `B`,
- batch-level cancel causes on `B`,
- and how a replay-safe quota moves across:
  1. the penultimate selected segment surface,
  2. the final selected activation surface,
  3. the final decoded vocabulary surface.

That work exposed a sharper mismatch: the old replay-safe path was still using
*local quota completion* as its trainer unit. It could count a decoded bit as
"quota complete" when enough local candidates had been selected, even if those
local changes collapsed onto overlapping final votes and never formed a usable
final-surface quota.

The replay-safe path now treats the unit as a **final-surface quota**, not a
local quota:

```text
choose desired final segment votes for decoded bit k
then build replay-safe local changes for those votes
```

So replay-safe feasibility and completion are counted in terms of **distinct
final output votes per decoded bit**, and quota assembly will not spend more
than one candidate on the same `(decoded bit, final output vote)` pair.

Under that stricter final-surface accounting, the focused sparse-to-dense
replay result becomes:

```text
B transition audit:
        fn->correct:        0
        correct1->fn:       0
        fp->correct:        0
        correct0->fp:       0
        unchanged-wrong:    0
        unchanged-correct:  0

B per-candidate net:
        positive: 0
        zero:     0
        negative: 0

B batch-cancel causes:
        target-one-loss:        0
        target-zero-loss:       0
        selected-node-change:   0
        duplicate/overlap:      0

safe quota split:
        feasible:             0
        selected:             0
        accepted:             0
        completed:            0
        local-realized:       0
        local-crossed:        0
        final-vote-reached:   0
        final-vote-changed:   0
        final-vote-crossed:   0
        decoded-vote-changed: 0
        decoded-crossed:      0
```

and the replay round stops immediately with:

```text
stop: replay-capacity-conflict
B deficit: 88 -> 88
safe-quota-feasible: 0
safe-quota-blocked: 7
```

So the previous `11` "feasible" replay-safe bits were not true final-surface
quota opportunities. They were artifacts of local-unit accounting. Once the
trainer is forced to assemble distinct final votes, the focused sparse-to-dense
replay probe has **no replay-safe final-surface-complete quota at all**.

That is a better diagnosis:

```text
not:
        replay-safe local work exists but washes out later

instead:
        there is no replay-safe final-surface quota to assemble on the shared
        selected-node surface in this state
```

The three-surface trace is still useful, but now it should be interpreted as a
diagnostic for candidate propagation quality, not as the definition of quota
completion. For each safe-quota trace entry, the trainer records:

1. the penultimate selected segment surface,
2. the final selected activation surface before decode,
3. the final decoded vocabulary surface.

The aggregate counters are:

```text
safe-quota-local-realized
safe-quota-local-crossed
safe-quota-final-vote-reached
safe-quota-final-vote-changed
safe-quota-final-vote-crossed
safe-quota-decoded-vote-changed
safe-quota-decoded-threshold-crossed
```

with failure buckets:

```text
lost-at-final-majority
lost-at-final-rotation-or-expansion
lost-at-unselected-final-node
lost-at-segment-vote-margin
lost-due-to-duplicate-final-segment
```

Before the final-surface quota change, a representative trace looked like:

```text
bit 9 target 1
penultimate node 0->0 ones 0->2 threshold 2 crossed 1
final node 0->0 condensed 0->0 threshold 2 changed 0 crossed 0
segment 0->0 threshold 2 changed 0
decode 0->0 distance 1->1
```

That trace is still correct as a *local propagation* observation, but it is no
longer enough to claim replay-safe quota feasibility. The trainer now requires
final-surface quota completeness, so duplicate-aware / overlap-aware
final-surface planning is part of the trainer unit itself rather than merely a
post-hoc diagnostic.

A first reversible final-node basis-transform diagnostic now exists. It applies
simple pairwise node-bit transforms only at decode time:

```text
identity
odd-bit swap
even-bit swap
half swap
```

For the focused sparse-to-dense replay state, the result is:

```text
identity:  A distance 0, B distance 30, B deficit 88
odd-swap:  A distance 1, B distance 31, B deficit 87
even-swap: A distance 1, B distance 31, B deficit 87
half-swap: A distance 0, B distance 30, B deficit 88
```

So the small reversible basis transforms tested so far do **not** expose an
obviously better replay surface:

```text
odd/even swap:
        slightly help B deficit
        but break solved A

half swap:
        effectively neutral
```

That is a useful negative result. It suggests the replay wall is not removed by
a trivial adjacent-node bit-lane redistribution. If routing or basis changes
help here, they will likely need to be more structured than a simple odd/even
pairwise swap.

A small sparse-to-dense
replay scaffold matrix repeats the same diagnostic across node count,
initialization, and rotations:

```text
nodes  init    rotation  A after B  B distance  B deficit  safe feasible  safe blocked  reserved zero  max feasible  rounds  crossings  selected  best dense  best polarity
2      zero    none      0          30          90         0              11            0              0             1       0          0         0           normal
2      zero    varied    1          32          94         0              0             0              0             0       0          0         0           complement
2      sparse  none      0          30          82         0              10            0              2             2       0          0         0           normal
2      sparse  varied    1          29          73         0              0             0              0             0       0          0         0           normal
4      zero    none      0          30          90         0              11            0              0             1       0          0         0           normal
4      zero    varied    1          32          94         0              0             0              0             0       0          0         0           complement
4      sparse  none      1          31          89         0              0             0              0             0       0          1         1           complement
4      sparse  varied    0          31          75         0              0             0              0             1       0          0         2           normal
8      zero    none      0          30          90         0              11            0              0             1       0          0         0           normal
8      zero    varied    1          32          94         0              0             0              0             0       0          0         0           complement
8      sparse  none      0          30          90         0              12            0              0             1       0          1         1           normal
8      sparse  varied    1          32          90         0              0             0              0             0       0          6         6           complement
```

The modest success condition was:

```text
A remains solved and B distance drops below 30
```

No matrix row met that condition. The only `B < 30` row is `2 sparse varied`,
but `A` is no longer solved. The rows where `A` remains solved keep `B` at
distance `30` or worse, and safe quota feasibility does not grow. This supports
the current read: replay-safe scaffolding can harvest isolated compatible
votes, but the shared selected-node surface does not expose enough compatible
dense-positive capacity for this sparse-to-dense pair.

A direct final-layer replay oracle probe is now in place as a stricter check on
whether the bottleneck is only a penultimate projection problem. On the focused
sparse-to-dense replay state it produces:

```text
segment replay final-layer oracle:
        A 0 -> 1
        B 30 -> 30
        B deficit 88 -> 64
        safe steps: 3
        oracle flips: 16
        replay-safe: no
```

So direct final-layer repairs can reduce dense `B`'s deficit, but they still do
not move `B`'s decoded distance and they clobber solved sparse `A`. That rules
out the simpler explanation that only penultimate repair projection is at
fault.

A follow-up transaction-style probe then allows exactly the next thing the
replay diagnostics suggested: let `B` temporarily clobber `A`, then run
final-layer repairs for `A`, and keep the transaction only if `A` is restored
while `B` stays better. The focused result is:

```text
segment replay final-layer transaction:
        A 0 -> 1 -> 1
        A false-positive excess 0 -> 2 -> 2
        B 30 -> 29 -> 29
        B deficit 88 -> 57 -> 57
        B steps: 4
        A repair steps: 16
        B flips: 64
        A repair flips: 0
        accepted: no
```

This is another useful boundary result:

```text
B can improve under direct final-layer oracle pressure,
but the same final-layer oracle does not restore A afterward.
```

So the replay wall is no longer just "no strict replay-safe path exists." It is
also:

```text
the simple clobber-then-repair transaction fails on the shared final surface
with the current oracle
```

That pushes the next mechanisms toward either more expressive transactional
repair, or some form of explicit capacity separation.

To make that failure less ambiguous, there is now a focused restore-failure
diagnostic on the exact post-clobber state. It asks a narrower question:

```text
after B has already clobbered A,
what zero-veto final-layer repairs exist for A's false positives?
```

On the same focused replay probe, the answer is:

```text
segment replay restore failure free:
        A 1, fp-excess 2, candidates 1, useful 0, veto 1, rejected 1
        reason: no-useful-repairs

segment replay restore failure preserve-B:
        A 1, fp-excess 2, candidates 1, useful 0, veto 1, rejected 1
        reason: no-useful-repairs
```

The per-bit zero-side audit now shows that the restore path does exist:

```text
offending bit 4:
        target: 0
        decoded: 1
        ones: 4
        threshold: 3
        max ones for zero: 2
        excess: 2
        segment votes currently one: 4
        clearable segment votes: 4
        candidate final output bits: 4
        projected condensed bits: 2
        direct candidate changed: yes
        candidate mask flips: 2
        accepted: no
        reason: no-useful-repairs
```

So the transaction failure is no longer "A has no zero-side restore candidate."
The explicit target-zero restore candidate exists, it projects into two
condensed-bit changes and two final-layer mask flips, and it still fails.

The restore/transaction split is now sharper than that first reading suggested.
There are two distinct transaction behaviors:

```text
multi-step transaction:
        A 0 -> 1 -> 1
        B 30 -> 29 -> 29
        accepted: no

single-step transaction:
        A 0 -> 0 -> 0
        B 30 -> 30 -> 30
        B deficit 88 -> 80 -> 80
        A restored: yes
        B preserved: yes
        accepted: yes
```

So the conflict is not "any dense B move immediately destroys sparse A." A
single final-layer oracle step for `B` is still reversible and replay-safe. The
failure appears once the denser multi-step `B` update pushes the final layer
into a state where `A` has a false-positive bit and the current zero-side
repair collector exposes no restore proposal.

The restore-failure probe has also been split into:

```text
A restore without preserving B
A restore while preserving B
```

and on this focused state both modes currently agree:

```text
segment replay restore failure free:
        A 1, fp-excess 2, candidates 1, useful 0, veto 1, rejected 1
        reason: no-useful-repairs

segment replay restore failure preserve-B:
        A 1, fp-excess 2, candidates 1, useful 0, veto 1, rejected 1
        reason: no-useful-repairs
```

The per-bit zero-side audit now makes that more concrete:

```text
offending bit 4:
        target: 0
        decoded: 1
        ones: 4
        threshold: 3
        max ones for zero: 2
        excess: 2
        segment votes currently one: 4
        clearable segment votes: 4
        candidate final output bits: 4
        projected condensed bits: 2
        direct candidate changed: yes
        candidate mask flips: 2
        accepted: no
        reason: no-useful-repairs
```

The next three-surface restore trace explains why that explicit candidate still
does not help `A`:

```text
A-fp bit 4:
        target: 0
        decoded: 1
        ones: 4
        threshold: 3
        max ones for zero: 2
        excess: 2
        segment ones: 4
        clearable: 4
        final output bits: 4
        projected condensed bits: 2
        direct candidate changed: yes
        mask flips: 2
        candidate votes: 4
        B-safe: no
        flips: 2
        final node: 0 -> 1
        condensed ones: 2 -> 2
        segment ones: 4 -> 4
        decoded: 1 -> 1
        accepted: no
        reason: no-useful-repairs
        propagation: selected-node-mismatch
```

The corresponding clear-vote trace shows the intended condensed clears, but
those clears never reach the decoder surface because selection moves to a
different final node:

```text
clear-vote bit 4 seg 0 out-bit 4:
        raw segment 1 -> 1
        condensed bit 4: 1 -> 0
        mask flips: 2
        majority: 4 -> 4
        threshold: 3
        cleared: no

clear-vote bit 4 seg 1 out-bit 68:
        raw segment 1 -> 1
        condensed bit 68: 1 -> 0
        mask flips: 2
        majority: 4 -> 4
        threshold: 3
        cleared: no

clear-vote bit 4 seg 2 out-bit 132:
        raw segment 1 -> 1
        condensed bit 4: 1 -> 0
        mask flips: 2
        majority: 4 -> 4
        threshold: 3
        cleared: no

clear-vote bit 4 seg 3 out-bit 196:
        raw segment 1 -> 1
        condensed bit 68: 1 -> 0
        mask flips: 2
        majority: 4 -> 4
        threshold: 3
        cleared: no
```

So the current diagnosis has moved again:

```text
not:
        no zero-side restore proposal exists

not:
        zero-side proposal exists but only fails because B must be preserved

not:
        zero-side proposal exists but cannot be projected into mask flips

instead:
        the explicit zero-side restore candidate exists and does project into
        final-layer mask flips, but selection jumps from final node 0 to node 1
        before the intended clears reach the decoder surface
```

That rules out segment-vote enumeration failure and mask-projection failure for
this focused bit. The next bottleneck is selection-aware restore scoring or a
forced-node restore path on the final surface, not further debugging of the
zero-side quota itself.

The next selection-aware restore probe then pins the `A` restore candidate to
the same final node it is trying to repair instead of judging usefulness only
after global final-node selection has already moved. That changes the restore
reading in an important way:

```text
segment replay restore failure free:
        A 1, fp-excess 2, candidates 1, useful 1, veto 1, rejected 0
        reason: none

segment replay restore failure preserve-B:
        A 1, fp-excess 2, candidates 1, useful 1, veto 1, rejected 0
        reason: no-b-safe-repairs
```

So the explicit target-zero restore candidate is now confirmed to be:

```text
selection-aware useful for A on the node it targets
```

The remaining problem is no longer "the restore candidate does not help A at
all." It is the sharper split:

```text
1. pinned-node restore improves A's decoder objective on the intended node
2. preserving B still blocks accepting that repair
3. global selection still moves from node 0 to node 1, so the real decoder
   surface does not inherit the local A restore
```

The per-bit restore trace now also records the forced-node objective directly:

```text
A-fp bit 4:
        forced node-0 objective: distance/fp-excess 1/2 -> 0/0
        global selected node: 0 -> 1
        globally decoded bit: 1 -> 1
```

So the current restore candidate is no longer just "locally promising." On the
originally selected node it really does repair `A`; the repair is lost only
because final-node selection moves to another unrepaired node.

A dedicated contender-aware restore diagnostic now separates that next question
cleanly instead of overloading the single-node restore trace. It starts from
the exact post-`B` clobber state, applies the original node-0 zero repair, and
if selection moves to node 1 it applies the same target-zero repair there too.
On the focused 2-node replay case the result is:

```text
segment replay contender restore free:
        bit 4
        A after B: 1 / fp-excess 2
        node 0 -> 1 after first restore
        forced node-0 objective: 1/2 -> 0/0
        contender node: 1
        after repairing both nodes:
                A 0 / fp-excess 0
                B 30 / deficit 60
                flips: 4
                useful: yes
                B-safe: yes
                accepted: yes

segment replay contender restore preserve-B:
        bit 4
        after repairing both nodes:
                A 0 / fp-excess 0
                B 30 / deficit 60
                flips: 4
                useful: yes
                B-safe: no
                accepted: no
```

That changes the diagnosis again:

```text
contender-aware target-zero restoration is sufficient to restore A
under normal selection on this case
```

The remaining replay conflict is no longer selection alone. It is the tighter
multi-example tradeoff:

```text
repairing all contender nodes restores A,
but doing so still degrades B's replay objective enough to fail the strict
preserve-B acceptance rule
```

So the next trainer should be contender-aware first, then replay-aware within
that contender set. The selected-node mismatch is now a solved subproblem for
this focused case; the remaining question is how to preserve more of `B` while
repairing every final node that can become selected.

To make that tradeoff explicit, contender-aware restore now reports transaction
retention levels instead of only a single `b-safe` bit:

```text
B_STRICT_PRESERVED:
        distance_after_restore <= distance_after_B
        and deficit_after_restore <= deficit_after_B

B_DISTANCE_PRESERVED:
        distance_after_restore <= distance_after_B

B_PROGRESS_PRESERVED:
        distance_after_restore <= distance_before_transaction
        and deficit_after_restore < deficit_before_transaction

B_NO_REGRESSION:
        distance_after_restore <= distance_before_transaction
        and deficit_after_restore <= deficit_before_transaction
```

On the focused contender restore:

```text
strict preserved:   no
distance preserved: no
progress preserved: yes
no regression:      yes
```

That is enough to justify a scaffold-style transaction mode. A dedicated
transaction scaffold loop now applies one bounded `B` final-layer oracle step,
repairs `A` with contender-aware restore if needed, and accepts the round if:

```text
A returns solved
and
B does not regress relative to the pre-transaction state
and
B deficit improves
```

On the same focused sparse-to-dense case:

```text
segment replay transaction scaffold:
        rounds accepted / attempted: 3 / 4
        A: 0/0 -> 0/0
        B: 30/88 -> 30/64
        frontier min deficit: 1 -> 1
        deficit-1 bits:        1 -> 13
        deficit-2 bits:        0 -> 0
        cheapest completion:  1 -> 1
        top-4 completion:     10 -> 4
        B flips: 48
        restore flips: 0
        strict/distance/progress/no-regression rounds: 3 / 3 / 3 / 3
        frontier-improved rounds: 3
        distance improved: no
        deficit improved: yes
        frontier improved: yes
        A solved: yes
```

So the current replay path is now more specific:

```text
strict replay-safe final-surface repair: none
contender-aware restore: fixes A
transaction scaffold: keeps A solved while improving B's completion frontier
```

The next refinement is no longer "can we restore A?" It is "which contender
restore choice preserves the most of B's dense-support gain?" The contender
enumeration now makes that explicit for the offending `A` bit and contender
node:

```text
contender clear set 0: votes {0,1}  -> A 0/0, B 30/60   best
contender clear set 1: votes {0,2}  -> A 1/0, B 31/59
contender clear set 2: votes {0,3}  -> A 0/0, B 30/60
contender clear set 3: votes {1,2}  -> A 0/0, B 30/60
contender clear set 4: votes {1,3}  -> A 1/0, B 31/59
```

So, on this focused case, the B-preserving path does **not** reopen at
`29/57` or `29/58`. Different contender clear sets do matter, but all
A-restoring clear sets still converge near `B 30/60`. The scaffold loop now
uses frontier-aware acceptance, so accepted rounds are the ones that make dense
bits cheaper to complete, not just the ones that shave aggregate deficit.

The round-by-round trace now makes the phase split explicit:

```text
round 0: B 30/88 -> 30/80, deficit-1 bits 1 -> 5, accepted
round 1: B 30/80 -> 30/72, deficit-1 bits 5 -> 9, accepted
round 2: B 30/72 -> 30/64, deficit-1 bits 9 -> 13, accepted
round 3: transaction-aware completion search attempted, rejected, state stays 30/64
```

So the strict replay-safe scaffold is currently a **frontier builder**: it
accumulates many deficit-1 bits without yet finishing one. The next useful
policy is therefore not another restore variant; it is a finish-nearest-bit
phase that explicitly tries to convert one of those deficit-1 bits into a real
decoded distance drop.

The completion phase now behaves opportunistically:

```text
1. enumerate quota-size-1 completion candidates
2. score them after replay restore
3. keep the best survivor if any exist
4. otherwise leave the scaffold state unchanged
```

On this focused sparse-to-dense case, that search still finds no
transaction-surviving completion candidate. So the frontier mechanics are now
stable, and the next bottleneck is completion candidate quality after
replay-aware scoring rather than scaffold scheduling.

The finish rejection buckets narrow that further:

```text
finish candidates:                  54
finish candidates that complete B:  27
pre-restore distance helpful:       27
pre-restore deficit helpful:        27
clobbers A:                         54
create A correct0 -> FP:            54
create A correct1 -> FN:            27
switch A selected node:             27
contender restore available:         0
restore erases B:                    0
restore fails:                      54
restores A:                          0
post-restore distance preserved:     0
post-restore progress preserved:     0
rejected no pre-restore gain:       27
rejected A not restored:             0
rejected B gain lost:                0
```

So the current completion failure is not "restore gives the B gain back." It is
earlier and simpler:

```text
every completion candidate disturbs A,
and none of those disturbed-A candidates are restored by the current
contender-aware path.
```

The new per-candidate finish trace makes that shape concrete. There are two
distinct finish families:

```text
family 1:
        B 30/64 -> 29/63 before restore
        A 0/0 -> 1/2/0
        A selected node stays 0
        damage is correct0 -> false positive only
        restore contender unavailable
        rejected: restore-failed

family 2:
        B 30/64 -> 30/64 before restore
        A 0/0 -> 2/2/1
        A selected node switches 0 -> 1
        damage is correct0 -> false positive plus correct1 -> false negative
        rejected: no-pre-restore-gain
```

So the current completion refinement should focus on the **A-clobbering
completion surface** itself: the useful finish candidates do complete `B`
before restore, but they all leave `A` in a post-completion state that the
current contender-aware restore cannot enter.

The family-1 post-finish audit sharpens that again. For every useful
`B 30/64 -> 29/63` finish candidate:

```text
family1-restore:
        candidate repairs:      1
        useful repairs:         1
        target-zero repairs:    1
        rejected repairs:       0
        no-flip reason:         none
        damaged A bit:          ones 4, threshold 3, max-zero 2, excess 2
        clearable votes:        4
        mask flips:             2
```

So family-1 finish candidates are **not** outside `A`'s zero-side restore basin
in general. They are outside the **current contender-aware restore basin**
specifically. The zero-side repair exists from the post-finish state; the
missing step is a restore mode that operates on the post-finish damage set
rather than only on the earlier contender-switch pattern.

That damage-set restore has now been tested directly on the focused case. For
each family-1 finish candidate, the trainer:

```text
1. applies the B finish
2. collects all A correct0 -> false-positive bits
3. builds explicit negative segment quota repairs for those bits
4. applies the restore batch on the post-finish selected node
5. keeps it only if A is solved and B stays distance-improved or progress-preserved
```

On this case the family-1 restore batch still does not survive:

```text
family1-restore:
        candidate repairs:   1
        useful repairs:      1
        target-zero repairs: 1
        rejected repairs:    0
        damage-set bits:     1
        node:                0 -> 0
        damage-set restore available: yes
        excess reduced:      no
        restored A:          yes
        distance preserved:  no
        progress preserved:  no
```

So the next bottleneck is narrower still, and different:

```text
family-1 finish candidates create A damage that is locally repairable on the
zero-side and globally restorable on the selected surface, but every A-restoring
clear set still gives back the exact B threshold crossing.
```

The next refinement tried a B-aware clear-set search on that same family-1
damage:

```text
for each family-1 finish candidate:
        enumerate clearable segment-vote sets for the damaged A false-positive bit
        apply each clear set on the post-finish selected node
        score A restore first, then B distance / deficit retention
```

On the focused sparse-to-dense case this still finds no accepted restore set.
The per-clear-set split now makes the shape explicit:

```text
restoring clear sets:
        exist
        preserve selected node 0 -> 0
        restore A to 0/0
        return B to 30/80

non-restoring clear sets:
        switch node 0 -> 1
        leave A unsolved
        worsen B to 31/81
```

So the state is now:

```text
family-1 finish:
        real B distance gain exists

family-1 A zero-side repair:
        candidate exists and restores A

family-1 B-aware clear-set restore:
        no B-distance-compatible surviving set yet
```

At this point the replay path was also moved under a focused memory-safety
harness. A dedicated test entrypoint now runs only the replay diagnostics:

```text
./build/test-itty-feed-model --focus-replay
ASAN_OPTIONS=detect_leaks=0 ./build-asan-clang/test-itty-feed-model --focus-replay
```

This exposed a real use-after-free in the finish damage-set restore path. ASan
reported the fault in the layer snapshot helpers used by
`itty_feed_model_apply_finish_damage_set_restore_current_state()`. The bug was
an ownership mistake:

```text
itty_feed_model_restore_layer_state_snapshot()
        consumes and frees the snapshot shell

caller
        still treated best_snapshot/base_snapshot as live
        and later freed them again
```

The fix was to null those pointers immediately after restore consumption. After
that change, the focused replay path runs cleanly under ASan in the narrowed
harness.

Importantly, the corruption fix did **not** change the replay conclusions. The
focused result after the fix is still:

```text
A: 0/0 -> 0/0
B: 30/88 -> 30/80
deficit-1 bits: 1 -> 5
top-4 completion cost: 10 -> 4
```

and family-1 finish candidates still behave the same way:

```text
B finish before restore:
        30/80 -> 29/79

best family-1 restore:
        A -> 0/0
        B -> 30/80
```

So the memory-safety repair improves confidence in the diagnostics, but it does
not reopen a `B` distance-preserving finish path on this case.

The focused case is therefore best labeled:

```text
family-1 finish:
        A-fixable
        B-distance-incompatible
        on the current 2-node sparse -> dense shared final surface
```

This is no longer a restore-validity problem. It is a local final-surface
capacity conflict:

```text
B can be completed
A can be restored
but the same local support cannot currently satisfy both at once
```

That makes the next useful diagnostic an overlap audit between:

```text
B finish vote(s)
A restore clear set(s)
```

at the levels of final output vote, condensed bit, and realized mask flip. That
audit is now in place on the focused replay harness.

The useful result is that the family-1 contradiction is **not always exact**.
For several `B 30/80 -> 29/79` family-1 finish candidates, the best
`A -> 0/0` restoring clear set shows:

```text
overlap final output bit:   0
overlap condensed bit:      0
overlap mask flip location: 0
overlap mask flip dir:      0
```

while still returning `B` to `30/80`. Other family-1 candidates do show some
overlap, but the zero-overlap cases matter more because they rule out the
simplest explanation:

```text
B needs vote v = 1
A needs the same vote v = 0
```

The current focused diagnosis is therefore sharper:

```text
family-1 finish:
        A-restoring clear sets exist
        but no A-restoring clear set preserves B's threshold crossing

conflict:
        not always an exact same-vote contradiction
        sometimes a broader shared threshold/support conflict
        on the selected final surface
```

That pushes the next useful mechanism away from single-vote overlap debugging
and toward a joint finish-and-restore transaction with replacement votes or
explicit extra capacity/routing.

That means the next restore mechanism should be selection-aware at acceptance
time as well, not only at candidate generation time. The current failure is now
best read as:

```text
selection-aware A restore exists,
and it can restore A globally,
but it is not replay-compatible with B's distance crossing
on the current shared final surface
```

A node-specialization diagnostic on the same state does not show an alternate
final node that is already better for dense B:

```text
A sparse:
        selected node: 0
        best decoded-distance node: 0
        selected distance: 0
        best distance: 0
        popcount gap: 8

B dense:
        selected node: 0
        best decoded-distance node: 0
        selected distance: 30
        best distance: 30
        best deficit node: 0
        selected deficit: 74
        best deficit: 74
        popcount gap: 8
```

So final-node specialization remains a plausible future capacity mechanism, but
this tiny 2-node replay pair does not yet expose an unused better node. The
immediate replay path is replay-safe scaffold accumulation: keep grouping safe
votes by decoded bit, track whether safe quota feasibility grows over rounds,
and only escalate to explicit routing once a wider shape shows

### Focused Width Sweep

The focused replay harness was then extended into a transaction-capacity matrix
over:

```text
nodes:      2, 4, 8
init:       zero, sparse
rotation:   none, varied
```

Each row records:

```text
A solved after scaffold?
B scaffold distance / deficit / frontier
best family-1 finish distance
best restore distance
best replacement distance
replacement candidate count / safe count / survivor flag
selected node / best dense node / polarity
```

Representative rows:

```text
2 zero none:
        A solved yes
        B scaffold 30 / 34, d1 28, top4 4
        best finish 29
        best restore 30
        replacement survivors no

2 zero varied:
        A solved no
        B scaffold 30 / 30, d1 30, top4 4
        best finish 29
        best restore 31
        best replacement 31
        replacement survivors yes

4 zero none:
        A solved yes
        B scaffold 30 / 58, d1 16, top4 4
        best finish 29
        best restore 30
        replacement survivors no

8 sparse none:
        A solved yes
        B scaffold 30 / 76, d1 7, top4 4
        best finish 30
        best restore 29
        best replacement 29
        replacement survivors yes
```

This changes the interpretation of the earlier `2-node` result in an important
way:

```text
more width can expose replacement survivors
but not cleanly or uniformly
```

Specifically:

- the original `2 zero none` focused case still behaves like a real shared
  selected-surface capacity wall:
  scaffold helps `B`, finish exists, restore exists, but no replacement
  survivor remains while `A` stays solved.
- some wider or rotated configurations do expose replacement survivors.
- several of those survivor rows do **not** preserve the sparse replay example,
  so width alone is not yet a complete replay solution.

The current replay picture is therefore:

```text
2-node shared selected final surface:
        capacity-limited for sparse -> dense completion

wider models:
        sometimes expose alternate compatible surfaces
        but replay stability is configuration-dependent
```

So the next question is no longer:

```text
is extra capacity useful at all?
```

It is:

```text
how should width be routed or regularized so that extra capacity behaves like
stable per-example memory instead of another unstable shared surface?
```

### Forced-Route Replay Diagnostic

To answer that, the focused harness now also reports forced-node decoder
objectives for the configurations that matter most:

```text
2-zero-none
4-zero-none
8-sparse-none
```

For each final node it prints:

```text
A forced distance / false-positive excess / false-negative deficit / popcount
B forced distance / false-positive excess / false-negative deficit / popcount
whether that route is currently selected
whether that route is best by decoded distance
```

Focused result:

```text
2-zero-none:
        route 0: A 0/0/0 pop 60, selected+best
                 B 30/0/34 pop 64, selected+best
        route 1: A 0/0/0 pop 60
                 B 30/0/34 pop 64

4-zero-none:
        route 0: A 0/0/0 pop 36, selected+best
                 B 30/0/58 pop 40, selected+best
        routes 1..3:
                 same forced A/B quality, just unselected

8-sparse-none:
        route 2: A 0/0/0 pop 14, selected+best
                 B 30/0/82 pop 18, selected
        route 0: A 1/2/0 pop 12
                 B 29/0/83 pop 16, best for B by decoded distance
        route 7: A 1/0/1 pop 6
                 B 31/0/89 pop 10
```

This is the first direct sign that width is becoming route-like rather than
just more votes:

```text
2-zero-none and 4-zero-none:
        every route is effectively the same surface

8-sparse-none:
        route 2 is the stable sparse-A route
        route 0 is the better dense-B route
        but route 0 is not A-safe
```

So the widened case does expose latent alternate memory surfaces, but they are
not yet stably assigned:

```text
stable route:
        selected and best for the same example

latent route:
        best by decoded distance but not selected or not replay-safe
```

That makes the next mechanism more concrete:

```text
width alone is insufficient
the model now needs explicit route ownership / route selection pressure
```

The current replay question is therefore:

```text
can A keep route 2 while B is explicitly trained onto route 0
or another unowned route, with selector pressure to keep those routes selected?
```

### Route Classification And Assignment

The focused replay harness now also classifies routes as:

```text
stable / latent / unsafe / dead
```

under a simple selector-margin rule.

The current assignment summary for the requested cases is:

```text
2-zero-none:
        A route 0 stable
        B route 1 dead

4-zero-none:
        A route 0 stable
        B route 1 dead

8-sparse-none:
        A route 2 stable
        B route 0 unsafe
        B route 0 is best by decoded distance, but not selected

8-zero-none:
        A route 0 stable
        B route 1 dead

4-sparse-none:
        A route 3 stable
        B route 2 unsafe
        B route 2 improves dense distance, but is not selected
```

So the route picture is now:

```text
zero-init narrow models:
        effectively one shared route surface

sparse-init wider models:
        expose latent or unsafe alternate routes
        but not yet a stable B-owned route
```

This sharpens the next mechanism again:

```text
the model does not just need width
it needs route ownership plus selector pressure
```

In other words, the current widened survivors are better read as:

```text
latent routes exist
but the trainer still lacks a way to make B claim them stably
while preserving A's owned route
```

### Route-Aware Training Probe

The next diagnostic step was a minimal route-aware trainer:

```text
1. choose an A owner route
2. choose the best alternate B route
3. train only that B route with a route-pinned final-layer oracle
4. require A forced route and A global decode to remain solved
5. measure whether B's forced route and global decode improve
6. check whether the assigned B route is actually selected with a selector margin
```

Focused results:

```text
2-zero-none:
        A route 0 stays solved
        B route 1 ends 30 / 34
        selected yes, margin yes

4-zero-none:
        A route 0 stays solved
        B route 1 ends 30 / 58
        selected yes, margin yes

8-sparse-none:
        A route 2 stays solved
        B route 0 ends 30 / 84
        selected yes, margin no

8-zero-none:
        A route 0 stays solved
        B route 1 ends 30 / 74
        selected yes, margin yes

4-sparse-none:
        A route 3 stays solved
        B route 2 ends 30 / 66
        selected yes, margin yes
```

The important row is still `8-sparse-none`:

```text
before route-aware training:
        B route 0 forced = 29 / 83

after route-aware training:
        B route 0 forced = 30 / 84
```

So the current route-pinned oracle does **not** make the latent dense route
usable. It keeps `A` solved, but it does not preserve the dense-B improvement,
and it does not establish a selector margin on the assigned B route.

This is an important boundary:

```text
route diagnostics show latent alternate surfaces
but the current oracle, even when pinned to a route,
does not yet train B onto them as stable owned memory
```

So the next route mechanism needs to be stronger than route-pinned decoder
training alone. It likely needs:

```text
route ownership semantics
plus explicit selector-pressure updates
plus replay protection scoped to the owned A route
```

### Route-Owned Two-Stage Probe

A stricter route-owned diagnostic was then added in two stages:

```text
Stage 1:
        train only the assigned B route
        require A forced owned route to stay solved
        require global A decode to stay solved
        score B only on the forced assigned route

Stage 2:
        from the accepted Stage-1 prefix,
        try to make the assigned B route win global selection
        with a selector-margin style readout
```

Focused result:

```text
2-zero-none:
        A route 0, B route 1
        Stage 1 accepted 7 steps, then rejected because global A broke
        replay-safe prefix ends with:
                A forced 0/0
                A global 2/4
                B forced/global 28/28

4-zero-none:
        A route 0, B route 1
        Stage 1 accepted 8 steps
        replay-safe prefix ends with:
                A forced/global 0/0
                B forced/global 30/30
        Stage 2 can make the selected route line up,
        but there is still no useful dense-B route

8-sparse-none:
        A route 2, B route 0
        Stage 1 accepted 1 step, then rejected because global A broke
        replay-safe prefix ends with:
                A forced/global 0/0
                B forced/global 29/29
        Stage 2 does not establish selector margin

8-zero-none:
        A route 0, B route 1
        same pattern as other zero-init rows:
        no useful alternate route appears

4-sparse-none:
        A route 3, B route 2
        Stage 1 accepted 5 steps, then rejected because global A broke
        replay-safe prefix ends with:
                A forced/global 0/0
                B forced/global 28/28
```

The important row remains `8-sparse-none`:

```text
latent route exists:
        B forced route 0 can reach 29

but route ownership is not stable:
        one accepted forced-route step is possible
        then A owner-route margin collapses
        selector margin is still not established
```

So the boundary is now:

```text
latent alternate routes exist in sparse wide models
but the current route-owned training probe still lacks
enough owner-margin protection / selector control
to turn them into stable memory slots
```

### Owner-Margin Failure Split

The route-owned Stage-1 probe was then tightened again to separate:

```text
A owner-route break
A owner-route margin loss
A global selector break
B forced-route regression
```

This changed the immediate diagnosis.

Focused result:

```text
2-zero-none:
        A route 0, B route 1
        Stage 1 rejects immediately with:
                A owner-margin 0 -> -8

4-zero-none:
        A route 0, B route 1
        Stage 1 rejects immediately with:
                A owner-margin 0 -> -4

8-sparse-none:
        A route 2, B route 0
        Stage 1 accepts 1 forced-route B step:
                A forced/global 0/0
                B forced/global 29/83
        but then rejects on:
                A owner-margin 0 -> -2

8-zero-none:
        A route 0, B route 1
        Stage 1 rejects immediately with:
                A owner-margin 0 -> -2

4-sparse-none:
        A route 3, B route 2
        Stage 1 rejects immediately with:
                A owner-margin 2 -> -2
```

So the next bottleneck is now sharper than the earlier “global selector break”
wording:

```text
the assigned A owner route can remain decoded-correct,
but B route training reduces its popcount margin below safety
before a stable B-owned route is established
```

That means the next missing mechanism is:

```text
A owner-margin protection
before B selector-margin training
```

The route training order is now best viewed as:

```text
1. assign A owner route and B candidate route
2. protect A owner-route margin
3. train B on the forced candidate route
4. only then train B selector margin
```

In other words, the widened sparse model has exposed useful latent routes, but
the trainer still needs an explicit policy for preserving the owner route’s
selection margin while those routes are being used.

### Margin-Backed Route Training

The next step was to make Stage 1 transactional around the owned `A` route
margin instead of treating selector protection as a one-shot precondition:

```text
Stage 1a:
        if A owner margin <= floor,
        try a selector-protection top-up

        candidate classes:
                owner-route lift
                competitor-route suppress

Stage 1b:
        apply one forced-route B update

        if that spends A margin or breaks A global selection,
        try another selector-protection top-up on the post-B state

        accept only if final state keeps:
                A forced owner route solved
                A global decode solved
                A owner margin >= floor
                B forced route improved or preserved useful progress
```

This is still deliberately narrow. The top-up is a one-bit final-layer selector
repair, not a full selector trainer.

Focused result:

```text
2-zero-none:
        Stage 1a suppress succeeds:
                A margin 0 -> 2
        but the first B-route step still overspends it:
                A margin 2 -> -6
        reject: a-owner-margin

4-zero-none:
        Stage 1a owner lift succeeds:
                A margin 0 -> 2
        margin quota can be restored after B updates,
        but even a B-aware selector-recovery block gives back the B step
        reject: b-route-regressed-topup

8-sparse-none:
        Stage 1a owner lift succeeds:
                A margin 0 -> 2
        then two margin-backed B steps survive
        while maintaining the reserve:
                committed state:
                        A global 0/0
                        A owner margin 2
                        B forced route 29/83
        the next probe step is rejected on:
                b-route-regressed-topup
        split:
                raw B step:
                        B 29/79 -> 29/79
                        A still 0/0
                        A margin falls below reserve
                after B-aware selector recovery:
                        B 29/79 -> 30/80
                        A margin restored to reserve
        selector-margin Stage 2 is still not stable:
                final A global 1/2

8-zero-none:
        Stage 1a owner lift succeeds:
                A margin 0 -> 2
        but the B-aware selector-recovery block still gives back the B step
        reject: b-route-regressed-topup

4-sparse-none:
        repeated top-ups are attempted,
        but ownership still fails under continued B updates
        and the next step still rejects on:
                a-owner-margin
```

The important row is still `8-sparse-none`:

```text
A route 2 can be protected from margin 0 -> 2
B route 0 can hold a useful forced-route state at 29/79
selector-margin quota repair can now keep that reserve intact
across the accepted Stage-1 B prefix
but the next useful B step is currently erased by the paired
B-aware selector-recovery block
before selector ownership becomes stable
```

So the boundary is now narrower again:

```text
latent storage exists
owner-margin protection exists
margin-backed B updates exist
selector-margin quota repair exists
but route ownership is still limited by the cost of recovering A ownership
after each B step, even with B-aware selector recovery,
and B route selection is still not established
```

The next missing mechanism is therefore not generic route storage. It is a
stronger selector policy:

```text
maintain A owner-route margin as protected state,
pair B forced-route steps with selector top-ups when needed,
use recovery-cost-aware B candidate selection,
and distinguish:
        raw B-step gain
        B gain after selector recovery,
and only after stable forced-route storage exists,
train B selector margin separately.
```
`selected_by_popcount != best_by_target_distance` or a non-selected node with a
meaningfully lower dense deficit.

A virtual node-polarity diagnostic now scores every final node twice under the
segment decoder:

```text
(node j, normal):     decode activation_j
(node j, complement): decode ~activation_j
```

This is intentionally non-mutating. It tests whether a complemented
interpretation could serve as an anti-code route before introducing a permanent
node complement or selector change.

On the same `sparse -> dense` replay state, polarity does not help yet:

```text
A sparse:
        best: node 0 normal
        normal node 0 distance: 0
        complement node 0 distance: 54

B dense:
        best: node 0 normal
        normal node 0 distance: 30
        complement node 1 distance: 35
```

So complement polarity is a reasonable diagnostic to keep, but it is not the
escape hatch for this specific 2-node replay state. The useful signal remains
the replay-safe scaffold result (`B deficit 88 -> 74` while `A` stays solved),
not an alternate complemented node.

The replay distinction is now:

```text
replay-safe-progress:
        the current example improves while solved replay examples remain solved

replay-prefix-failed:
        prefix/bisection would reject, but per-candidate filtering finds progress

replay-safe-no-gain:
        replay-safe candidates exist, but none help the current example

replay-blocked-all:
        all useful candidates are replay unsafe

replay-single-candidate-conflict:
        the best individual current-example repair is replay unsafe
```

Segment node-selection diagnostics decode every final node under the segment
decoder and compare popcount selection to decoder-aware alternatives:

```text
selected_by_popcount
best_by_target_distance
best_by_false_negative_deficit
best_by_false_positive_excess
popcount_gap
distance_gap_between_selected_and_best
```

On the current targeted shapes, popcount selection is not yet the bottleneck:

```text
shape          selected  best distance  best deficit  best excess  pop gap before->after  distance gap
2-node hard    0->0      0->0           0->0          0->0         0->0                   0->0
2-node sparse  0->0      0->0           0->0          0->0         36->896                0->0
4-node zero    0->0      0->0           0->0          0->0         0->0                   0->0
4-node sparse  0->0      0->0           0->0          0->0         86->222                0->0
```

So the current sparse cost is not caused by popcount selecting the wrong final
node. A decoder-aware node selector should stay on the shelf until a wider or
more randomized shape shows `selected_by_popcount != best_by_target_distance`
or a positive selected-vs-best distance gap.

Segment-native margin histograms use buckets `0..7` plus an overflow bucket.
For the initial sparse stalls:

```text
shape          FN deficit histogram       one-margin histogram       zero-safety histogram
2-node sparse  [0,0,0,0,0,0,0,0,32]       [0,0,0,0,0,0,0,0,0]        [0,0,0,0,0,0,0,0,32]
4-node sparse  [0,0,0,0,0,2,0,0,4]        [0,1,0,0,0,0,0,0,25]       [0,0,0,0,0,0,0,0,32]
```

The 2-node sparse case is still far from threshold on every target-one bit.
The 4-node sparse case is closer: two false-negative bits are deficit `5`, four
are overflow, and many correct target-one bits have high positive margin.

Layer-5 projection diagnostics under segment condense are safer than under the
AND decoder:

```text
shape          accepted  strict seen  blocker seen  harmful  neutral  c1 -> FN
2-node sparse  1         0            2             0        6        0
4-node sparse  1         0            2             0        6        0
```

So segment condense removes the old layer-5 target-one fragility, but layer 5
still provides only small blocker-helpful assists on these sparse shapes. It is
safe enough to keep measuring, but not yet strong enough to become the main
trainer. Under the quota-shaped layer-6 path, layer 5 is not part of the sparse
default: it adds flips without beating the no-layer-5 layer-6 repair totals.

`itty_feed_model_train_backwards_one` adds the first backward credit assignment
rule for one-node-per-layer feed models. It starts from the target expanded to
the final output width, then walks layers from last to first:

```text
desired_condensed      = reduce_by_half(desired_output)
                       | reduce_rotated_by_half(desired_output, rotation)
train current layer toward desired_condensed
desired_previous_input = desired_condensed XOR updated_mask
```

The previous layer then treats that desired input as its desired output. This is
not calculus backpropagation; it is a deterministic bit-native backward pass
through the feed-node equations. Rotated layers use the rotated inverse fold
during this backward pass, while final vocabulary decoding still uses ordinary
shape folding.

### Work Queues and Manager

Files:

- `src/itty-work-queue.h`
- `src/itty-work-queue-private.h`
- `src/itty-work-queue.c`
- `src/itty-manager.h`
- `src/itty-manager-private.h`
- `src/itty-manager.c`

`itty_work_queue_t` is a single background worker thread with a mutex-protected
FIFO queue.

`itty_manager_t` owns one work queue per CPU and distributes work round-robin.
It also owns condition objects used by the pipeline fence mechanism.

The manager now exposes a task layer above raw queue entries:

- `itty_manager_submit` schedules a callback and returns an `itty_task_t`,
- `itty_manager_wait_for_task` waits for one task,
- `itty_manager_task_get_result` retrieves the callback result,
- `itty_manager_wait_for_all` waits for all active manager-submitted tasks,
- `itty_task_group_t` collects tasks for fan-out/fan-in execution.

Task groups are the intended primitive for layer-style network execution: submit
one task per node, wait for the group, then collect results in submission order.
The lower-level `itty_manager_enqueue_work` API remains available for legacy
raw work items, but manager drain accounting is attached to the task API.

This layer is now general infrastructure for both pipeline barriers and
manager-driven network feed.

### Pipeline

Files:

- `src/itty-pipeline.h`
- `src/itty-pipeline-private.h`
- `src/itty-pipeline.c`

The pipeline is an ordered list of operations. Each operation has:

- an operation id,
- a callback,
- user data,
- a manager.

`itty_pipeline_process` submits non-fence operations into a manager task group.
When it encounters a fence, it waits for the current task group to finish, marks
the fence as passed, then starts a fresh group for later operations.

This gives the current pipeline a simple barrier model:

- operations before a fence may run concurrently,
- operations after a fence do not start until prior submitted operations finish,
- a fence does not impose ordering among the operations before it.

The pipeline is not yet a full dependency graph and does not propagate callback
results.

### Manager-Driven Network Feed

File:

- `src/itty-network.c`

`itty_network_feed_with_manager` is the parallel feed entry point. For feed-only
layers, it preserves the same layer-by-layer semantics as `itty_network_feed`,
but builds a staged execution buffer for each layer through a private per-layer
feed-layer plan object. Layers containing affinity nodes use a mixed-layer
runner. Both execution and presentation choose through the same internal layer
runner decision, so the feed-only and mixed-layer cases are not branched
separately in each caller. The current feed-layer plan has three stages:

- `network modulate inputs`: XOR input bit strings with each node's modulation masks,
- `network condense nodes`: condense each node's modulated inputs,
- `network double outputs`: double each condensed node output into the layer output list.

The network lowering also uses prose-style descriptor names such as
`network input`, `network mask`, `network modulated input`,
`network condensed output`, and `network layer output`, matching the exec-buffer
presentation conventions used by affinity.

`itty_network_layer_present` is the general one-layer inspection API: feed-only
layers are shown through the network feed-layer plan, while mixed layers include
feed-node headings and embedded affinity exec-buffer presentations. The private
feed-layer plan remains internal.

`itty_network_present_plan` walks the whole network and joins per-layer
presentations under `network layer N` headings. Feed-only layers use the private
network layer plan. Mixed layers present feed-node headings and embed affinity
exec-buffer presentations under `network affinity node N` headings. The
presenter advances with each layer's output shape so later layers are shown in
the same structural context used by the manager feed.

Nodes with no modulated inputs produce an empty output bit string. This keeps
the lowered manager path aligned with the synchronous feed for empty input
lists, empty mask lists, and empty layers.

The command processor runs each stage through the manager, then waits at the
stage boundary before moving to the next stage.

This provides deterministic layer outputs while allowing node computations
within a layer to run concurrently.

Layers containing affinity nodes are currently executed node-by-node by the
manager feed path. Feed nodes in mixed layers use the synchronous feed-node
operation, and affinity nodes use `itty_affinity_probe_list_with_manager`.

If no manager is supplied, `itty_network_feed_with_manager` delegates to
`itty_network_feed`. If a manager is supplied and the lowered plan cannot be
built or executed, the manager path returns `NULL` rather than silently falling
back to the synchronous implementation.

The current manager-driven feed is still synchronous at layer boundaries:

- layers execute sequentially,
- nodes within one layer can execute concurrently,
- task results are collected before the next layer starts.

That model matches the current feed-forward network shape and is a reasonable
foundation for later dependency-aware execution.

### Execution Buffers

Files:

- `src/itty-exec-buffer.h`
- `src/itty-exec-buffer.c`

`itty_exec_buffer_t` is the first intermediate representation for bit-string
work. Instead of high-level network code directly performing each operation, it
registers word buffers, adds typed commands, and inserts explicit stage
barriers.

The current descriptor model supports word buffers with:

- external storage, such as existing bit strings or mmap-backed regions,
- owned scratch storage allocated by the exec buffer,
- rebinding external descriptors to new word storage,
- read-only or read-write access tags,
- optional associated bit-string metadata for cache invalidation,
- optional debug names.

`itty_exec_buffer_find_descriptor` resolves a descriptor debug name to a buffer
id using the same uniqueness rule as stage lookup: no match or multiple matches
returns false. Duplicate descriptor names are still legal for presentation and
numeric id access.

Commands reference `itty_exec_buffer_slice_t` values rather than owning bit
strings directly. A slice identifies:

- buffer id,
- word offset,
- number of words,
- optional logical bit length.

A zero logical bit length means "use the full word capacity." Nonzero logical
length is slice-local, so the same descriptor can be viewed as a raw word buffer
in one command and as a semantic bit prefix in another command.
Callers create those views explicitly with `itty_exec_buffer_get_word_slice`
or `itty_exec_buffer_get_bit_slice`.

Exec buffers also expose intent-oriented aliases over the same slice
representation:

- `itty_exec_buffer_value_t`: one word, used for dynamic parameters,
- `itty_exec_buffer_array_t`: a contiguous word array,
- `itty_exec_buffer_bits_t`: a word-backed span with a logical bit length.

`itty_exec_buffer_get_value`, `itty_exec_buffer_get_array`, and
`itty_exec_buffer_get_bits` construct those views while preserving the same
descriptor and validation path as generic slices.

Command add functions validate slices before appending commands. A slice must
refer to an existing descriptor, stay within that descriptor's word capacity,
and have a logical bit length no larger than its own word capacity. Destination
slices must also point at read-write descriptors. This keeps descriptor and
slice mistakes from surviving until graph execution.

The initial command set is intentionally small:

- `XOR`,
- `XNOR`,
- `POPCOUNT`,
- `CONDENSE`,
- `WEIGHTED_CONDENSE`,
- `DOUBLE`,
- `CLEAR_ARRAY_RANGE`.

The command processor can run synchronously or through an `itty_manager_t`.
Manager-backed execution submits one task per command within a stage and waits
for the stage's task group before continuing. That gives the current runtime a
simple and explicit fan-out/fan-in execution model.

Exec buffers can also run one stage at a time. `itty_exec_buffer_run_stage`
and `itty_exec_buffer_run_stage_with_manager` make CPU barriers explicit: a
caller can run early stages, inspect or mutate registered buffers, then resume
later stages without rebuilding the graph.

Stages can carry optional debug names via `itty_exec_buffer_begin_named_stage`.
The unnamed `itty_exec_buffer_begin_stage` remains the generic barrier helper.
Stage names are included in `itty_exec_buffer_present`, which makes staged
graphs easier to inspect without exposing private structures.
`itty_exec_buffer_find_stage` resolves a named stage to its numeric index, so
callers can cache semantic stage boundaries instead of hardcoding construction
order. `itty_exec_buffer_run_named_stage` and its manager-backed variant are
convenience helpers for direct semantic stage execution. Duplicate stage names
are allowed for presentation, but name-based lookup and execution fail when a
name is ambiguous; callers can still run those stages by numeric index.

`itty_exec_buffer_present` returns a heap-allocated textual description of the
current descriptor table, stages, and commands. It is meant as a debugging and
test inspection tool for staged graphs such as affinity plans.

External descriptors can be rebound with `itty_exec_buffer_rebind_words` or
`itty_exec_buffer_rebind_bit_string`. Owned scratch descriptors are deliberately
not rebindable, so graph reuse does not blur storage ownership.

`CONDENSE` currently preserves existing bit-string/list semantics by adapting
buffer slices into temporary read-only bit-string views and using the existing
condense implementation internally. That keeps the leading-zero behavior
unchanged while moving scheduling and buffer ownership toward descriptor-backed
execution.

`WEIGHTED_CONDENSE` is the descriptor-backed output-voting primitive. It takes
input slices, a word slice containing one integer vote budget per input, and a
destination slice whose logical bit length defines the output length. It walks
bit positions directly, adds the votes for inputs that have a one at that bit,
and emits a one only when the sum reaches a strict majority of the total vote
budget. Because the command follows the destination slice's logical bit length,
it does not transpose the list and it does not process padding bits above the
requested logical output.

`POPCOUNT` respects a source slice's logical bit length when one is present.
This keeps semantic scoring from counting padding bits while preserving
whole-word popcount behavior for ordinary word slices.

`CLEAR_ARRAY_RANGE` is the first dynamic policy-oriented array command. It
takes a writable array plus two one-word values: `start` and `count`. At run
time it clears `count` words starting at `start`, clamping the range to the
destination array. A zero count or a start beyond the destination is a no-op.
This gives affinity a descriptor-backed way to express causal score masking
without rebuilding the graph for every probe position.

### Affinity

Files:

- `src/itty-affinity.h`
- `src/itty-affinity.c`
- `src/itty-position.h`
- `src/itty-position.c`

`itty_affinity_t` is the first bit-native attention-like block. It borrows a
trait list and an imprint list. Each trait is paired with the imprint at the
same index. The lists must have the same length.

For one probe, the affinity block:

1. scores each trait with exec-buffer `XNOR` and `POPCOUNT`,
2. optionally adds an explicit locality bonus based on probe/trait indexes,
3. converts scores into exact integer vote budgets,
4. mixes imprints with exec-buffer `WEIGHTED_CONDENSE`.

In transformer terms, this is closer to content-addressable memory than a
full attention layer:

```text
probe compares to traits -> integer vote budgets -> weighted condense of imprints
```

`itty_affinity_probe_at` is the sequence-aware entry point. It accepts a
probe index and locality window. A trait at the same index receives the full
window as bonus score, and the bonus falls linearly to zero at the edge of the
window:

```text
distance       = abs(probe_index - trait_index)
locality_bonus = distance >= window ? 0 : window - distance
```

`itty_affinity_probe_with_options` is the extensible entry point. Its options
struct currently carries total votes, probe index, locality window,
Gray-position scoring fields, and a causal flag. The simpler probe functions
are wrappers around this options path.

`itty_affinity_plan_t` is the reusable fixed-width form of a probe graph. A
plan is built for a specific score bit length and can then be reused for
compatible probes. The graph still has stable physical word widths, but the
match slices carry logical bit lengths so `POPCOUNT` scores semantic content
rather than padding.

`itty_affinity_plan_present` exposes the compiled plan as the underlying
exec-buffer presentation without making the exec buffer itself public. This is
primarily for debugging and tests: a two-trait plan should show XNOR commands
in the `affinity match` stage, POPCOUNT commands in the `affinity score` stage,
CLEAR_ARRAY_RANGE in the `affinity causal mask` stage, and WEIGHTED_CONDENSE
in the `affinity output` stage.
Those labels are private affinity constants; the plan resolves them once after
building and stores the resulting stage indexes for probe execution.

Affinity scores over `score_bit_length` when it is provided in probe options.
When it is zero, affinity scores over the probe's logical bit length, falling
back to the probe's stored bit capacity when the logical length is zero. In
other words, the probe defines the default question width; extra bits in a trait
do not change the content score for that probe, and all-zero stored probes can
still ask a fixed-width question.

When `causal` is true, traits after the probe index are hard-masked after
content scoring and before locality/Gray score adjustments:

```text
if trait_index > probe_index:
    score_i = 0
```

When `gray_position_bits` and `gray_position_weight` are nonzero, the affinity
score receives an additional bounded Gray-code position score:

```text
score_i += gray_position_weight *
           gray_similarity(probe_index, trait_index, gray_position_bits)
```

`itty_affinity_probe_list` applies the affinity block to every bit string in a
probe list. It copies the caller's base options for each probe and overwrites
`probe_index` with the current list index before probing. This is the first
synchronous sequence-level affinity pass and is the natural base for
self-attention-like contextualization.

`itty_affinity_probe_list_with_manager` is the manager-backed version. It
submits one task per probe, waits for the task group, then builds the result
list in probe-index order. The parallel execution path is therefore allowed to
complete out of order while preserving deterministic output order.

The block lowers the raw data-parallel work into one probe-local exec-buffer
graph. Raw similarity scores use one `XNOR` command per trait followed by one
`POPCOUNT` command per match buffer. The next stage uses `CLEAR_ARRAY_RANGE`
with dynamic one-word values to zero causal future scores. After those stages
run, affinity applies the remaining CPU policy barrier, fills the registered
vote buffer, then resumes the graph with `WEIGHTED_CONDENSE` for final imprint
mixing.

The synchronous probe-list helper reuses an affinity plan across consecutive
same-width probes. The manager-backed probe-list helper still creates one plan
per worker task via the normal probe path, avoiding shared mutable plan state
and nested manager execution.

The remaining CPU-side boundary is policy-heavy rather than word-buffer-heavy:
causal masking, locality and Gray-code positional bonuses, and exact vote
allocation stay in regular code so those semantics can continue to evolve
without adding control-flow commands to the exec-buffer layer.

### Position Helpers

Files:

- `src/itty-position.h`
- `src/itty-position.c`

The position helper layer currently provides:

- explicit locality bonus,
- Gray-code position encoding,
- bounded Gray-code similarity.

`itty_position_locality_bonus` is the primary distance-bias primitive. It is
monotonic by numeric distance and is already used by positioned affinity
probes.

`itty_position_gray_code` and `itty_position_gray_encode` provide bit-native
position identity. `itty_position_gray_similarity` compares Gray codes over an
explicit bit width so unused high bits do not dominate the score.

Gray code is intentionally treated as identity with local smoothness, not as a
complete distance metric. Adjacent positions differ by one bit, but distant
positions can still be close in Gray-code Hamming space.

### Command-Line Demo

File:

- `src/main.c`

The `itty-bitty` executable has two modes.

Context generation:

```sh
itty-bitty <vocabulary_text_file> <vocabulary_bit_string_file> <context_output_file>
```

It reads stdin, encodes it using the vocabulary, and writes a binary context
file.

Inference demo:

```sh
itty-bitty <vocabulary_text_file> <vocabulary_bit_string_file> <model_file> <context_file> <number_of_layers> <nodes_per_layer>
```

It loads context bit strings, builds a network from sequential model masks, runs
the manager-driven feed path, asks the network to select the final activation,
and decodes that activation to the nearest vocabulary token.

Model masks are read as:

- layer `i` uses `1 << i` words per mask,
- each node receives `nodes_per_layer` masks,
- there are `nodes_per_layer` nodes per layer.

This is a demo convention, not a stable model format.

## Data Flow

The current end-to-end data flow is:

1. Text vocabulary file plus binary vocabulary file create a vocabulary.
2. Input text is encoded to a context binary file.
3. Context file is mapped and read into one-word input bit strings.
4. Model file is mapped and read into modulation masks.
5. A network is constructed from those masks.
6. Input bit strings feed through the network.
7. The final activation is selected from the returned output list.
8. The selected activation is folded to vocabulary width.
9. The folded activation is XORed against each vocabulary token.
10. The token with the smallest disagreement popcount is translated back to
    text.

The current network selection rule is implemented by
`itty_network_select_output`: choose the output bit string with the largest
popcount, preserving the first winner on ties. The lower-level primitive is
still popcount argmax, but callers should treat this as the network's final
output convention rather than duplicating the selection policy.

The current vocabulary decoding rule is implemented by
`itty_vocabulary_decode_nearest`: repeatedly fold the selected activation with
`itty_bit_string_reduce_by_half` until it reaches vocabulary token width, then
choose the token with the smallest XOR popcount distance. Folding currently
uses the existing AND-based half reduction, so only bits present in both halves
survive each fold.

## Current Testing

Tests exist for:

- bit-string primitives,
- bit-string lists,
- mmap-backed bit-string files,
- vocabulary encoding/decoding,
- affinity,
- work queues,
- manager dispatch,
- pipeline fences.

The tests are useful smoke coverage but still mostly exercise small happy-path
examples. Edge-case and sanitizer coverage should be expanded before major
feature work.

## Known Limitations

The following limitations are architectural, not just missing code:

- Training currently only covers the one-layer, one-node, one-input feed-model
  MVP case.
- There is no stable model file format.
- There is no model metadata or shape validation.
- Vocabulary lookup is linear and prefix matching is simple.
- Pipeline results are not captured or composed.
- The work manager has no backpressure or cancellation policy.
- Allocation failures are not handled consistently across the codebase.
- Leading-zero and logical-length semantics are intentionally not fully settled.

## Plausible Next Steps

### 1. Make The Current Runtime Robust

Before adding training, the existing runtime should be made boring and
predictable.

Good near-term work:

- add `-Wall`, `-Wextra`, and sanitizer-friendly build options,
- run ASan/UBSan regularly,
- add tests for empty files, zero-size maps, resize shrink/grow, unknown
  vocabulary input, zero-popcount vote allocation, multi-word transpose/condense, and
  insufficient model files,
- handle allocation failures consistently.

This stage should avoid redefining leading-zero semantics unless a specific bug
requires it.

### 2. Define Model Shapes And Metadata

The demo currently infers model shape from CLI arguments and layer index. That
is not enough for reliable experiments.

A minimal model header could include:

- magic/version,
- word size or fixed serialization width,
- number of layers,
- nodes per layer,
- inputs per node per layer,
- words per mask per layer,
- vocabulary compatibility metadata.

The first version can remain simple and binary, but it should let the loader
validate that enough data exists before constructing a network.

### 3. Add Optional Runtime Tracing

The library no longer prints network feed internals directly. If interactive
tracing becomes useful again, keep it separate from normal execution:

- library returns structured outputs and status,
- optional trace callback receives layer/node/intermediate values,
- CLI decides how to present traces.

That would make tests cleaner and make the library usable from other programs.

### 4. Improve Vocabulary Lookup

The current longest-prefix scan is acceptable for tiny vocabularies. For larger
experiments, replace it with a prefix trie.

Useful behavior to decide:

- whether unknown input is an error, a special token, or byte fallback,
- whether vocabulary entries can be multi-character tokens,
- whether whitespace/newlines are tokens or separators,
- whether duplicate text or duplicate bit strings are allowed.

### 5. Add Shape-Safe Network Construction

Network nodes should validate their inputs:

- expected number of modulation masks,
- expected words per mask,
- expected input count,
- expected output width.

Instead of silently truncating to the shorter list during node modulation, the
network layer can either reject mismatched shapes or make the truncation policy
explicit.

### 6. Deepen Output Decoding

The current final decoder folds the selected activation to vocabulary width and
chooses the nearest vocabulary entry by XOR popcount distance. Useful next
decoding experiments:

- return popcount-ranked candidate lists,
- compare AND-folding against XOR-folding or majority-folding,
- add a layer-specific projection back to vocabulary width,
- replace nearest-token lookup with a separate decoder block.

These choices are closely tied to the training story and should be evaluated
with small deterministic experiments.

### 7. Design Training As A First-Class Pass

The README sketches a training idea based on XOR differences and backward
folding. To make that concrete, the runtime probably needs:

- an explicit forward pass result object,
- retained per-layer/per-node intermediates,
- an error bit-string representation,
- a backward pass over network structure,
- mask update rules,
- tests over tiny deterministic examples.

Training should be introduced with a tiny supervised task where expected
behavior is easy to inspect by hand.

### 8. Deepen Affinity

A useful next direction is deepening the current bit-native cousin of
transformer attention. The goal is not to reproduce floating-point scaled
dot-product attention directly, but to preserve the same high-level shape with
bit-string primitives:

```text
transformer attention:
probe compares to keys -> softmax weights -> weighted sum of values

itty-bitty attention:
probe compares to traits -> integer vote budgets -> weighted condense of imprints
```

The natural primitive stack is:

1. Compare a probe bit string against trait bit strings with XNOR/popcount.
2. Convert the resulting scores into exact integer vote budgets.
3. Let paired imprint bit strings vote with those budgets.
4. Emit an output bit string using weighted majority/condense.

For one probe:

```text
score_i = popcount(xnor(probe, trait_i))
votes_i = allocate_votes(score_i, total_votes)
output  = weighted_condense(imprints, votes)
```

Regular `itty_bit_string_list_condense` gives each input one vote per bit.
`itty_bit_string_list_weighted_condense` gives each imprint `votes_i` votes per
bit:

```text
ones(bit_j) = sum(votes_i for imprints where imprint_i[j] == 1)
output[j]   = ones(bit_j) >= majority_threshold
```

This is closer to attention than simply choosing the nearest trait. Imprints
paired with strongly matching traits dominate, but weaker matches can still
affect bits where enough of them agree.

The current API lives in a new affinity/memory module rather than inside the
feed-forward network layer:

```c
typedef struct itty_affinity_t itty_affinity_t;

itty_affinity_t *itty_affinity_new (itty_bit_string_list_t *traits,
                                    itty_bit_string_list_t *imprints);
itty_bit_string_t *itty_affinity_probe (itty_affinity_t *affinity,
                                        itty_bit_string_t  *probe,
                                        size_t              total_votes);
void itty_affinity_free (itty_affinity_t *affinity);
```

For a self-attention-like pass over a context list, each token probes the same
list:

```text
for each position i:
    probe    = token_i
    traits  = context tokens
    imprints = context tokens
    output_i = affinity_probe(probe, traits, imprints, total_votes)
```

The current helper for this shape is `itty_affinity_probe_list`, which produces
one output imprint per input probe. `itty_affinity_probe_list_with_manager`
uses the same semantics but parallelizes probes through the manager.

For causal generation, future positions are masked before vote allocation:

```text
if trait_index > probe_index:
    score_i = 0
```

The execution buffer already has the basic scoring commands (`XNOR` and
`POPCOUNT`) plus `WEIGHTED_CONDENSE` for descriptor-backed output voting.

Vote allocation and top-k style selection should remain CPU-side until the
semantics settle, because they are control-flow-heavy compared with word-buffer
transforms.

### 9. Deepen Positional Encoding

Attention over a sequence needs position information. There are two related but
separate concepts:

- position identity: where a token is,
- distance bias: how strongly nearby tokens should be preferred.

The first distance-bias mechanism is already implemented as an explicit score
bonus:

```text
distance        = abs(probe_index - trait_index)
locality_bonus  = max(0, window - distance)
score_i         = content_score_i + locality_bonus
```

This gives nearby traits extra votes before allocation without making distance
the only factor. A far token can still win if its content match is strong
enough, while near tokens get a natural advantage.

Gray codes are available as a standalone position identity encoding because
adjacent integers always differ by one bit:

```text
gray(n) = n ^ (n >> 1)
```

Plain binary counters have poor local Hamming behavior because adjacent
positions can flip many bits. Gray codes avoid that for adjacent positions, so
XNOR/popcount comparison treats nearby positions more gently. However, Gray
code Hamming distance is not a perfect numeric distance metric; distant
positions can still be close in Hamming space. It should not be the only
locality mechanism.

A clean scoring model keeps content and position separate:

```text
content_score_i  = popcount(xnor(probe_token, trait_token_i))
position_score_i = popcount(xnor(position(probe_index), position(trait_index)))
locality_bonus_i = max(0, window - abs(probe_index - trait_index))

score_i = content_score_i
        + position_weight * position_score_i
        + locality_bonus_i
```

The `position_weight` can be implemented as integer multiplication during
scoring or as repeated position bits in a purely bit-string representation.

For richer locality, position encodings can be multi-scale:

```text
position(i) = gray(i) || gray(i / 4) || gray(i / 16)
```

Fine bits distinguish nearby positions, while coarse bits identify blocks and
larger neighborhoods. This is still experimental, but it fits the bit-native
similarity model better than a single absolute counter.

A later position module extension could expose multi-scale encodings:

```c
itty_bit_string_t *itty_position_multiscale_gray_encode (size_t position,
                                                         size_t number_of_words,
                                                         size_t scale_count);
```

Gray-code position scores are available through affinity probe options. They
should remain lower-weight than explicit locality unless an experiment shows
otherwise, because Gray-code similarity is position identity with local
smoothness rather than a monotonic distance metric.

### 10. Deepen Manager And Pipeline Semantics

The pipeline and work manager can evolve in two directions:

- a general async task runner for file/model operations,
- or the execution engine for network layers/nodes.

If it becomes the network execution engine, it needs:

- dependency tracking,
- failure propagation,
- cancellation/shutdown semantics,
- deterministic test hooks.

The current task-group model is enough for layer-level fan-out/fan-in. A later
dependency graph could be built on top of execution-buffer descriptors and
commands rather than replacing the queue layer again.

## Suggested Milestone Order

1. Stabilize existing runtime behavior with sanitizer-backed tests.
2. Add model metadata and shape validation.
3. Split tracing from library logic.
4. Generalize feed-model training beyond the one-input MVP case.
5. Add model metadata and serialization.
6. Add synchronous affinity over bit-string lists.
7. Add explicit locality bias, then experiment with Gray-code position scores.
8. Revisit pipeline integration only after the synchronous path is correct.

This order keeps the project moving without committing too early to unsettled
questions like logical length, leading-zero policy, or the final training rule.
Latest focused diagnostic update: reserved selector-lane probe
--------------------------------------------------------------

Added a focused replay diagnostic that treats the final capacity as two disjoint
lanes on the `8-sparse-none` route-owned case:

```text
selector lane: reserved upper slice only
decoder lane:  all remaining lower bits
```

The probe keeps the existing route-owned flow, but:

```text
A owner-margin protection:
        selector-lane flips only

B forced-route decode training:
        decoder-lane flips only

evaluation:
        route popcount from selector lane only
        segment-condense decode from decoder lane only
```

Instead of a 50/50 split, the diagnostic now sweeps small reserved selector
slices:

```text
selector bits: 4, 8, 16
decoder bits:  rest
```

Current focused result:

```text
segment replay route-owned lane-split:
  sel=4  dec=60  routes 0/6  A 0 1/0 -> 0 1/0  B 0 32/93 -> 0 32/93
                   probe 32/92 -> 32/92  A-sm 0 -> 0  reject a-owner-margin

  sel=8  dec=56  routes 0/6  A 0 1/0 -> 0 1/0  B 0 32/93 -> 0 32/93
                   probe 32/92 -> 32/92  A-sm 0 -> 0  reject a-owner-margin

  sel=16 dec=48  routes 2/0  A 0 1/0 -> 0 1/0  B -1 32/93 -> 0 32/93
                   probe 32/92 -> 32/92  A-sm 0 -> 0  reject a-owner-margin
```

Interpretation:

```text
The disjoint lane construction still prevents direct cross-lane mutation by
construction, but the reserved-slice selector lane does not yet recover the
route-owned behavior on the focused sparse-width case.

Unlike the original route-owned path, the lane-separated route geometry is
already poor before the earlier `b-route-regressed-topup` boundary:

        A global decode starts wrong (1/0)
        B forced-route decode stays far away (32/93)
        A selector margin never builds above 0
```

So the stronger current conclusion is:

```text
simply reserving a small selector-only slice is not enough as a drop-in fix;
the lane split changes the useful route geometry itself.
```

That means selector-lane work remains a representation experiment, not yet a
validated route-ownership repair. The next selector-lane iteration, if pursued,
should keep the old route assignment as a control and compare whether the lane
split is starving decoder support or collapsing route separation before any
useful ownership can form.

Latest focused diagnostic update: shadow selector head
------------------------------------------------------

The next selector experiment kept the decoder surface unchanged and added a
separate selector-only shadow head:

```text
decoder surface:
        original full final activation and original final masks

selector surface:
        separate final-mask bank, same final-layer input,
        popcount only, never used for segment decode
```

Focused `8-sparse-none` result:

```text
segment replay shadow-selector:
  routes A/B 2/0
  A selector margin 2 -> 2
  A shadow-global 0/0
  A forced-route 0/0
  B forced-route 30/86 -> 28/60
  selector flips 1
  decoder flips 64
  accepted steps 8
  reject none
  B selected route under selector head: 2
  selector gap: 2
```

Interpretation:

```text
The additive shadow selector behaves much better than the reserved-slice lane.
It preserves the old decoder geometry well enough to:

        keep A solved globally,
        keep A owner margin at reserve,
        and improve B on its forced route substantially.
```

This is the first focused result that cleanly separates selector support from
decoder support without collapsing the route-owned storage path.

The remaining boundary has shifted again:

```text
route storage under separate selector support:
        works

B route selection under the selector head:
        not established yet
```

So the current shadow-head result says:

```text
the main coupling problem was real;
separating selector support from decoder support reopens route-owned B storage;
the next missing mechanism is explicit B selector-head training so B selects
route 0 instead of staying on route 2.
```

Latest focused diagnostic update: shadow selector inventory and Stage 2 buckets
-----------------------------------------------------------------------------

The shadow-selector probe was then instrumented to print the selector inventory
for the dense `B` example before any Stage-2 selector training:

```text
segment replay shadow-selector:
  a_route=2
  b_route=0
  b_desired_pop=12
  b_winner_route=2
  b_winner_pop=16
  b_gap=4
  b_need=5
  inv_lift=122
  inv_suppress=8
  inv_mixed=0
  a_safe_lift=0
  a_safe_suppress=0
  b_eff_lift=122
  b_eff_suppress=8
  a_comp_lift=122
  a_comp_suppress=5
  a_comp_eff_lift=122
  a_comp_eff_suppress=5
  a_sm_before=2
  a_sm_after=2
  a_shadow_dist=0
  a_shadow_fp=0
  a_forced_dist=0
  a_forced_fp=0
  b_forced_before_dist=30
  b_forced_before_def=86
  b_forced_after_dist=28
  b_forced_after_def=60
  b_sm_before=-4
  b_sm_after=-4
  b_sel=2
  b_sel_gap=2
```

That inventory makes the selector bottleneck much sharper:

```text
desired B route popcount: 12
current winner route popcount: 16
needed net movement toward route 0: 5

B-effective route-0 lifts: 122
B-effective route-2 suppressions: 8

A-safe lifts: 0
A-safe suppressions: 0

A-compensation route-2 lifts: 122
A-compensation competitor suppressions: 5
```

So the selector head has no shortage of directions that help `B`, but none of
the currently visible one-step selector moves are compatible with `A`'s route-2
ownership constraint.

The Stage-2 selector pass was then tightened again into a transaction:

```text
1. build a B-selector block toward route 0
2. try an A-selector repair block that restores route-2 ownership
3. accept only the final combined state
```

The focused `8-sparse-none` result after the strict/scaffold Stage-2 split is:

```text
stage2_strict_acc=0
stage2_strict_rej_no_candidates=0
stage2_strict_rej_a_unsafe=0
stage2_strict_rej_comp_erases=0
stage2_strict_rej_insufficient_margin=8
stage2_strict_best_margin_after_b=6
stage2_strict_best_margin_after_repair=-2
stage2_strict_best_a_margin_after_repair=2
stage2_strict_a_safe_blocks=8
stage2_strict_b_selecting_blocks=0

stage2_scaffold_acc=1
stage2_scaffold_rej_no_candidates=0
stage2_scaffold_rej_a_unsafe=0
stage2_scaffold_rej_comp_erases=7
stage2_scaffold_rej_insufficient_margin=0
stage2_scaffold_best_margin_after_b=2
stage2_scaffold_best_margin_after_repair=-2
stage2_scaffold_best_a_margin_after_repair=2
stage2_scaffold_a_safe_blocks=8
stage2_scaffold_b_selecting_blocks=0
```

This changes the interpretation again:

```text
now:
        the selector head can realize A-safe combined blocks,
        and scaffold mode can improve B selector margin from -4 to -2,
        but no realized block yet makes B select route 0
```

A bounded joint selector search with visited-state pruning was then added so
Stage 2 no longer depended on the earlier greedy `B block -> A repair` order.
The focused `8-sparse-none` result under that bounded search is:

```text
b_sm_before=-4
b_sm_after=-2
b_sel=2
b_sel_gap=2

stage2_strict_acc=0
stage2_strict_rej_no_candidates=0
stage2_strict_rej_a_unsafe=0
stage2_strict_rej_comp_erases=0
stage2_strict_rej_insufficient_margin=1
stage2_strict_best_margin_after_b=12
stage2_strict_best_margin_after_repair=-2
stage2_strict_best_a_margin_after_repair=18
stage2_strict_a_safe_blocks=17
stage2_strict_b_selecting_blocks=0

stage2_scaffold_acc=1
stage2_scaffold_rej_no_candidates=0
stage2_scaffold_rej_a_unsafe=0
stage2_scaffold_rej_comp_erases=0
stage2_scaffold_rej_insufficient_margin=0
stage2_scaffold_best_margin_after_b=12
stage2_scaffold_best_margin_after_repair=-2
stage2_scaffold_best_a_margin_after_repair=18
stage2_scaffold_a_safe_blocks=17
stage2_scaffold_b_selecting_blocks=0
```

This is a stronger selector boundary:

```text
the problem is no longer "the greedy builder realizes no joint block"

instead:
        bounded A-safe joint selector states do exist,
        scaffold mode still improves B margin to -2,
        but no bounded A-safe state makes B select route 0
```

The decoder side is still intact under the shadow selector:

```text
A selector ownership under the shadow head:
        stable

B decoder storage on route 0:
        30/86 -> 28/60
```

So the current boundary is:

```text
shadow selector proves selector/decode separation is useful;
route-local decoder storage for B works;
bounded joint selector search still does not realize a B-selecting state.
```

The current diagnostic implication is:

```text
there are many B-helpful selector moves
there are also many A-compensation moves
the selector head can realize A-safe combined states and partial B margin
recovery
but the current bounded search still does not find any A-safe state where
B actually selects route 0
```

At this point the missing mechanism is no longer decoder repair or selector
recovery. It is a selector-head routing policy that can convert available
`B`-effective directions plus `A`-compensation moves into a realized route-0
selection block without breaking `A`'s owned route.

Latest focused diagnostic update: widened shadow selector
---------------------------------------------------------

After fixing the width-assumption overflow in the shared final-repair path, the
shadow-selector experiment was rerun with a wider selector head while keeping
the decoder model at the earlier width. The widened run no longer crashed under
ASan, so the result below reflects the selector-capacity experiment rather than
the old memory bug.

Focused widened-shadow result:

```text
segment replay shadow-selector (wide selector head):
  a_route=2
  b_route=0

  A ownership:
    a_sm 2 -> 2
    a_shadow 0/0
    a_forced 0/0

  B selector inventory:
    b_desired_pop = 16
    b_winner_route = 2
    b_winner_pop = 30
    b_gap = 14
    b_need = 15

    inv_lift = 242
    inv_suppress = 15
    a_safe_lift = 0
    a_safe_suppress = 0
    a_comp_lift = 243
    a_comp_suppress = 12

  B decoder route:
    b_forced 30/86 -> 30/86

  Stage 2 selector routing:
    b_sm_before = -14
    b_sm_after = -2
    b_sel = 2
    b_sel_gap = 2

    stage2_strict_acc = 0
    stage2_scaffold_acc = 2

    scaffold rounds:
      round 0: -14 -> -6
      round 1:  -6 -> -2
      round 2:  stall at -2
```

This changes the selector-capacity interpretation in an important way:

```text
widening the selector head does help selector routing movement;
the widened selector scaffold can move B from -14 to -2 under A ownership;
but B still does not switch to route 0.
```

The tradeoff is also sharper than before:

```text
baseline shadow selector:
        B decoder storage works strongly     (30/86 -> 28/60)
        selector scaffold improves to -2

wider selector head:
        selector scaffold improves much more (-14 -> -2 across two rounds)
        but B decoder storage no longer improves at all (30/86 -> 30/86)
```

So widening selector capacity alone is not a clean win in the current focused
setup. It strengthens selector-margin movement, but it does not preserve the
earlier route-0 decoder-storage gain that made the baseline shadow-selector
result compelling.

The current boundary is now:

```text
selector/decode separation is still the right architecture direction;
extra selector capacity improves routing frontier movement;
but selector widening alone does not yet produce a stable route-0 selection
while retaining the earlier B decoder-storage improvement.
```

Latest focused diagnostic update: shadow-selector density sweep
--------------------------------------------------------------

After fixing the focused diagnostic stack-overflow bug in the Stage-2 selector
search harness, the selector-density sweep was rerun under ASan so the results
would be trustworthy again.

Focused densities tested:

```text
selector_words = 1
selector_density = 1/16, 1/8, 1/4
```

All three densities now complete cleanly under the focused ASan harness. The
important result is that they all converge to the same selector frontier shape.

Representative summary:

```text
selector 1/16:
        A ownership: stable
        B decoder route: 30/86 -> 28/60
        B selector margin: -4 -> -2
        B selected route: 2
        scaffold rounds: one accepted, then stall at -2

selector 1/8:
        A ownership: stable
        B decoder route: 30/86 -> 28/60
        B selector margin: -4 -> -2
        B selected route: 2
        scaffold rounds: one accepted, then stall at -2

selector 1/4:
        A ownership: stable
        B decoder route: 30/86 -> 28/60
        B selector margin: -4 -> -2
        B selected route: 2
        scaffold rounds: one accepted, then stall at -2
```

The selector inventory counts do change with density, but the reachable
route-switching frontier does not:

```text
1/16:
        B-effective lifts/suppresses: 123 / 6
        A-compensation lifts/suppresses: 124 / 3

1/8:
        B-effective lifts/suppresses: 121 / 8
        A-compensation lifts/suppresses: 122 / 5

1/4:
        B-effective lifts/suppresses: 101 / 28
        A-compensation lifts/suppresses: 102 / 25
```

But in every case:

```text
A ownership remains protected
B decoder storage remains strong (30/86 -> 28/60)
B selector scaffold reaches -2
and then stalls before switching to route 0
```

So the selector-density question is now answered cleanly:

```text
changing selector sparse density alone does not reopen route-0 selection
for the focused 8-sparse-none shadow-selector case.
```

That is a useful narrowing. The remaining selector bottleneck is no longer an
obvious density-tuning problem. The next selector-only experiments should be
structurally stronger than density changes, for example:

```text
1. wider selector head
2. second selector layer
3. route-local selector adapters
```

Latest focused diagnostic update: slack-vector selector search
--------------------------------------------------------------

The next Stage-2 selector refinement changed the search from raw margin-only
ranking to an explicit slack-vector ranking and pruning scheme. For each
selector-search state the diagnostic now tracks:

```text
A owner-margin shortfall
B route-selection shortfall
B decoder distance shortfall
B decoder deficit shortfall
```

Those slack terms are used in two places:

```text
1. dominance pruning:
        prune states already dominated on A/B selector slack and B decoder slack

2. best-state ranking:
        prefer lower selector shortfall first,
        then better B decoder preservation,
        then route margin / popcount gap / step count
```

This is closer to a discrete finite-difference / slack-transport view of the
selector problem than the earlier pure end-state bucket logic.

Focused `8-sparse-none`, `selector_words=1`, `selector_density=1/8` result:

```text
A ownership:
        a_sm_after = 2
        A shadow/global = 0/0

B decoder storage:
        b_forced_after_dist = 28
        b_forced_after_def  = 60

B selector routing:
        b_sm_after = -2
        b_sel      = 2
        b_sel_gap  = 2

Stage 2:
        strict_acc    = 0
        scaffold_acc  = 1
        scaffold_rounds = 1
        round 0: B margin -> -2
        round 1: stall
        reject reason = candidate-block-search-budget-exhausted
```

So the selector-search interpretation tightens again:

```text
the search is now scoring explicit selector/decoder slack vectors;
the frontier still reaches only B margin -2;
and the route-0 switch still does not happen.
```

That means the current bottleneck is no longer just a weak scalar ranking over
candidate selector states. Even with explicit slack-aware search, the focused
shadow-selector head still stalls at the same routing boundary:

```text
A ownership: stable
B decoder storage: stable and useful (30/86 -> 28/60)
B selector scaffold: partial (-4 -> -2)
B selector completion: still missing
```

The next selector-only step therefore remains a capacity/representation change,
not more scoring refinement on the same one-word selector head.

Latest focused diagnostic update: frozen-decoder selector sweep
---------------------------------------------------------------

The next selector experiment froze the successful decoder state first, then
swapped only the selector head. This isolates selector routing capacity from
decoder-storage side effects.

Frozen decoder baseline:

```text
A assigned route: 2
B assigned route: 0
A owner margin:   2
B forced decode:  28/60
```

Then the selector head was varied while keeping the decoder fixed:

```text
selector words:   1, 2, 4
selector density: 1/16, 1/8, 1/4
```

Results obtained so far:

```text
1 word, 1/16:
        A safe
        B forced stays 28/60
        B selector margin = -2
        B selected route = 2
        no B-selecting block

1 word, 1/8:
        A safe
        B forced stays 28/60
        B selector margin = -2
        B selected route = 2
        no B-selecting block

1 word, 1/4:
        A breaks badly
        B forced stays 28/60
        B selector margin = -16
        B selected route = 6

2 words, 1/16:
        A safe
        B forced stays 28/60
        B selector margin = -2
        B selected route = 2

2 words, 1/8:
        A safe
        B forced stays 28/60
        B selector margin = -2
        B selected route = 2
        one scaffold step accepted, but no route switch

2 words, 1/4:
        A breaks badly
        B selector margin = -22
        B selected route = 1

4 words, 1/8:
        A owner margin falls to 0
        B forced stays 28/60
        B selector margin = -2
        B selected route = 2
        one scaffold step accepted, but no route switch
```

The useful interpretation is:

```text
freezing the decoder works as intended:
        B route-0 storage remains 28/60

selector width alone is not enough:
        larger selector heads still stall at B selector margin -2

denser selector initialization is actively bad:
        1/4 density destroys A ownership and does not help B routing
```

So the selector-only picture is sharper now:

```text
the decoder route really is stored;
the selector head is the remaining bottleneck;
naive width increases do not complete route selection;
and dense selector masks make ownership worse.
```

That suggests the next selector experiments should move beyond simple width or
density tuning and test richer selector structure, for example:

```text
1. second selector layer
2. route-local selector adapters
3. more explicitly associative selector routing
```

Seeded frozen-selector follow-up:

```text
4 words, 1/8 density, seed 41218:
        A owner margin = 2
        B forced decode = 28/60
        B selector margin = -2
        B selected route = 2

4 words, 1/8 density, seed 41219:
        A owner margin = 2
        B forced decode = 28/60
        B selector margin = 0
        B selected route = 0
```

That is the first frozen-selector result that reaches route `0` at all while
preserving the frozen decoder state. But it still does not satisfy the stricter
selector-win criterion because the route-0 hit is fragile:

```text
B reaches route 0 for one seed,
but without the stable selector gap required by the strict routing criterion.
```

So the selector boundary tightens again:

```text
route reachability exists under frozen decoder storage;
stable selector routing is still missing.
```

Latest focused diagnostic update: associative route-key selector
---------------------------------------------------------------

The next structural selector experiment replaced the shadow-selector popcount
repair logic with a minimal route-key selector diagnostic over the frozen
decoder state.

Setup:

```text
decoder:
        frozen at the successful route-0 storage state
        B forced route 0 = 28/60

selector:
        route score_r = popcount(xnor(probe, route_key_r))
        selected route = argmax(score_r)
```

In the focused diagnostic:

```text
route_key_2 = clone(A probe)
route_key_0 = clone(B probe)
other route keys = fixed mixed-pattern baseline
```

Focused result:

```text
A route = 2
A selected route = 2
A selected gap = 1

B route = 0
B selected route = 0
B selected gap = 1

B forced decode remains 28/60
```

That is the first selector-structure result that simultaneously gives:

```text
A -> route 2
B -> route 0
positive selector gaps for both
and preserved frozen decoder storage for B
```

The focused `A -> B -> A` follow-up was then checked directly by running an
additional replay attempt on `A` after the route-key selector was established:

```text
replay_a_flips = 0
after_replay_a_forced_dist = 0
after_replay_b_forced_dist = 28
after_replay_b_forced_def  = 60
after_replay_a_selected_route = 2
after_replay_a_selected_gap   = 1
after_replay_b_selected_route = 0
after_replay_b_selected_gap   = 1
```

So in this focused case:

```text
the extra A replay step is a no-op,
and the route-key selector keeps both route assignments and both positive gaps.
```

The focused chain was then extended one more step to `A -> B -> A -> B`:

```text
replay_b_flips = 8
after_replay2_a_forced_dist = 0
after_replay2_b_forced_dist = 27
after_replay2_b_forced_def  = 57
after_replay2_a_selected_route = 2
after_replay2_a_selected_gap   = 1
after_replay2_b_selected_route = 0
after_replay2_b_selected_gap   = 1
```

So in the focused route-key diagnostic:

```text
A keeps route 2 with positive gap
B keeps route 0 with positive gap
A remains solved
and the second B replay step improves B further (28/60 -> 27/57)
```

This changes the selector conclusion materially:

```text
the remaining bottleneck was not route-local decoder storage;
it was route-address representation in the selector;
and an associative route-key selector can resolve it in the focused case.
```

This is still only a focused diagnostic, not yet a general selector mechanism.
But it is the strongest multi-example routing result so far:

```text
frozen decoder storage works
associative selector routing works
the focused A -> B -> A replay check stays stable
```

Latest focused diagnostic update: add C to route-key selector
-------------------------------------------------------------

The next stress test kept the same route-key selector setup and introduced a
third example `C` with its own probe and target, while keeping the `A/B`
decoder state and route assignments fixed.

Focused `A/B/C` result:

```text
A route = 2
B route = 0
C route = 1

before C training:
        A selected route = 2, gap = 1
        B selected route = 0, gap = 1
        C selected route = 1, gap = 1

after C training:
        A selected route = 2, gap = 1
        B selected route = 0, gap = 1
        C selected route = 1, gap = 1
```

Decoder effects:

```text
B after A->B->A->B:
        forced distance = 27
        forced deficit  = 57

C before training:
        forced distance = 15
        forced deficit  = 0

C training:
        replay_c_flips = 0

C after training:
        forced distance = 15
        forced deficit  = 0
```

So the current `C` result is:

```text
the route-key selector can allocate a third route cleanly
without collapsing the existing A/B assignments or gaps;
but this particular C case does not yet improve decoder storage.
```

That is still useful because it separates two questions:

```text
route allocation / addressing for C: yes
route-local decoder improvement for C: not yet in this focused case
```

So the route-key branch now supports the stronger claim:

```text
associative route addressing scales past the first two examples
in the focused diagnostic,
but decoder training for newly claimed routes is still example-dependent.
```
