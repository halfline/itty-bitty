# Itty Bitty

Itty Bitty is an experimental C codebase for bit-native model execution. The
current project has a small runtime core and a separate experiments area. The
runtime is the part in `src/`; archived and active research prototypes live in
`experiments/`.

## Layout

- `src/`: current runtime, CLI, and default tests.
- `experiments/feed-model-lab/`: archived monolithic feed-model training lab.
- `experiments/route-key-selector/`: focused associative route-key prototype.
- `experiments/self-describing-gray-payload/`: fixed-header Gray payload prototype.
- `experiments/route-key-gray-payload/`: combined ownership-plus-decoder prototype.
- `notes/`: long-form lab notes and historical architecture narrative.

See [ARCHITECTURE.md](/run/host/var/srv/sources/github/itty-bitty/ARCHITECTURE.md) for the
current architecture summary and
[notes/ARCHITECTURE-feed-model-lab.md](/run/host/var/srv/sources/github/itty-bitty/notes/ARCHITECTURE-feed-model-lab.md)
for the archived feed-model notebook.

## Build

```sh
meson setup build
meson compile -C build
```

Meson options:

- `-Dexperimental_feed_model=true`: build the archived feed-model lab and its
  tests.
- `-Dexperimental_prototypes=true`: build the small standalone prototypes.

## Run

The CLI is still demo-oriented. It can:

- encode tokenized input into a bit-string context file,
- load a simple model file,
- run the network,
- decode the selected output through the vocabulary.

Example:

```sh
printf " a\n b\n" > vocabulary.txt
dd if=/dev/urandom of=vocab.bin count=2 bs=8
printf " a" | ./build/itty-bitty vocabulary.txt vocab.bin context.bin
dd if=/dev/urandom of=model.bin count=4 bs=8
./build/itty-bitty vocabulary.txt vocab.bin model.bin context.bin 2 2
```

## Status

The runtime core is the part to extend carefully:

- bit strings and bit-string lists,
- mmap-backed bit-string files,
- vocabulary,
- network and inference,
- exec-buffer and manager/work-queue machinery,
- affinity and runtime metrics.

The feed-model training system is preserved as research history, not as the
current public design.
