# Itty Bitty

Itty Bitty is an experimental C codebase for bit-native model execution. The
current project has a small runtime core and a separate experiments area. The
runtime is the part in `src/`; focused prototypes live in `experiments/`.

## Layout

- `src/`: current runtime, CLI, and default tests.
- `experiments/route-key-selector/`: focused associative route-key prototype.
- `experiments/self-describing-gray-payload/`: fixed-header Gray payload prototype.
- `experiments/route-key-gray-payload/`: combined ownership-plus-decoder prototype.

See [ARCHITECTURE.md](/run/host/var/srv/sources/github/itty-bitty/ARCHITECTURE.md) for the
current architecture summary.

## Build

```sh
meson setup build
meson compile -C build
```

Meson option:

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

The active design surface is the runtime plus the three focused prototypes in
`experiments/`.
