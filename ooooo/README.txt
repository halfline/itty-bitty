Generated gray-payload demo files

Files:
  vocabulary.txt         token text file
  vocabulary.bin         token payload bit string
  context.bin            encoded input context
  model.bin              three-layer one-node zero mask model
  expected-payload.bin   expected decoded payload

Network shape:
  layer 1: input adapter
  layer 2: middle layer
  layer 3: final output layer

Suggested commands:
  itty-bitty infer ooooo/vocabulary.txt ooooo/vocabulary.bin ooooo/model.bin ooooo/context.bin 64 3 1
  itty-bitty-trainer run ooooo/model.bin ooooo/context.bin ooooo/expected-payload.bin 64 3 1
