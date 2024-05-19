# Itty Bitty

The Itty Bitty project is an innovative exploration into neural network architectures that utilize bit string manipulations instead of traditional floating-point vectors. This project aims to investigate the feasibility and performance of bitwise operations as the fundamental building blocks for neural networks, potentially offering significant computational efficiency and resource savings.

##  Motivation

Traditional neural networks rely heavily on floating-point arithmetic to perform computations. However, floating-point operations can be computationally intensive and resource-demanding. The Itty Bitty project proposes an alternative approach, leveraging the simplicity and efficiency of bitwise operations to perform neural network computations. This method has the potential to significantly reduce computational overhead, making neural networks more efficient and accessible, especially in resource-constrained environments.

##  Overview
###   Bit String Manipulation

The project includes a tiny runtime library for creating and manipulating bit strings. Bit strings are sequences of bits (binary digits), which are the simplest form of data representation in computing. By using bitwise operations, we can potentially perform complex manipulations efficiently.

###   Neural Network Simulation

On top of the bit string library, the project builds a basic neural network simulation. This neural network uses layers of nodes, where each node performs operations on bit strings instead of traditional numerical vectors. The network processes inputs through multiple layers, each performing bitwise manipulations and transformations. It is the hope that this network will become fleshed out enough to validate or invalidate the proposed idea.

####    Key Concepts

  - Bit Masks Instead of Weights:
    - Each node uses bit masks applied with XNOR operations instead of traditional weights. This allows inputs to "vote" on which bits make it through by looking at the majority.

  - Self-Attention Mechanism:
    - Future implementations will use the pop count of XNOR results to find the Hamming similarity between two inputs, replacing traditional methods like dot product or cosine similarity.

  - Increasing Bit Precision:
    - At each layer of the network, the number of bits is doubled, increasing the precision available for modeling relationships as the network goes deeper.

  - Training with Bitwise Operations:
    - Training is not yet implemented, but the proposed approach will not use gradient descent. Instead, it will leverage XOR operations to measure the difference between inferred and supervised answers. This difference will be propagated backward by folding the result in half at each layer and propagating only what is shared between both halves.

## Building the Project

To build the project, use the following command:

```sh
meson _build
cd _build; ninja
```

## Running the Project

To run the project, use the following command:

```sh
./itty_bitty <bit_string_file> <number_of_layers> <nodes_per_layer>
```

For example:

```sh
dd if=/dev/urandom count=10 of=model
./itty_bitty model 3 4
```

This command will create a neural network with 3 layers and 4 nodes per layer, using the bit strings from `model`

## Example Use Case

The project includes a demonstration of the bit string and neural network libraries. The example reads bit strings from a file, constructs a neural network with the specified number of layers and nodes, and processes the bit strings through the network.

## Future Directions

The Itty Bitty project is still in its early stages. Future work will focus on:

- Completion: We've to reach a minimum viable product. We need to add training and different kinds of blocks for the network
 - Training Implementation: Developing a training mechanism based on XOR operations and backward propagation using bitwise folding.
 - Advanced Features: Implementing self-attention mechanisms using Hamming similarity measures.
- Optimization: Exploring various optimization techniques to enhance the efficiency (including using multiple threads for computation and potentially investigating intrinsics)
- Performance Evaluation: Comparing the performance of bitwise neural networks against traditional floating-point neural networks.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
Contributing

This is just an experiment to see the viability of a kooky idea, it's not really a full fledged project looking for contributors.

## Authors

Ray Strode

We hope you find the Itty Bitty project an intriguing and valuable exploration into alternative neural network architectures!
