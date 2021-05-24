# YaoFlux
Flux models to use for Quantum ML.

## Short version
This Julia package allows you to seemlessly plug differentiable quantum circuits into your [FluxML](https://fluxml.ai) models.
This allows you to perform Quantum Machine Learning (QML) in Julia. In my (not thoroughly benchmarked) experience, this is orders of magnitude
faster than any other QML platform I am aware of (that's `pennylane` and `qiskit`).
We aim to expand the code in this repository as use cases arise. Hopefully other people will find this useful too -- contributions always welcome!

## Long version
[`Yao.jl`](https://github.com/QuantumBFS/Yao.jl) comes with AD support to differentiate circuits, enabling Quantum Machine Learning (QML) applications.
However, this support isn't integrated into Julia's "native" AD, meaning that bindings must always be written first before quantum circuits
can be differentiated.

The bindings provided here are specifically meant for use with the `FluxML` Julia package.
The `Circuit` struct defines a QML model as they are typically used in the literature:
it is composed of two Yao `AbstractBlock`s, i.e. two quantum circuits: the _encoding_ circuit and the _variational_ circuit.
The model is the quantum circuit made of the composition of the encoding and variational circuit, initialised in the zero state.
The free parameters of the encoding circuit are used to encode the input data, whilst the free parameters of the variational circuit
are the trainable weights of the ML model.
More precisely, given an input `x`, the model is evaluated as follows:
 1. a quantum state |ψ> is initialised to the zero input state |00..0> (the number of qubits being set equal to the circuit size);
 2. |ψ> is sent through the encoding circuit, with the free parameters of the encoding circuit set to the input `x`;
 3. |ψ> is then sent through the variational circuit, with the free parameters forming the trainable parameters `θ`.
Computing the gradient of the model will give the gradient with respect to the free parameters `θ` of the variational circuit.

### TODO Features for the future
Fixing non-trainable parameters, avoiding copying circuits over and over (this is due to Yao's `dispatch!` semantics),
advanced techniques such as data reuploading.

### Available bindings
The [Yao project](https://yaoquantum.org) has published various such bindings in two places to my knowledge:
as part of their [`QuAlgorithmZoo`](https://github.com/QuantumBFS/QuAlgorithmZoo.jl/blob/master/examples/PortZygote/zygote_patch.jl)
examples,
as well as in the separate repository [`YaoAD`](https://github.com/QuantumBFS/YaoAD.jl).
This package is inspired from those, but meant to be easier to use in conjunction with Flux. I have run into issues using these bindings
before and should probably submit an issue about it at some point.