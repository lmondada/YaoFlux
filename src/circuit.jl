using Yao
using YaoBlocks.AD: apply_back
using ChainRulesCore
using Flux

# helper function with home-grown derivative
function apply_circuit(reg, params; circ::AbstractBlock)
    circ = deepcopy(circ)
    dispatch!(circ, params)
    ψ = ArrayReg(copy(reg))
    ψ |> circ
    state(ψ)
end

# derivative of apply_circuit
function ChainRulesCore.rrule(::typeof(apply_circuit),
        reg, params; circ::AbstractBlock)
    circ = deepcopy(circ)
    out = apply_circuit(reg, params; circ)
    out, function apply_circuit_pullback(outδ)
        dispatch!(circ, params)
        (_, regδ), paramsδ = apply_back((ArrayReg(copy(out)), ArrayReg(copy(outδ))), circ)
        return (NoTangent, state(regδ), paramsδ)
    end
end

# empty circuit
empty_circ = chain()
# zero state
zero_ket(nqb) = reshape([convert(ComplexF64, i == 1) for i = 1:2^nqb], 2^nqb, 1)

struct Circuit
    θ ::Vector
    circ ::AbstractBlock
    enccirc ::AbstractBlock
end

# constructors
Circuit(circ ::AbstractBlock) = Circuit(
    zeros(nparameters(circ)),
    circ,
    empty_circ(nqubits(circ))
)
Circuit(circ ::AbstractBlock, enccirc ::AbstractBlock) = Circuit(
    zeros(nparameters(circ)),
    circ,
    enccirc
)

# model call
function (m::Circuit)(x)
    init = zero_ket(nqubits(m.enccirc))
    enc = apply_circuit(init, x, circ=m.enccirc)
    ret = apply_circuit(enc, m.θ, circ=m.circ)
end

# trainable parameters
Flux.trainable(m::Circuit) = (m.θ,)

# other overloads
Yao.nparameters(m::Circuit) = nparameters(m.circ)