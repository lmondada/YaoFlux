using Test, YaoFlux
using Yao, Flux
using FiniteDifferences
using LinearAlgebra: norm
using Random

nonzero = xs -> findall(x -> abs2(x) > ϵ, reshape(xs, length(xs)))
ϵ = 1e-7

@testset "Circuit model" begin
    circ = chain(3, put(1 => H), cnot(1, 2), cnot(1, 3), put(2 => Rx(0.)))

    varcirc = Circuit(circ)
    out = varcirc([])
    @test nonzero(out) == [1, 8]

    ps = params(varcirc)
    @test length(ps) == 1
    @test length(ps[1]) == nparameters(circ)
    @test nparameters(varcirc) == nparameters(circ)

    target_ψ = √2/2 * [1, 0, 0, 0, 0, 0, 0, 1]
    loss(x) = norm(varcirc(x) - target_ψ)
    grad = Flux.gradient(ps) do
        train_loss = loss([])
    end
    @test all(grad[varcirc.θ] .< ϵ)
end

@testset "apply_circuit computes gradients correctly" begin
    Random.seed!(3)
    circ = chain(3, put(1 => Rx(0.)), cnot(1, 2), cnot(1, 3), put(2 => Rx(0.)))
    reg = ArrayReg(rand(8, 1) + 1im * rand(8, 1))
    normalize!(reg)
    params = rand(2)
    f = θ -> to_vec(YaoFlux.apply_circuit(reg, θ; circ))[1]
    fdm_jac = jacobian(central_fdm(5, 1), f, params)[1]
    exact_jac = Flux.jacobian(f, params)[1]
    @test norm(fdm_jac - exact_jac) < ϵ
end