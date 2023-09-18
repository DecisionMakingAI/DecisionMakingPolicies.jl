using DecisionMakingPolicies
using Test
import Zygote: gradient, pullback

using Random, Distributions
using ComponentArrays

zeros32(::AbstractRNG, dims...) = zeros(Float32, dims...)


@testset "Lux Network Tests" begin
    num_inputs = 2
    num_hidden = 2
    num_actions = 3
    basis(x) = x .* 2 .+ 1
    l = LinearSoftmax(num_hidden, num_actions)
    policy = LinearPolicyWithBasis(basis, l)
    rng = Random.default_rng()
    ps, st = setup(rng, policy)
    ps = ps |> ComponentArray
    ps2, st2 = setup(rng, l)
    ps2 = ps2 |> ComponentArray
    @test ps == ps2
    @test st == st2

    x32 = randn(Float32, num_inputs)
    x64 = randn(Float64, num_inputs)
    d32, st_post = policy(x32, ps, st)
    @test eltype(d32.p) == eltype(ps)
    d64, st_post = policy(x64, ps, st)
    @test eltype(d64.p) == eltype(ps)

    a = 1
    logp, grad, st = grad_logpdf(policy, x32, a, ps, st)
    @test eltype(logp) == eltype(ps)
    @test eltype(grad) == eltype(ps)
    basisout = basis(x32)
    logpl, gl, stl = grad_logpdf(l, basisout, a, ps, st)
    @test logp ≈ logpl
    @test grad ≈ gl
    @. ps = randn() / (num_hidden * num_actions)
    logpl, gl, stl = grad_logpdf(l, basisout, a, ps, st)
    logp, grad, st = grad_logpdf(policy, x32, a, ps, st)
    @test logp ≈ logpl
    @test grad ≈ gl

    logp, grad, st = grad_logpdf(policy, x32, a, ps, st)
    grada = gradient(p -> logpdf(policy, x32, a, p, st)[1], ps)[1]
    @test grad ≈ grada
    @test eltype(grada) == eltype(ps)
    
    
    a, logp, grad = sample_with_trace(policy, x32, ps, st)
    @test eltype(logp) == eltype(ps)
    @test eltype(grad) == eltype(ps)
    logpa, grada, _ = grad_logpdf(policy, x32, a, ps, st)
    @test logp ≈ logpa
    @test grad ≈ grada


    # test with normal policy
    l = LinearNormal(num_hidden, num_actions)
    policy = LinearPolicyWithBasis(basis, l)
    ps, st = setup(rng, policy)
    ps = ps |> ComponentArray
    ps2, st2 = setup(rng, l)
    ps2 = ps2 |> ComponentArray
    @test ps == ps2
    @test st == st2

    x32 = randn(Float32, num_inputs)
    x64 = randn(Float64, num_inputs)
    d32, st_post = policy(x32, ps, st)
    @test eltype(d32.μ) == eltype(ps)
    d64, st_post = policy(x64, ps, st)
    @test eltype(d64.μ) == eltype(ps)

    a = zeros(Float32, num_actions)
    logp, grad, st = grad_logpdf(policy, x32, a, ps, st)
    @test eltype(logp) == eltype(ps)
    @test eltype(grad) == eltype(ps)
    basisout = basis(x32)
    logpl, gl, stl = grad_logpdf(l, basisout, a, ps, st)
    @test logp ≈ logpl
    @test grad ≈ gl
    @. ps.weight = randn() / (num_hidden * num_actions)
    logpl, gl, stl = grad_logpdf(l, basisout, a, ps, st)
    logp, grad, st = grad_logpdf(policy, x32, a, ps, st)
    @test logp ≈ logpl
    @test grad ≈ gl

    logp, grad, st = grad_logpdf(policy, x32, a, ps, st)
    grada = gradient(p -> logpdf(policy, x32, a, p, st)[1], ps)[1]
    @test grad ≈ grada
    @test eltype(grada) == eltype(ps)

    a, logp, grad = sample_with_trace(policy, x32, ps, st)
    @test eltype(logp) == eltype(ps)
    @test eltype(grad) == eltype(ps)
    logpa, grada, _ = grad_logpdf(policy, x32, a, ps, st)
    @test logp ≈ logpa
    @test grad ≈ grada
end