using DecisionMakingPolicies
using Test
import Zygote: gradient, pullback
import LuxCore
import Lux
using Random, Distributions
using ComponentArrays


@testset "Lux Network Tests" begin
    num_inputs = 2
    num_hidden = 5
    num_actions = 3

    # test for softmax
    basis = Lux.Chain(Lux.Dense(num_inputs => num_hidden, Lux.relu))
    l = LinearSoftmax(num_hidden, num_actions)
    policy = DeepPolicy(basis, l)
    rng = Random.default_rng()
    ps, st = setup(rng, policy)
    ps = ps |> ComponentArray

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
    netout, stb = basis(x32, ps.basis, st.basis)
    logpl, gl, stl = grad_logpdf(l, netout, a, ps.linear, st.linear)
    @test logp ≈ logpl
    @test grad.linear ≈ gl
    @test grad.basis ≈ zero(grad.basis)
    @. ps.linear = randn() / (num_hidden * num_actions)
    logpl, gl, stl = grad_logpdf(l, netout, a, ps.linear, st.linear)
    logp, grad, st = grad_logpdf(policy, x32, a, ps, st)
    @test logp ≈ logpl
    @test grad.linear ≈ gl
    @test !(grad.basis ≈ zero(grad.basis))

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


    X = randn(Float32, num_inputs, 10)
    A = rand(1:num_actions, 10)
    gref = zero(ps)
    logps = []
    for i in 1:10
        logp, grad, st = grad_logpdf(policy, X[:, i], A[i], ps, st)
        gref += grad
        push!(logps, logp)
    end
    logp, grad, st = grad_logpdf(policy, X, A, ps, st)
    @test logp ≈ sum(logps)
    @test grad ≈ gref

    logp, back = pullback(ps -> sum(logpdf(policy, X, A, ps, st)[1]), ps)
    grad = back(one(logp))[1]
    @test grad ≈ gref
    @test eltype(grad) == eltype(ps)
    @test eltype(logp) == eltype(ps)
    @test logp ≈ sum(logps)

    # test for normal policy
    l = LinearNormal(num_hidden, num_actions)
    policy = DeepPolicy(basis, l)
    rng = Random.default_rng()
    ps, st = setup(rng, policy)
    ps = ps |> ComponentArray

    x32 = randn(Float32, num_inputs)
    x64 = randn(Float64, num_inputs)
    d32, st_post = policy(x32, ps, st)
    @test eltype(d32.μ) == eltype(ps)
    d64, st_post = policy(x64, ps, st)
    @test eltype(d64.μ) == eltype(ps)

    a32 = zeros(Float32, num_actions)
    a64 = zeros(Float64, num_actions)
    logp, grad, st = grad_logpdf(policy, x32, a32, ps, st)
    @test eltype(logp) == eltype(ps)
    @test eltype(grad) == eltype(ps)
    netout, stb = basis(x32, ps.basis, st.basis)
    logpl, gl, stl = grad_logpdf(l, netout, a32, ps.linear, st.linear)
    @test logp ≈ logpl
    @test grad.linear ≈ gl
    @test grad.basis ≈ zero(grad.basis)
    @. ps.linear.weight = randn() / (num_hidden * num_actions)
    logpl, gl, stl = grad_logpdf(l, netout, a32, ps.linear, st.linear)
    logp, grad, st = grad_logpdf(policy, x32, a32, ps, st)
    @test logp ≈ logpl
    @test grad.linear ≈ gl
    @test !(grad.basis ≈ zero(grad.basis))

    logp, grad, st = grad_logpdf(policy, x32, a32, ps, st)
    grada = gradient(p -> logpdf(policy, x32, a32, p, st)[1], ps)[1]
    @test grad ≈ grada
    @test eltype(grada) == eltype(ps)

    a32, logp, grad = sample_with_trace(policy, x32, ps, st)
    @test eltype(logp) == eltype(ps)
    @test eltype(grad) == eltype(ps)
    logpa, grada, _ = grad_logpdf(policy, x32, a32, ps, st)
    @test logp ≈ logpa
    @test grad ≈ grada

    bsize = 10
    X = randn(Float32, num_inputs, bsize)
    A = randn(Float32, num_actions, bsize)
    gref = zero(ps)
    logps = []
    for i in 1:bsize
        logp, grad, st = grad_logpdf(policy, X[:, i], A[:, i], ps, st)
        @. gref += grad
        push!(logps, logp)
    end

    logp, grad, st = grad_logpdf(policy, X, A, ps, st)
    @test logp ≈ sum(logps)
    @test grad ≈ gref

    logp, back = pullback(ps -> sum(logpdf(policy, X, A, ps, st)[1]), ps)
    grad = back(one(logp))[1]
    @test grad ≈ gref
    @test eltype(grad) == eltype(ps)
    @test eltype(logp) == eltype(ps)
    @test logp ≈ sum(logps)
end