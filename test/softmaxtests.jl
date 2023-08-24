using DecisionMakingPolicies
using DecisionMakingPolicies: pdf_softmax, logpdf_softmax, softmax!, softmax
using Test
import Zygote: gradient, pullback
import LuxCore
using ChainRulesTestUtils
using Random, Distributions
using ComponentArrays
using DistributionsAD

# test for types correctness. 
# test for chaging parameters
# test for correct probability distribution
# test for correct logpdf
# test for gradients through logpdf
# test for logpdf and gradient output
# test for correct gradients and sample with trace
# test for both with and without buffered input
# test for autodiff gradient
# test for both single sample and batch sample
# test for tabular (integer) input

zeros32(::AbstractRNG, dims...) = zeros(Float32, dims...)
mysoftmax(x) = exp.(x) ./ sum(exp.(x))

@testset "Softmax rules" begin
    x32 = collect(Float32, 1:3)
    x64 = collect(Float64, 1:3)
    test_rrule(softmax, x32, rtol=1f-3)
    test_rrule(softmax, x64)

    x32 = zeros(Float32, 3)
    x64 = zeros(Float64, 3)
    test_rrule(softmax, x32, rtol=1f-3)
    test_rrule(softmax, x64)

    θ32 = randn(Float32, 3, 2)
    θ64 = randn(Float64, 3, 2)
    x32 = randn(Float32, 3)
    x64 = randn(Float64, 3)
    test_rrule(pdf_softmax, θ32, x32, rtol=1f-3)
    test_rrule(pdf_softmax, θ64, x64)
    test_rrule(logpdf_softmax, θ32, x32, 1, rtol=1f-3)
    test_rrule(logpdf_softmax, θ64, x64, 1)   
    
    x32 = randn(Float32, 3, 11)
    x64 = randn(Float64, 3, 15)
    A32 = rand(1:2, 11)
    A64 = rand(1:2, 15)
    test_rrule(pdf_softmax, θ32, x32, rtol=1f-3)
    test_rrule(pdf_softmax, θ64, x64)
    test_rrule(logpdf_softmax, θ32, x32, A32, rtol=1f-3)
    test_rrule(logpdf_softmax, θ64, x64, A64)
end

@testset "Linear Softmax Tests" begin
    num_actions = 2
    num_inputs = 3
    for (T, initf) in zip([Float32, Float64], [zeros32, LuxCore.zeros])
        p1 = LinearSoftmax(num_inputs, num_actions, init_weights=initf)
        p2 = LinearSoftmax(num_inputs, num_actions, init_weights=initf, buffered=true)

        rng = Random.default_rng()
        
        
        ps1, st1 = LuxCore.setup(rng, p1)
        ps1 = ps1 |> ComponentArray
        
        ps2, st2 = LuxCore.setup(rng, p2)
        ps2 = ps2 |> ComponentArray

        x32 = collect(Float32, 1:num_inputs)
        x64 = collect(Float64, 1:num_inputs)
        for x in [x32, x64]
            fill!(ps1.weight, 0.0)
            fill!(ps2.weight, 0.0)
            d1, st1_post = p1(x, ps1, st1)
            @test typeof(d1) <: Categorical{T}
            @test eltype(d1.p) == T
            @test all(isapprox.(d1.p,  ones(T, num_actions)/num_actions))

            d2, st2_post = p2(x, ps2, st2)
            @test typeof(d2) <: Categorical{T}
            @test eltype(d2.p) == T
            @test all(isapprox.(d2.p,  ones(T, num_actions)/num_actions))
            @test all(isapprox.(st2_post.p, d2.p))

            ps1.weight[1,1] = 1.0
            ps1.weight[2,2] = 2.0
            ps1.weight[3,1] = 0.5
            ps2 .= ps1
            
            p = mysoftmax([ps1.weight[1,1] * x[1] + ps1.weight[3,1] * x[3], ps1.weight[2,2] * x[2]])

            # test logp of first action
            logp, st1_post = logpdf(p1, x, 1, ps1, st1)
            @test typeof(logp) == T
            @test isapprox(logp, log(p[1]))
            # test logp of second action
            logp, st1_post = logpdf(p1, x, 2, ps1, st1)
            @test typeof(logp) == T
            @test isapprox(logp, log(p[2]))

            # test logp of first action with buffered layer
            logp, st2_post = logpdf(p2, x, 1, ps2, st2)
            @test typeof(logp) == T
            @test isapprox(logp, log(p[1]))
            @test all(isapprox.(st2_post.p, p))

            # test logp of second action with buffered layer
            logp, st2_post = logpdf(p2, x, 2, ps2, st2)
            @test typeof(logp) == T
            @test isapprox(logp, log(p[2]))
            @test all(isapprox.(st2_post.p, p))

            # test gradient of logp of first action
            ga1 = gradient(ps -> logpdf(p1, x, 1, ps, st1)[1], ps1)[1]
            logp1, gl1, st1_post = grad_logpdf(p1, x, 1, ps1, st1)
            @test typeof(logp1) == T
            @test eltype(gl1) == T
            @test eltype(ga1) == T
            @test typeof(ga1) <: ComponentArray{T}
            @test typeof(gl1) <: ComponentArray{T}
            @test isapprox(gl1, ga1)

            ga2 = gradient(ps -> logpdf(p2, x, 2, ps, st2)[1], ps2)[1]
            logp2, gl2, st2_post = grad_logpdf(p2, x, 2, ps2, st2)
            @test typeof(logp2) == T
            @test eltype(gl2) == T
            @test eltype(ga2) == T
            @test typeof(ga2) <: ComponentArray{T}
            @test typeof(gl2) <: ComponentArray{T}
            @test isapprox(gl2, ga2)
            @test isapprox(st2_post.ψ, gl2)

            a, logp1s, ψ, st1_post = sample_with_trace(p1, x, ps1, st1)
            @test typeof(a) == Int
            @test typeof(logp1s) == T
            @test typeof(ψ) <: ComponentArray{T}
            if a == 1
                @test logp1s ≈ logp1
                @test ψ ≈ gl1
            else
                @test logp1s ≈ logp2
                @test ψ ≈ gl2
            end

            a, logp2s, ψ, st2_post = sample_with_trace(p2, x, ps2, st2)
            @test typeof(a) == Int
            @test typeof(logp2s) == T
            @test typeof(ψ) <: ComponentArray{T}
            if a == 1
                @test logp2s ≈ logp1
                @test ψ ≈ gl1
            else
                @test logp2s ≈ logp2
                @test ψ ≈ gl2
            end

            # test auto grad p1
            g1 = gradient(ps -> logpdf(p1(x, ps, st1)[1], 1), ps1)[1]
            g2 = gradient(ps -> logpdf(p1, x, 1, ps, st1)[1], ps1)[1]
            g3 = grad_logpdf(p1, x, 1, ps1, st1)[2]
            @test eltype(g1) == T
            @test eltype(g2) == T
            @test eltype(g3) == T
            @test isapprox(g1, g2)
            @test isapprox(g1, g3)

            # test auto grad p2
            # g1 = gradient(ps -> logpdf(p2(x, ps, st2)[1], 1), ps2)[1]
            g2 = gradient(ps -> logpdf(p2, x, 1, ps, st2)[1], ps2)[1]
            g3 = grad_logpdf(p2, x, 1, ps2, st2)[2]
            @test eltype(g1) == T
            @test eltype(g2) == T
            @test eltype(g3) == T
            @test isapprox(g1, g2)
            @test isapprox(g1, g3)

            
        end

        x32 = randn(Float32, num_inputs, 9)
        x64 = randn(Float64, num_inputs, 13)
        A32 = rand(1:num_actions, 9)
        A64 = rand(1:num_actions, 13)

        for (x,a) in zip([x32, x64], [A32, A64])
            prob1, st1_post = logpdf(p1, x, a, ps1, st1)
            prob2, st2_post = logpdf(p2, x, a, ps2, st2)
            prob3 = [logpdf(p1, vec(x[:, i]), a[i], ps1, st1)[1] for i in axes(x, 2)]
            @test eltype(prob1) == T
            @test eltype(prob2) == T
            @test length(prob1) == length(prob2)
            @test isapprox(prob1, prob2)
            @test isapprox(prob1, prob3)

            logp1, g1, st1_post = grad_logpdf(p1, x, a, ps1, st1)
            logp2, g2, st2_post = grad_logpdf(p2, x, a, ps2, st2)
            g3 = vec(sum(reduce(hcat, [copy(grad_logpdf(p1, x[:, i], a[i], ps1, st1)[2]) for i in axes(x, 2)]), dims=2))
            @test eltype(logp1) == T
            @test eltype(logp2) == T
            @test eltype(g1) == T
            @test eltype(g2) == T
            @test length(logp1) == length(logp2)
            @test length(g1) == length(g2)
            @test isapprox(logp1, logp2)
            @test isapprox(g1, g2)
            @test isapprox(g1, g3)

            g1 = gradient(ps -> sum(logpdf(p1, x, a, ps, st1)[1]), ps1)[1]
            g2 = gradient(ps -> sum(logpdf(p2, x, a, ps, st2)[1]), ps2)[1]
            @test eltype(g1) == T
            @test eltype(g2) == T
            @test isapprox(g1, g2)
            @test isapprox(g1, g3)

        end

    end
end
