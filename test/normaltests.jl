using DecisionMakingPolicies
using DecisionMakingPolicies: comp_sigma
using Test
import Zygote: gradient, pullback
import LuxCore
using ChainRulesTestUtils
using ChainRulesCore
using Random, Distributions
using ComponentArrays
using LinearAlgebra

zeros32(::AbstractRNG, dims...) = zeros(Float32, dims...)

@testset "Normal rules" begin
    rng = Random.default_rng()
    num_actions = 2
    init_std32(rng, dims...) = ones(Float32, dims...) * -2.1f0
    init_std64(rng, dims...) = ones(Float64, dims...) * -2.1
    l32b = StatelessNormal(num_actions, init_mean=zeros32, init_std=init_std32, buffered=true)
    l64b = StatelessNormal(num_actions, init_mean=zeros, init_std=init_std64, buffered=true)
    ps32b, st32b = setup(rng, l32b)
    ps64b, st64b = setup(rng, l64b)
    test_rrule(comp_sigma, ps32b.logσ, st32b ⊢ NoTangent(), rtol=1f-2)
    test_rrule(comp_sigma, ps64b.logσ, st64b ⊢ NoTangent(), rtol=1e-2)
end

@testset "Stateless Normal Tests" begin
    for num_actions = [1,3]
        for (T, initf) in zip([Float32, Float64], [zeros32, LuxCore.zeros])
            p1 = StatelessNormal(num_actions, init_mean=initf, init_std=initf)
            p2 = StatelessNormal(num_actions, init_mean=initf, init_std=initf, buffered=true)

            rng = Random.default_rng()
            
            ps1, st1 = LuxCore.setup(rng, p1)
            ps1 = ps1 |> ComponentArray
            
            ps2, st2 = LuxCore.setup(rng, p2)
            ps2 = ps2 |> ComponentArray

            d1, st1_post = p1(ps1, st1)
            d2, st2_post = p2(ps2, st2)
            if num_actions == 1
                @test typeof(d1) <: Normal{T}
                @test eltype(d1.μ) == T
                @test all(isapprox.(d1.μ,  ps1.μ))
                @test all(isapprox.(d1.σ,  exp(ps1.logσ)))
                @test typeof(d2) <: Normal{T}
                @test eltype(d2.μ) == T
                @test all(isapprox.(d2.μ,  ps2.μ))
                @test all(isapprox.(d2.σ,  exp(ps2.logσ)))
            else 
                @test typeof(d1) <: MvNormal{T}
                @test eltype(d1.μ) == T
                @test all(isapprox.(d1.μ,  ps1.μ))
                @test all(isapprox.(diag(d1.Σ),  map(abs2, exp.(ps1.logσ))))
                @test typeof(d2) <: MvNormal{T}
                @test eltype(d2.μ) == T
                @test all(isapprox.(d2.μ,  ps2.μ))
                @test all(isapprox.(diag(d2.Σ),  map(abs2, exp.(ps2.logσ))))
            end
            x = randn(T, 20)
            d1p, st1_postp = p1(x, ps1, st1)
            @assert d1p == d1
            d2p, st2_postp = p2(x, ps2, st2)
            @assert d2p == d2

            # test logp of first action
            if num_actions == 1
                a = one(T)
            else
                a = ones(T, num_actions)
            end
            logp, st1_post = logpdf(p1, a, ps1, st1)
            @test typeof(logp) == T
            @test isapprox(logp, logpdf(d1, a))

            # test for ignoring observation input
            logpp, st1_postp = logpdf(p1, x, a, ps1, st1)
            @test typeof(logpp) == T
            @test logpp ≈ logp
            
            
            # test logp of first action with buffered layer
            logp, st2_post = logpdf(p2, a, ps2, st2)
            @test typeof(logp) == T
            @test isapprox(logp, logpdf(d1, a))

            # test for ignoring observation input
            logpp, st2_postp = logpdf(p2, x, a, ps2, st2)
            @test typeof(logpp) == T
            @test logpp ≈ logp
            


            # test gradient of logp of first action
            ga1 = gradient(ps -> logpdf(p1, a, ps, st1)[1], ps1)[1]
            logp1, gl1, st1_post = grad_logpdf(p1, a, ps1, st1)
            @test typeof(logp1) == T
            @test eltype(gl1) == T
            @test eltype(ga1) == T
            @test typeof(ga1) <: ComponentArray{T}
            @test typeof(gl1) <: ComponentArray{T}
            @test isapprox(gl1, ga1)

            # test for ignoring observation input
            logp1p, gl1p, st1_postp = grad_logpdf(p1, x, a, ps1, st1)
            @test typeof(logp1p) == T
            @test eltype(gl1p) == T
            @test isapprox(gl1, gl1p)
            @test isapprox(logp1, logp1p)

            ga2 = gradient(ps -> logpdf(p2, a, ps, st2)[1], ps2)[1]
            logp2, gl2, st2_post = grad_logpdf(p2, a, ps2, st2)
            @test typeof(logp2) == T
            @test eltype(gl2) == T
            @test eltype(ga2) == T
            @test typeof(ga2) <: ComponentArray{T}
            @test typeof(gl2) <: ComponentArray{T}
            @test isapprox(gl2, ga2)
            @test isapprox(st2_post.ψ, gl2)

            # test for ignoring observation input
            logp2p, gl2p, st2_postp = grad_logpdf(p2, x, a, ps2, st2)
            @test typeof(logp2p) == T
            @test eltype(gl2p) == T
            @test isapprox(ga2, gl2p)
            @test isapprox(logp2, logp2p)


            # test sample with trace
            a, logp1s, ψ, st1_post = sample_with_trace(p1, ps1, st1)
            logp1, gl1, st1_post = grad_logpdf(p1, a, ps1, st1)
            @test eltype(a) == T
            if num_actions == 1
                @test a isa T
            else
                @test a isa Vector{T}
            end
            @test typeof(logp1s) == T
            @test typeof(ψ) <: ComponentArray{T}
            @test logp1s ≈ logp1
            @test ψ ≈ gl1

            # test ignoring observation input
            a, logp1sp, ψp, st1_postp = sample_with_trace(p1, x, ps1, st1)
            # no error means we are ok

            # test sample with trace using buffer
            a, logp2s, ψ, st2_post = sample_with_trace(p2, ps2, st2)
            logp1, gl1, st1_post = grad_logpdf(p1, a, ps1, st1)
            @test eltype(a) == T
            if num_actions == 1
                @test a isa T
            else
                @test a isa Vector{T}
            end
            @test typeof(logp2s) == T
            @test typeof(ψ) <: ComponentArray{T}
            @test logp2s ≈ logp1
            @test ψ ≈ gl1

            # test ignoring observation input
            a, logp2sp, ψp, st2_postp = sample_with_trace(p2, x, ps2, st2)

            # test auto grad p1 including observation input to be ignored
            g1 = gradient(ps -> logpdf(p1(x, ps, st1)[1], a), ps1)[1]
            g2 = gradient(ps -> logpdf(p1, x, a, ps, st1)[1], ps1)[1]
            g3 = grad_logpdf(p1, a, ps1, st1)[2]
            @test eltype(g1) == T
            @test eltype(g2) == T
            @test eltype(g3) == T
            @test isapprox(g1, g2)
            @test isapprox(g1, g3)

            # test auto grad p2
            g2 = gradient(ps -> logpdf(p2, a, ps, st2)[1], ps2)[1]
            @test eltype(g2) == T
            @test isapprox(g1, g2)
            g3 = grad_logpdf(p2, a, ps2, st2)[2]
            @test eltype(g3) == T
            @test isapprox(g1, g3)

            A32 = randn(Float32, num_actions, 9)
            A64 = randn(Float64, num_actions, 13)

            for a in [A32, A64]
                prob1, st1_post = logpdf(p1, a, ps1, st1)
                prob2, st2_post = logpdf(p2, a, ps2, st2)
                if num_actions == 1
                    prob3 = [logpdf(p1, a[1, i], ps1, st1)[1] for i in axes(a,2)]
                else
                    prob3 = [logpdf(p1, a[:, i], ps1, st1)[1] for i in axes(a,2)]
                end
                @test eltype(prob1) == T
                @test eltype(prob2) == T
                @test length(prob1) == length(prob2)
                @test isapprox(prob1, prob2)
                @test isapprox(prob1, prob3)

                logp1, g1, st1_post = grad_logpdf(p1, a, ps1, st1)
                logp2, g2, st2_post = grad_logpdf(p2, a, ps2, st2)
                # g3 = vec(sum(reduce(hcat, [copy(grad_logpdf(p1, a[i], ps1, st1)[2]) for i in eachindex(a)]), dims=2))
                g3 = zero(ps1)
                for i in axes(a, 2)
                    if num_actions == 1
                        g3 .+= grad_logpdf(p1, a[1,i], ps1, st1)[2]
                    else
                        g3 .+= grad_logpdf(p1, a[:, i], ps1, st1)[2]
                    end
                end
                @test eltype(logp1) == T
                @test eltype(logp2) == T
                @test eltype(g1) == T
                @test eltype(g2) == T
                @test length(logp1) == length(logp2)
                @test length(g1) == length(g2)
                @test isapprox(logp1, logp2)
                @test isapprox(g1, g2)
                @test isapprox(g1, g3)

                g1 = gradient(ps -> sum(logpdf(p1, a, ps, st1)[1]), ps1)[1]
                g2 = gradient(ps -> sum(logpdf(p2, a, ps, st2)[1]), ps2)[1]
                @test eltype(g1) == T
                @test eltype(g2) == T
                @test isapprox(g1, g2)
                @test isapprox(g1, g3)
            end
        end
    end
end
