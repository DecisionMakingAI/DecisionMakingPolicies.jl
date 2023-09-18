using DecisionMakingPolicies
using DecisionMakingPolicies: pdf_softmax, logpdf_softmax, softmax!, softmax, logpdf_stateless_softmax
using Test
import Zygote: gradient, pullback
import LuxCore
using ChainRulesTestUtils
using ChainRulesCore
using Random, Distributions
using ComponentArrays

zeros32(::AbstractRNG, dims...) = zeros(Float32, dims...)
mysoftmax(x) = exp.(x) ./ sum(exp.(x))

@testset "Softmax rules" begin
    num_inputs = 3
    num_actions = 2
    x32 = collect(Float32, 1:num_inputs)
    x64 = collect(Float64, 1:num_inputs)
    test_rrule(softmax, x32, rtol=1f-2)
    test_rrule(softmax, x64)

    x32 = zeros(Float32, num_inputs)
    x64 = zeros(Float64, num_inputs)
    test_rrule(softmax, x32, rtol=1f-2)
    test_rrule(softmax, x64)

    θ32 = randn(Float32, num_inputs, num_actions)
    θ64 = randn(Float64, num_inputs, num_actions)
    θ32sl = randn(Float32, num_actions) ./ num_actions
    θ64sl = randn(Float64, num_actions) ./ num_actions
    x32 = randn(Float32, num_inputs)
    x64 = randn(Float64, num_inputs)
    test_rrule(pdf_softmax, θ32, x32, rtol=1f-2)
    test_rrule(pdf_softmax, θ64, x64)
    buff32lin = (p=zeros(Float32, num_actions), ψ=(;weight=zeros(Float32, (num_inputs, num_actions))))
    buff64lin = (p=zeros(Float64, num_actions), ψ=(;weight=zeros(Float64, (num_inputs, num_actions))))
    buff32stateless = (p=zeros(Float32, num_actions), ψ=(;weight=zeros(Float32, num_actions)))
    buff64stateless = (p=zeros(Float64, num_actions), ψ=(;weight=zeros(Float64, num_actions)))
    for i in 1:num_actions
        test_rrule(logpdf_softmax, θ32, x32, i, rtol=1f-2)
        test_rrule(logpdf_softmax, θ64, x64, i)   
        test_rrule(logpdf_stateless_softmax, θ32sl, i, rtol=1f-2)
        test_rrule(logpdf_stateless_softmax, θ64sl, i, rtol=1e-2)
        test_rrule(logpdf_softmax, buff32lin ⊢ NoTangent(), θ32, x32, i, rtol=1f-2)
        test_rrule(logpdf_softmax, buff64lin ⊢ NoTangent(), θ64, x64, i, rtol=1e-2)
        test_rrule(logpdf_stateless_softmax, buff32stateless ⊢ NoTangent(), θ32sl, i, rtol=1f-2)
        test_rrule(logpdf_stateless_softmax, buff64stateless ⊢ NoTangent(), θ64sl, i, rtol=1e-2)
    end
    x32 = randn(Float32, num_inputs, 11)
    x64 = randn(Float64, num_inputs, 15)
    A32 = rand(1:num_actions, 11)
    A64 = rand(1:num_actions, 15)
    test_rrule(pdf_softmax, θ32, x32, rtol=1f-2)
    test_rrule(pdf_softmax, θ64, x64)
    test_rrule(logpdf_softmax, θ32, x32, A32, rtol=1f-2)
    test_rrule(logpdf_softmax, θ64, x64, A64)
    test_rrule(logpdf_stateless_softmax, θ32sl, A32, rtol=1f-2)
    test_rrule(logpdf_stateless_softmax, θ64sl, A64, rtol=1e-2)
end

@testset "Linear Softmax Tests" begin
    num_actions = 2
    num_inputs = 3
    for (T, initf) in zip([Float32, Float64], [zeros32, LuxCore.zeros])
        p1 = LinearSoftmax(num_inputs, num_actions, init_weights=initf)
        p2 = LinearSoftmax(num_inputs, num_actions, init_weights=initf, buffered=true)

        rng = Random.default_rng()
        
        
        ps1, st1 = setup(rng, p1)
        ps1 = ps1 |> ComponentArray
        
        ps2, st2 = setup(rng, p2)
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


@testset "Stateless Softmax Tests" begin
    num_actions = 3
    for (T, initf) in zip([Float32, Float64], [zeros32, LuxCore.zeros])
        p1 = StatelessSoftmax(num_actions, init_weights=initf)
        p2 = StatelessSoftmax(num_actions, init_weights=initf, buffered=true)

        rng = Random.default_rng()
        
        
        ps1, st1 = LuxCore.setup(rng, p1)
        ps1 = ps1 |> ComponentArray
        
        ps2, st2 = LuxCore.setup(rng, p2)
        ps2 = ps2 |> ComponentArray

        
        fill!(ps1.weight, 0.0)
        fill!(ps2.weight, 0.0)
        d1, st1_post = p1(ps1, st1)
        @test typeof(d1) <: Categorical{T}
        @test eltype(d1.p) == T
        @test all(isapprox.(d1.p,  ones(T, num_actions)/num_actions))
        x = randn(T, 20)
        d1p, st1_postp = p1(x, ps1, st1)
        @assert d1p == d1

        d2, st2_post = p2(ps2, st2)
        @test typeof(d2) <: Categorical{T}
        @test eltype(d2.p) == T
        @test all(isapprox.(d2.p,  ones(T, num_actions)/num_actions))
        @test all(isapprox.(st2_post.p, d2.p))
        d2p, st2_postp = p2(x, ps2, st2)
        @assert d2p == d2

        ps1.weight[1] = 1.0
        ps1.weight[2] = 2.0
        ps1.weight[3] = 0.5
        ps2 .= ps1
            
        p = mysoftmax(ps1.weight)

        # test logp of first action
        logp, st1_post = logpdf(p1, 1, ps1, st1)
        @test typeof(logp) == T
        @test isapprox(logp, log(p[1]))
        # test logp of second action
        logp, st1_post = logpdf(p1, 2, ps1, st1)
        @test typeof(logp) == T
        @test isapprox(logp, log(p[2]))

        # test for ignoring observation input
        logpp, st1_postp = logpdf(p1, x, 2, ps1, st1)
        @test typeof(logpp) == T
        @test logpp ≈ logp
        

        # test logp of first action with buffered layer
        logp, st2_post = logpdf(p2, 1, ps2, st2)
        @test typeof(logp) == T
        @test isapprox(logp, log(p[1]))
        @test all(isapprox.(st2_post.p, p))

        # test logp of second action with buffered layer
        logp, st2_post = logpdf(p2, 2, ps2, st2)
        @test typeof(logp) == T
        @test isapprox(logp, log(p[2]))
        @test all(isapprox.(st2_post.p, p))

        # test for ignoring observation input
        logpp, st2_postp = logpdf(p2, x, 2, ps2, st2)
        @test typeof(logpp) == T
        @test logpp ≈ logp
        


        # test gradient of logp of first action
        ga1 = gradient(ps -> logpdf(p1, 1, ps, st1)[1], ps1)[1]
        logp1, gl1, st1_post = grad_logpdf(p1, 1, ps1, st1)
        @test typeof(logp1) == T
        @test eltype(gl1) == T
        @test eltype(ga1) == T
        @test typeof(ga1) <: ComponentArray{T}
        @test typeof(gl1) <: ComponentArray{T}
        @test isapprox(gl1, ga1)

        # test for ignoring observation input
        logp1p, gl1p, st1_postp = grad_logpdf(p1, x, 1, ps1, st1)
        @test typeof(logp1p) == T
        @test eltype(gl1p) == T
        @test isapprox(gl1, gl1p)
        @test isapprox(logp1, logp1p)

        ga2 = gradient(ps -> logpdf(p2, 2, ps, st2)[1], ps2)[1]
        logp2, gl2, st2_post = grad_logpdf(p2, 2, ps2, st2)
        @test typeof(logp2) == T
        @test eltype(gl2) == T
        @test eltype(ga2) == T
        @test typeof(ga2) <: ComponentArray{T}
        @test typeof(gl2) <: ComponentArray{T}
        @test isapprox(gl2, ga2)
        @test isapprox(st2_post.ψ, gl2)

        # test for ignoring observation input
        logp2p, gl2p, st2_postp = grad_logpdf(p2, x, 2, ps2, st2)
        @test typeof(logp2p) == T
        @test eltype(gl2p) == T
        @test isapprox(ga2, gl2p)
        @test isapprox(logp2, logp2p)

        logp2, gl2, st1_post = grad_logpdf(p1, 2, ps1, st1)
        logp3, gl3, st1_post = grad_logpdf(p1, 3, ps1, st1)

        # test sample with trace
        a, logp1s, ψ, st1_post = sample_with_trace(p1, ps1, st1)
        @test typeof(a) == Int
        @test typeof(logp1s) == T
        @test typeof(ψ) <: ComponentArray{T}
        if a == 1
            @test logp1s ≈ log(p[1])
            @test ψ ≈ gl1
        elseif a==2
            @test logp1s ≈ log(p[2])
            @test ψ ≈ gl2
        else
            @test logp1s ≈ log(p[3])
            @test ψ ≈ gl3
        end

        # test ignoring observation input
        a, logp1sp, ψp, st1_postp = sample_with_trace(p1, x, ps1, st1)
        @test typeof(a) == Int
        @test typeof(logp1sp) == T
        @test typeof(ψp) <: ComponentArray{T}
        if a == 1
            @test logp1sp ≈ log(p[1])
            @test ψp ≈ gl1
        elseif a==2
            @test logp1sp ≈ log(p[2])
            @test ψp ≈ gl2
        else
            @test logp1sp ≈ log(p[3])
            @test ψp ≈ gl3
        end

        # test sample with trace using buffer
        a, logp2s, ψ, st2_post = sample_with_trace(p2, ps2, st2)
        @test typeof(a) == Int
        @test typeof(logp2s) == T
        @test typeof(ψ) <: ComponentArray{T}
        if a == 1
            @test logp2s ≈ log(p[1])
            @test ψ ≈ gl1
        elseif a==2
            @test logp2s ≈ log(p[2])
            @test ψ ≈ gl2
        else
            @test logp2s ≈ log(p[3])
            @test ψ ≈ gl3
        end

        # test ignoring observation input
        a, logp2sp, ψp, st2_postp = sample_with_trace(p2, x, ps2, st2)
        @test typeof(a) == Int
        @test typeof(logp2sp) == T
        @test typeof(ψp) <: ComponentArray{T}
        if a == 1
            @test logp2sp ≈ log(p[1])
            @test ψp ≈ gl1
        elseif a==2
            @test logp2sp ≈ log(p[2])
            @test ψp ≈ gl2
        else
            @test logp2sp ≈ log(p[3])
            @test ψp ≈ gl3
        end

        # test auto grad p1 including observation input to be ignored
        g1 = gradient(ps -> logpdf(p1(x, ps, st1)[1], 1), ps1)[1]
        g2 = gradient(ps -> logpdf(p1, x, 1, ps, st1)[1], ps1)[1]
        g3 = grad_logpdf(p1, 1, ps1, st1)[2]
        @test eltype(g1) == T
        @test eltype(g2) == T
        @test eltype(g3) == T
        @test isapprox(g1, g2)
        @test isapprox(g1, g3)

        # test auto grad p2
        g2 = gradient(ps -> logpdf(p2, 1, ps, st2)[1], ps2)[1]
        @test eltype(g2) == T
        @test isapprox(g1, g2)
        g3 = grad_logpdf(p2, 1, ps2, st2)[2]
        @test eltype(g3) == T
        @test isapprox(g1, g3)

        A32 = rand(1:num_actions, 9)
        A64 = rand(1:num_actions, 13)

        for a in [A32, A64]
            prob1, st1_post = logpdf(p1, a, ps1, st1)
            prob2, st2_post = logpdf(p2, a, ps2, st2)
            prob3 = [logpdf(p1, a[i], ps1, st1)[1] for i in eachindex(a)]
            @test eltype(prob1) == T
            @test eltype(prob2) == T
            @test length(prob1) == length(prob2)
            @test isapprox(prob1, prob2)
            @test isapprox(prob1, prob3)

            logp1, g1, st1_post = grad_logpdf(p1, a, ps1, st1)
            logp2, g2, st2_post = grad_logpdf(p2, a, ps2, st2)
            # g3 = vec(sum(reduce(hcat, [copy(grad_logpdf(p1, a[i], ps1, st1)[2]) for i in eachindex(a)]), dims=2))
            g3 = zero(ps1)
            for i in eachindex(a)
                g3 .+= grad_logpdf(p1, a[i], ps1, st1)[2]
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
