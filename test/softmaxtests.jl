using DecisionMakingPolicies
using Test
import DecisionMakingPolicies: grad_logpdf!, logpdf!, sample_with_trace!
import DecisionMakingPolicies: logpdf_softmax
import Zygote: gradient, pullback
using ChainRulesTestUtils


@testset "Stateless Softmax Tests" begin
    num_actions = 2
    T = Float32
    p = StatelessSoftmax(T, num_actions)
    @test length(p.θ) == num_actions
    @test eltype(p.θ) == T
    @test size(p.θ) == (num_actions,)
    
    a = rand(p())
    @test typeof(a) <: Int
    logp1 = logpdf(p(), 1)
    logp2 = logpdf(p, 1)
    @test typeof(logp1) == T
    @test typeof(logp2) == typeof(logp1)
    @test isapprox(logp1, logp2)
    g = (θ=zero(p.θ),)
    gauto1 = gradient(p->logpdf(p(),1),p)[1].θ
    gauto2 = gradient(p->logpdf(p,1),p)[1].θ
    
    @test eltype(gauto1) == eltype(gauto2)
    @test eltype(gauto1) == eltype(p.θ)
    @test all(isapprox.(gauto1, gauto2))

    logp3, _ = grad_logpdf!(g, p, 1)
    @test typeof(logp3) == T
    @test logp3 == logp2
    @test all(isapprox.(g.θ, gauto1))
    g2 = Array{T,1}([0.5, -0.5])
    @test all(isapprox.(g.θ, g2))

    A = zeros(Int,1)
    A2 = zeros(Int,2)
    _, logp4, _ = sample_with_trace!(g, A, p)
    @test A[1] > 0
    if A[1] == 1
        @test all(isapprox.(g.θ, g2))
    else
        @test all(isapprox.(g.θ, -g2))
    end
    sample_with_trace!(g, view(A2,1:1), p)
    @test A2[1] > 0
    @test A2[2] == 0
    a1 = A2[1]
    sample_with_trace!(g, view(A2,2:2), p)
    @test A2[1] == a1
    @test A2[2] > 0
end

@testset "Stateless Softmax Buffer Tests" begin
    num_actions = 2
    T = Float32
    p = StatelessSoftmax(T, num_actions)
    buff = SoftmaxBuffer(p)
    
    @test eltype(buff.action) == Int
    @test eltype(buff.p) == T
    @test eltype(buff.ψ[1]) == T
    @test size(buff.ψ[1]) == size(p.θ)
    @test size(buff.p) == size(p.θ)

    a = rand(p(buff))
    @test typeof(a) <: Int
    logp1 = logpdf(p(), 1)
    logp2 = logpdf(p(buff), 1)
    logp3 = logpdf!(buff, p, 1)
    @test typeof(logp1) == T
    @test typeof(logp2) == typeof(logp1)
    @test typeof(logp3) == typeof(logp1)
    @test isapprox(logp1, logp2)
    @test isapprox(logp1, logp3)

    gauto1 = gradient(p->logpdf(p,1),p)[1].θ
    logp3, _ = grad_logpdf!(buff, p, 1)
    @test typeof(logp3) == T
    @test logp3 == logp1
    @test all(isapprox.(buff.ψ[1], gauto1))
    g2 = Array{T,1}([0.5, -0.5])
    @test all(isapprox.(buff.ψ[1], g2))

    
    _, logp4, _ = sample_with_trace!(buff, p)
    @test buff.action[1] > 0
    if buff.action[1] == 1
        @test all(isapprox.(buff.ψ[1], g2))
    else
        @test all(isapprox.(buff.ψ[1], -g2))
    end
end



@testset "Linear Softmax Tests" begin
    num_actions = 2
    num_features = 3
    T = Float32
    p = LinearSoftmax(T, num_features, num_actions)

    @test eltype(p.θ) == T
    @test size(p.θ) == (num_features, num_actions)
    s = Array{T, 1}([0.0, 1.0, 2.0])
    a = rand(p(s))
    @test typeof(a) <: Int
    logp1 = logpdf(p(s), 1)
    logp2 = logpdf(p, s, 1)
    @test typeof(logp1) == T
    @test typeof(logp2) == typeof(logp1)
    @test isapprox(logp1, logp2)
    g = (θ=zero(p.θ),)
    
    gauto1 = gradient(p->logpdf(p(s),1),p)[1].θ
    gauto2 = gradient(p->logpdf(p,s,1),p)[1].θ

    @test eltype(gauto1) == eltype(gauto2)
    @test eltype(gauto1) == eltype(p.θ)
    @test all(isapprox.(gauto1, gauto2))

    logp3, _ = grad_logpdf!(g, p, s, 1)
    @test typeof(logp3) == T
    @test logp3 == logp2
    @test all(isapprox.(g.θ, gauto1))
    g2 = Array{T,2}([0.0 0.0; 0.5 -0.5; 1.0 -1.0])
    @test all(isapprox.(g.θ, g2))
    logp4, gpi =  grad_logpdf(p, s, 1)
    @test all(isapprox.(gpi[1], g2))
    @test logp4 == logp2

    A = zeros(Int,1)
    A2 = zeros(Int,2)
    _, logp4, _ = sample_with_trace!(g, A, p, s)
    @test A[1] > 0
    if A[1] == 1
        @test all(isapprox.(g.θ, g2))
    else
        @test all(isapprox.(g.θ, -g2))
    end
    sample_with_trace!(g, view(A2,1:1), p, s)
    @test A2[1] > 0
    @test A2[2] == 0
    a1 = A2[1]
    sample_with_trace!(g, view(A2,2:2), p, s)
    @test A2[1] == a1
    @test A2[2] > 0

    a5, logp5, g5 = sample_with_trace(p, s)
    if a5 == 1
        @test all(isapprox.(g5[1], g2))
    else
        @test all(isapprox.(g5[1], -g2))
    end

    
end

@testset "Linear Softmax Batch Tests" begin
    num_actions = 2
    num_features = 3
    T = Float32
    p = LinearSoftmax(T, num_features, num_actions)
    vec(p.θ) .= collect(1:length(p.θ)) .* π ./ length(p.θ)
    @test eltype(p.θ) == T
    @test size(p.θ) == (num_features, num_actions)
    num_samples = 1
    for num_samples in [1, 11]
        X = rand(T, num_features, num_samples)
        A = rand(1:num_actions, num_samples) 
        blogps = zeros(T, num_samples)
        grad = zero(p.θ)
        for i in 1:num_samples
            logp, g = grad_logpdf(p, X[:, i], A[i])
            blogps[i] = logp
            @. grad += g[1]
        end
        logps = logpdf(p, X, A)
        @test logps isa Vector{T}
        @test logps == blogps

        logps = logpdf(p, X, reshape(A, (1,num_samples)))
        @test logps isa Matrix{T}
        @test size(logps) == (1,num_samples)
        @test logps[1,:] == blogps
        
        zlogps, gbf = pullback(p->sum(logpdf(p,X,A)),p)
        @test zlogps isa T
        @test zlogps == sum(blogps)
        g = gbf(1.0)[1]
        @test g.θ == grad
        g2 = gbf(1.0)[1]
        @test g2.θ == g.θ

        zlogps, gbf = pullback(p->sum(logpdf(p,X,reshape(A,(1,num_samples)))),p)
        @test zlogps isa T
        @test zlogps == sum(blogps)
        g = gbf(1)[1]
        @test g.θ == grad
        g2 = gbf(1)[1]
        @test g.θ == g2.θ

        blogps = zeros(T, (num_actions, num_samples))
        grad = zero(p.θ)
        ag = rand(T, num_actions, num_samples)
        for i in 1:num_samples
            for a in 1:num_actions
                logp, g = grad_logpdf(p, X[:, i], a)
                blogps[a, i] = logp
                @. grad += g[1] * ag[a,i]
            end
        end
        
        logps = logpdf(p, X)
        @test logps isa Matrix{T}
        @test logps == blogps

        zlogps, gbf = pullback(p->sum(logpdf(p,X) .* ag),p)
        @test zlogps isa T
        @test zlogps == sum(blogps .* ag)
        g = gbf(1)[1]
        @test all(isapprox.(g.θ,grad))
        g2 = gbf(1)[1]
        @test g.θ == g2.θ
    end
end

@testset "Linear Softmax Buffer Tests" begin
    num_actions = 2
    num_features = 3
    T = Float32
    p = LinearSoftmax(T, num_features, num_actions)
    buff = SoftmaxBuffer(p)
    @test eltype(buff.action) == Int
    @test eltype(buff.p) == T
    @test eltype(buff.ψ[1]) == T
    @test size(buff.ψ[1]) == size(p.θ)
    @test size(buff.p) == (num_actions,)


    s = Array{T, 1}([0.0, 1.0, 2.0])
    a = rand(p(s))
    @test typeof(a) <: Int
    logp1 = logpdf(p(s), 1)
    logp2 = logpdf(p(buff, s), 1)
    logp3 = logpdf!(buff, p, s, 1)
    @test typeof(logp1) == T
    @test typeof(logp2) == typeof(logp1)
    @test typeof(logp3) == typeof(logp1)
    @test isapprox(logp1, logp2)
    @test isapprox(logp1, logp3)
    
    gauto1 = gradient(p->logpdf(p(s),1),p)[1].θ
    logp3, _ = grad_logpdf!(buff, p, s, 1)
    @test typeof(logp3) == T
    @test logp3 == logp2
    @test all(isapprox.(buff.ψ[1], gauto1))
    g2 = Array{T,2}([0.0 0.0; 0.5 -0.5; 1.0 -1.0])
    @test all(isapprox.(buff.ψ[1], g2))

    _, logp4, _ = sample_with_trace!(buff, p, s)
    @test buff.action[1] > 0
    if buff.action[1] == 1
        @test all(isapprox.(buff.ψ[1], g2))
    else
        @test all(isapprox.(buff.ψ[1], -g2))
    end
end

# @testset "Softmax Rules Tests" begin
#     num_actions = 2
#     num_features = 3
#     T = Float32
#     p = LinearSoftmax(T, num_features, num_actions)
#     x = randn(T, num_features)
#     X = randn(T, (num_features, 10))
#     A = rand(1:10)

#     # test_rrule(logpdf_softmax, p.θ, x, 1)
#     # test_rrule(logpdf_softmax, p.θ, x, 2)
#     # test_rrule(logpdf_softmax, p.θ, x)
#     # test_rrule(logpdf_softmax, p.θ, X, A)
#     # test_rrule(logpdf_softmax, p.θ, X)

    
# end