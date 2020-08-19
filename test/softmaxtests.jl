using DecisionMakingPolicies
using Test

import Zygote: gradient

@testset "Stateless Softmax Tests" begin
    num_actions = 2
    T = Float32
    p = StatelessSoftmax()
    θ = initparams(p, T, num_actions)
    @test eltype(θ) == T
    @test size(θ) == (2,)

    a = rand(p(θ))
    @test typeof(a) <: Int
    logp1 = logpdf(p(θ), 1)
    logp2 = logpdf(p, θ, 1)
    @test typeof(logp1) == T
    @test typeof(logp2) == typeof(logp1)
    @test isapprox(logp1, logp2)
    g = similar(θ)
    gauto1 = gradient(θ->logpdf(p(θ),1),θ)[1]
    gauto2 = gradient(θ->logpdf(p,θ,1),θ)[1]

    @test eltype(gauto1) == eltype(gauto2)
    @test eltype(gauto1) == eltype(θ)
    @test all(isapprox.(gauto1, gauto2))

    logp3 = grad_logpdf!(g, p, θ, 1)
    @test typeof(logp3) == T
    @test logp3 == logp2
    @test all(isapprox.(g, gauto1))
    g2 = Array{T,1}([0.5, -0.5])
    @test all(isapprox.(g, g2))

    A = zeros(Int,1)
    A2 = zeros(Int,2)
    logp4 = sample_with_trace!(g, A, p, θ)
    @test A[1] > 0
    if A[1] == 1
        @test all(isapprox.(g, g2))
    else
        @test all(isapprox.(g, -g2))
    end
    sample_with_trace!(g, view(A2,1:1), p, θ)
    @test A2[1] > 0
    @test A2[2] == 0
    a1 = A2[1]
    sample_with_trace!(g, view(A2,2:2), p, θ)
    @test A2[1] == a1
    @test A2[2] > 0
end


@testset "Linear Softmax Tests" begin
    num_actions = 2
    num_features = 3
    T = Float32
    p = LinearSoftmax()
    θ = initparams(p, T, num_features, num_actions)
    @test eltype(θ) == T
    @test size(θ) == (num_features, num_actions)
    s = Array{T, 1}([0.0, 1.0, 2.0])
    a = rand(p(θ, s))
    @test typeof(a) <: Int
    logp1 = logpdf(p(θ, s), 1)
    logp2 = logpdf(p, θ, s, 1)
    @test typeof(logp1) == T
    @test typeof(logp2) == typeof(logp1)
    @test isapprox(logp1, logp2)
    g = similar(θ)
    gauto1 = gradient(θ->logpdf(p(θ,s),1),θ)[1]
    gauto2 = gradient(θ->logpdf(p,θ,s,1),θ)[1]

    @test eltype(gauto1) == eltype(gauto2)
    @test eltype(gauto1) == eltype(θ)
    @test all(isapprox.(gauto1, gauto2))

    logp3 = grad_logpdf!(g, p, θ, s, 1)
    @test typeof(logp3) == T
    @test logp3 == logp2
    @test all(isapprox.(g, gauto1))
    g2 = Array{T,2}([0.0 0.0; 0.5 -0.5; 1.0 -1.0])
    @test all(isapprox.(g, g2))

    A = zeros(Int,1)
    A2 = zeros(Int,2)
    logp4 = sample_with_trace!(g, A, p, θ, s)
    @test A[1] > 0
    if A[1] == 1
        @test all(isapprox.(g, g2))
    else
        @test all(isapprox.(g, -g2))
    end
    sample_with_trace!(g, view(A2,1:1), p, θ, s)
    @test A2[1] > 0
    @test A2[2] == 0
    a1 = A2[1]
    sample_with_trace!(g, view(A2,2:2), p, θ, s)
    @test A2[1] == a1
    @test A2[2] > 0
end
