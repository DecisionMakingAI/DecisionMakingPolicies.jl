using SafeTestsets



@safetestset "SoftmaxTests" begin include("softmaxtests.jl") end
@safetestset "NeuralNetTests" begin include("luxtests.jl") end
@safetestset "BasisTests" begin include("basistest.jl") end
# @safetestset "NormalTests" begin include("normaltests.jl") end
