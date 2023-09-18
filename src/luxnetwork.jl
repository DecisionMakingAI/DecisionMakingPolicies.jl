
struct DeepPolicy{L1, L2} <: LuxCore.AbstractExplicitContainerLayer{(:basis, :linear)}
    basis::L1
    linear::L2
end

function (l::DeepPolicy)(x, ps, st::NamedTuple)
    y, st_b = l.basis(x, ps.basis, st.basis)
    z, st_l = l.linear(y, ps.linear, st.linear)
    return z, (basis = st_b, linear = st_l)
end

function logpdf(layer::DeepPolicy, x, a, params, states)
    y, st_b = layer.basis(x, params.basis, states.basis)
    z, st_l = logpdf(layer.linear, y, a, params.linear, states.linear)
    return z, (basis = st_b, linear = st_l)
end

function sample_return_logp(action, model, x, params, states)
    d, st = model(x, params, states)
    ignore_derivatives() do
        a = rand(d)
        push!(action, a)
    end
    logp = logpdf(d, action[1])
    return logp, st
end


function sample_with_trace(layer::DeepPolicy, x, params, states)
    action = []
    (logp, st), back = Zygote.pullback(ps -> sample_return_logp(action, layer, x, ps, states), params)
    grad =  back((one(logp), nothing))[1]
    a = action[1]
    return a, logp, grad, st
end

function sum_logpdf(logps, states)
    return sum(logps), states
end
function grad_logpdf(layer::DeepPolicy, x, a, params, states)
    (logp, st), back = Zygote.pullback(ps ->sum_logpdf(logpdf(layer, x, a, ps, states)...), params)
    grad =  back((one(logp), nothing))[1]
    return logp, grad, st
end

