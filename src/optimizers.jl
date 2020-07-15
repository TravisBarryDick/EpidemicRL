# ----- Gradient Ascent ----- #

mutable struct GradientAscent{StepSize}
    θ::Vector{Float64}
    stepsize::StepSize
end

copy(gd::GradientAscent) = GradientAscent(copy(gd.θ), gd.stepsize)

get_params(gd::GradientAscent) = gd.θ

function update(gd::GradientAscent, ∇)
    gd.θ += gd.stepsize .* ∇
    gd.θ
end

# ----- Adam Optimizer ----- #

mutable struct Adam
    # Parameter vector
    θ::Vector{Float64}
    # Adam parameters
    α::Float64
    β1::Float64
    β2::Float64
    ϵ::Float64
    # Adam state
    m::Vector{Float64}
    v::Vector{Float64}
    β1t::Float64
    β2t::Float64
end

function Adam(θ; stepsize = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8)
    m = zero(θ)
    v = zero(θ)
    Adam(θ, stepsize, beta1, beta2, epsilon, m, v, beta1, beta2)
end

copy(ad::Adam) =
    Adam(ad.θ, ad.α, ad.β1, ad.β2, ad.ϵ, copy(ad.m), copy(ad.v), ad.β1t, ad.β2t)

get_params(ad::Adam) = ad.θ

function update(ad::Adam, ∇)
    ad.m = ad.β1 * ad.m + (1 - ad.β1) * ∇
    ad.v = ad.β2 * ad.v + (1 - ad.β2) * (∇ .^ 2)
    hat_m = ad.m / (1 - ad.β1t)
    hat_v = ad.v / (1 - ad.β2t)
    ad.θ += ad.α * hat_m ./ (sqrt.(hat_v) .+ ad.ϵ)
    ad.β1t *= ad.β1
    ad.β2t *= ad.β2
    ad.θ
end
