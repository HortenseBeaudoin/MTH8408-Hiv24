# using Pkg
# Pkg.add("LinearAlgebra")
# Pkg.add("Krylov")
# Pkg.add("NLPModels")
# Pkg.add("Printf")
# Pkg.add("Logging")
# Pkg.add("SolverCore")
# Pkg.add("Test")
# Pkg.add("ADNLPModels")
# Pkg.add("NLSProblems")
# Pkg.add("SolverBenchmark")
# Pkg.add("Plots")
# Pkg.add("JSOSolvers")
# Pkg.add("CaNNOLeS")
# Pkg.add("NLPModelsJuMP")

using LinearAlgebra, Krylov, NLPModels, Printf, Logging, SolverCore, Test, ADNLPModels, NLSProblems, SolverBenchmark, Plots, JSOSolvers
using CaNNOLeS, NLPModelsJuMP

#Test problem:
FH(x) = [x[2]+x[1].^2-11, x[1]+x[2].^2-7]
x0H = [10., 20.]
###########################
neq = 2
#Utilise FH et x0H pour créer un ADNLSModel
himmelblau_nls = ADNLSModel!(FH, x0H, neq)
###########################

function gauss_newton(nlp      :: AbstractNLSModel, 
    x        :: AbstractVector, 
    ϵ        :: AbstractFloat;
    η₁       :: AbstractFloat = 1e-3, 
    η₂       :: AbstractFloat = 0.66, 
    σ₁       :: AbstractFloat = 0.25, 
    σ₂       :: AbstractFloat = 2.0,
    max_eval :: Int = 1_000, 
    max_time :: AbstractFloat = 60.,
    max_iter :: Int = typemax(Int64)
    )
######################################################
Fx = residual(nlp, x)
Jx = jac_residual(nlp, x)
######################################################
normFx = norm(Fx)

Δ = 1.

iter = 0    

el_time = 0.0
tired   = neval_residual(nlp) > max_eval || el_time > max_time
status  = :unknown

start_time = time()
too_small  = false
normdual   = norm(Jx' * Fx)
optimal    = min(normFx, normdual) ≤ ϵ

@info log_header([:iter, :nf, :primal, :status, :nd, :Δ],
[Int, Int, Float64, String, Float64, Float64],
hdr_override=Dict(:nf => "#F", :primal => "‖F(x)‖", :nd => "‖d‖"))

while !(optimal || tired || too_small)

#################################
#Compute a direction satisfying the trust-region constraint
(d, stats)  = lsmr(-Jx, Fx; radius=Δ)
#################################

too_small = norm(d) < 1e-15
if too_small #the direction is too small
status = :too_small
else
xp      = x + d
###########################
Fxp     = residual(nlp, xp)
###########################
normFxp = norm(Fxp)

Pred = 0.5 * (normFx^2 - norm(Jx * d + Fx)^2)
Ared = 0.5 * (normFx^2 - normFxp^2)

if Ared/Pred < η₁
Δ = max(1e-8, Δ * σ₁)
status = :reduce_Δ
else #success
x  = xp
###########################
Jx = jac_residual(nlp, x)
###########################
Fx = Fxp
normFx = normFxp
status = :success
if Ared/Pred > η₂ && norm(d) >= 0.99 * Δ
  Δ *= σ₂
end
end
end

@info log_row(Any[iter, neval_residual(nlp), normFx, status, norm(d), Δ])

el_time      = time() - start_time
iter   += 1

many_evals   = neval_residual(nlp) > max_eval
iter_limit   = iter > max_iter
tired        = many_evals || el_time > max_time || iter_limit
normdual     = norm(Jx' * Fx)
optimal      = min(normFx, normdual) ≤ ϵ
end

status = if optimal 
:first_order
elseif tired
if neval_residual(nlp) > max_eval
:max_eval
elseif el_time > max_time
:max_time
elseif iter > max_iter
:max_iter
else
:unknown_tired
end
elseif too_small
:stalled
else
:unknown
end

return GenericExecutionStats(nlp; status, solution = x,
               objective = normFx^2 / 2,
               dual_feas = normdual,
               iter = iter, 
               elapsed_time = el_time)
end


# stats = gauss_newton(himmelblau_nls, himmelblau_nls.meta.x0, 1e-6)
# @test stats.status == :first_order


function dsol(Fx, Jx, λ, τ)
    (d, stats) = lsqr(-Jx, Fx; λ)
    return d
end


function multi_sol(nlp, x, Fx, Jx, λ, τ; nl = 3)
    n = (nl-1)/2
    lam = []
    for i in n:1
        push!(lam, λ/(i*10))
    end
    push!(lam, λ)
    for i in 1:n
        push!(lam, λ*(i*10))
    end

    d = dsol(Fx, Jx, λ, τ)

    for l in lam
        srch = dsol(Fx, Jx, l, τ)
        if residual(nlp, x+srch) < residual(nlp, x+d)
            d = srch
        end
    end
    return d
end



function lm_param(nlp      :: AbstractNLSModel, 
    x        :: AbstractVector, 
    ϵ        :: AbstractFloat;
    η₁       :: AbstractFloat = 1e-3, 
    η₂       :: AbstractFloat = 0.66, 
    σ₁       :: AbstractFloat = 10.0, 
    σ₂       :: AbstractFloat = 0.5,
    max_eval :: Int = 10_000, 
    max_time :: AbstractFloat = 60.,
    max_iter :: Int = typemax(Int64)
    )
######################################################
Fx = residual(nlp, x)
Jx = jac_residual(nlp, x)
######################################################
normFx   = norm(Fx)
normdual = norm(Jx' * Fx)

iter = 0    
λ = 0.0
λ₀ = 1e-6
η = 0.5
τ = η * normdual

el_time = 0.0
tired   = neval_residual(nlp) > max_eval || el_time > max_time
status  = :unknown

start_time = time()
too_small  = false
optimal    = min(normFx, normdual) ≤ ϵ

@info log_header([:iter, :nf, :primal, :status, :nd, :λ],
[Int, Int, Float64, String, Float64, Float64],
hdr_override=Dict(:nf => "#F", :primal => "‖F(x)‖", :nd => "‖d‖"))

while !(optimal || tired || too_small)

###########################
# (d, stats)  = lsqr(Jx, -Fx, λ = λ, atol = τ)
d = multi_sol(nlp, x, Fx, Jx, λ, τ)
###########################

too_small = norm(d) < 1e-16
if too_small #the direction is too small
status = :too_small
else
xp      = x + d
###########################
Fxp     = residual(nlp, xp)
###########################
normFxp = norm(Fxp)

Pred = 0.5 * (normFx^2 - norm(Jx * d + Fx)^2 - λ*norm(d)^2)
Ared = 0.5 * (normFx^2 - normFxp^2)

if Ared/Pred < η₁
  λ = max(λ₀, σ₁ * λ)
  status = :increase_λ
else #success
  x  = xp
  ###########################
  Jx = jac_residual(nlp, x)
  ###########################
  Fx = Fxp
  normFx = normFxp
  status = :success
  if Ared/Pred > η₂
      λ = max(λ * σ₂, λ₀)
  end
end
end

@info log_row(Any[iter, neval_residual(nlp), normFx, status, norm(d), λ])

el_time      = time() - start_time
iter        += 1
many_evals   = neval_residual(nlp) > max_eval
iter_limit   = iter > max_iter
tired        = many_evals || el_time > max_time || iter_limit
normdual     = norm(Jx' * Fx)
optimal      = min(normFx, normdual) ≤ ϵ

η = λ == 0.0 ? min(0.5, 1/iter, normdual) : min(0.5, 1/iter)
τ = η * normdual
end

status = if optimal 
:first_order
elseif tired
if neval_residual(nlp) > max_eval
:max_eval
elseif el_time > max_time
:max_time
elseif iter > max_iter
:max_iter
else
:unknown_tired
end
elseif too_small
:stalled
else
:unknown
end

return GenericExecutionStats(nlp; status, solution = x,
                   objective = normFx^2 / 2,
                   dual_feas = normdual,
                   iter = iter, 
                   elapsed_time = el_time)
end


# stats = lm_param(himmelblau_nls, himmelblau_nls.meta.x0, 1e-6)
# @test stats.status == :first_order


# Benchmarking
using NLSProblems
n = 20
ϵ = 1e-6

solvers = Dict(
    :gauss_newton => model -> gauss_newton(model, model.meta.x0, ϵ),
    :lm_param => model -> lm_param(model, model.meta.x0, ϵ),
)

problems = (eval(problem)() for problem ∈ filter(x -> x != :NLSProblems, names(NLSProblems)))

stats = bmark_solvers(
  solvers, problems,
  skipif=prob -> (!unconstrained(prob) || get_nvar(prob) > 100 || get_nvar(prob) < 5),
)

cols = [:id, :name, :nvar, :objective, :dual_feas, :neval_residual, :neval_jac_residual, :neval_hess, :iter, :elapsed_time, :status]
header = Dict(
  :nvar => "n",
  :objective => "f(x)",
  :dual_feas => "‖∇f(x)‖",
  :neval_residual => "# f",
  :neval_jac_residual => "# ∇f",
  :neval_hess => "# ∇²f",
  :elapsed_time => "t",
)

for solver ∈ keys(solvers)
  pretty_stats(stats[solver][!, cols], hdr_override=header)
end

first_order(df) = df.status .== :first_order
unbounded(df) = df.status .== :unbounded
solved(df) = first_order(df) .| unbounded(df)

costnames = ["time", "residual", "residual jacobien"]
costs = [
  df -> .!solved(df) .* Inf .+ df.elapsed_time,
  df -> .!solved(df) .* Inf .+ df.neval_residual,
  df -> .!solved(df) .* Inf .+ df.neval_jac_residual,
]

using Plots
gr()

profile_solvers(stats, costs, costnames)