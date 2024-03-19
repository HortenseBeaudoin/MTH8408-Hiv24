using Pkg
# Pkg.add("Krylov")
# Pkg.add("LinearAlgebra")
# Pkg.add("Logging")
# Pkg.add("NLPModels")
# Pkg.add("NLPModelsIpopt")
# Pkg.add("Printf")
# Pkg.add("SolverCore")
# Pkg.add("Test")
# Pkg.add("PDENLPModels")
# Pkg.add("Gridap")
# Pkg.add("ADNLPModels")
# Pkg.add("NLPModelsIpopt")
# Pkg.add("Plots")
# Pkg.add("OptimizationProblems")
# Pkg.add("JSOSolvers")
# Pkg.add("SolverBenchmark")

using Krylov, LinearAlgebra, Logging, NLPModels, NLPModelsIpopt, Printf, SolverCore, Test
using PDENLPModels, Gridap, ADNLPModels

function quad_penalty_adnlp(nlp :: ADNLPModel, ρ :: Real)
    f = x -> obj(nlp, x) + (ρ/2)*(norm(cons(nlp, x)))^2
    x0 = nlp.meta.x0
    nlp_quad = ADNLPModel(f, x0)
   return nlp_quad
end

#Faire des tests pour vérifier que ça fonctionne.
fH(x) = (x[2]+x[1].^2-11)^2 + (x[1]+x[2].^2-7)^2
x0H = [10., 20.]
cH(x) = [x[1]-1]
himmelblau = ADNLPModel(fH, x0H, cH, [0.], [0.])

himmelblau_quad = quad_penalty_adnlp(himmelblau, 1)
@test himmelblau_quad.meta.ncon == 0
@test obj(himmelblau_quad, zeros(2)) == 170.5

#Ajouter au moins un autre test similaire avec des contraintes.
n = 10
nlp_test1 = ADNLPModel(x->dot(x, x), zeros(n), x->[sum(x) - 1], zeros(1), zeros(1))
nlp_test1_quad = quad_penalty_adnlp(nlp_test1, 1)
@test nlp_test1_quad.meta.ncon == 0
@test obj(nlp_test1_quad, zeros(n)) == 0.5

nlp_test2 = ADNLPModel(x->(x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0], x->[sum(x)-1], [0.0], [0.0])
nlp_test2_quad = quad_penalty_adnlp(nlp_test2, 1)
@test nlp_test2_quad.meta.ncon == 0
@test obj(nlp_test2_quad, zeros(2)) == 1.5

# Ajouter un test au cas ou `nlp.meta.lcon` ou `nlp.meta.ucon` ont des composantes differentes de 0.
fH2(x) = (x[2]+x[1].^2-11)^2 + (x[1]+x[2].^2-7)^2
x0H2 = [10., 20.]
cH2(x) = [x[1]-1]
himmelblau2 = ADNLPModel(fH2, x0H2, cH2, [0.1], [0.1])

himmelblau2_quad = quad_penalty_adnlp(himmelblau2, 1)
@test himmelblau2_quad.meta.ncon == 0
@test himmelblau2_quad.meta.lcon != 0
@test himmelblau2_quad.meta.ucon != 0
@test obj(himmelblau2_quad, zeros(2)) == 170.5

# EXERCICE KKT

function KKT_eq_constraint(nlp :: AbstractNLPModel, x, λ)
  if isapprox(grad(nlp, x), dot(jac(nlp, x),λ) ; atol=1e-6) || isapprox(cons(nlp, x), 0 ; atol=1e-6)
    return 1
  else
    return 0
  end
end


function quad_penalty(nlp      :: AbstractNLPModel,
    x        :: AbstractVector; 
    ϵ        :: AbstractFloat = 1e-3,
    η        :: AbstractFloat = 1e6, 
    σ        :: AbstractFloat = 2.0,
    max_eval :: Int = 1_000, 
    max_time :: AbstractFloat = 60.,
    max_iter :: Int = typemax(Int64)
    )
##### Initialiser cx et gx au point x;
cx = cons(nlp, x)
gx = grad(nlp, x)
######################################################
normcx = normcx_old = norm(cx)

ρ = 1.

iter = 0    

el_time = 0.0
tired   = neval_cons(nlp) > max_eval || el_time > max_time
status  = :unknown

start_time = time()
too_small  = false
normdual   = norm(gx) #exceptionnellement on ne va pas vérifier toute l'optimalité au début.
optimal    = max(normcx, normdual) ≤ ϵ

nlp_quad   = quad_penalty_adnlp(nlp, ρ)

@info log_header([:iter, :nf, :primal, :status, :nd, :Δ],
[Int, Int, Float64, String, Float64, Float64],
hdr_override=Dict(:nf => "#F", :primal => "‖F(x)‖", :nd => "‖d‖"))

while !(optimal || tired || too_small)

#Appeler Ipopt pour résoudre le problème pénalisé en partant du point x0 = x.
#utiliser l'option print_level = 0 pour enlever les affichages d'ipopt.
stats = ipopt(nlp_quad, print_level = 0)
################################################

if stats.status == :first_order
###### Mettre à jour cx avec la solution renvoyé par Ipopt
x = stats.solution
cx = cons(nlp, x)
##########################################################
normcx_old = normcx
normcx = norm(cx)
end

if normcx_old > 0.95 * normcx
ρ *= σ
end

@info log_row(Any[iter, neval_cons(nlp), normcx, stats.status])

nlp_quad   = quad_penalty_adnlp(nlp, ρ)

el_time      = time() - start_time
iter   += 1
many_evals   = neval_cons(nlp) > max_eval
iter_limit   = iter > max_iter
tired        = many_evals || el_time > max_time || iter_limit || ρ ≥ η
##### Utiliser la réalisabilité dual renvoyé par Ipopt pour `normdual`
normdual     = stats.dual_feas
###################################################################
optimal      = max(normcx, normdual) ≤ ϵ
end

status = if optimal 
:first_order
elseif tired
if neval_cons(nlp) > max_eval
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

return GenericExecutionStats(nlp, status = status, solution = x,
               objective = obj(nlp, x),
               primal_feas = normcx,
               dual_feas = normdual,
               iter = iter, 
               multipliers = stats.multipliers,
               elapsed_time = el_time,
               solver_specific = Dict(:penalty => ρ))
end

#Faire des tests pour vérifier que ça fonctionne.
stats = quad_penalty(himmelblau, x0H)
@test stats.status == :first_order
@test stats.solution ≈ [1.0008083416169895, 2.709969135758311] atol=1e-2
@test norm(cons(himmelblau, stats.solution)) ≈ 0. atol=1e-3

# Test sur les conditions de KKT À COMPLÉTER
print(stats.solution)
print(stats.iter)
print(stats.multipliers)

KKT_eq_constraint(himmelblau, stats.solution, stats.multipliers)


# #Tests supplémentaires.
# tol = 1e-3
# @testset "Simple problem" begin
#     n = 10
#     nlp = ADNLPModel(x->dot(x, x), zeros(n),
#                      x->[sum(x) - 1], zeros(1), zeros(1))

#     stats = with_logger(NullLogger()) do
#       quad_penalty(nlp, nlp.meta.x0, ϵ=1e-6)
#     end
#     dual, primal, status = stats.dual_feas, stats.primal_feas, stats.status
#     @test norm(n * stats.solution - ones(n)) < tol
#     @test dual < tol
#     @test primal < tol
#     @test status == :first_order
# end

# @testset "Rosenbrock with ∑x = 1" begin
#     nlp = ADNLPModel(x->(x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2, 
#                      [-1.2; 1.0],
#                      x->[sum(x)-1], [0.0], [0.0])

#     stats = with_logger(NullLogger()) do
#       quad_penalty(nlp, nlp.meta.x0)
#     end
#     dual, primal, status = stats.dual_feas, stats.primal_feas, stats.status
#     @test dual < tol#1e-6
#     @test primal < tol
#     @test status == :first_order
# end

# @testset "HS6" begin
#     nlp = ADNLPModel(x->(1 - x[1])^2, [-1.2; 1.0],
#                      x->[10 * (x[2] - x[1]^2)], [0.0], [0.0])

#     stats = with_logger(NullLogger()) do
#       quad_penalty(nlp, nlp.meta.x0)
#     end
#     dual, primal, status = stats.dual_feas, stats.primal_feas, stats.status
#     @test dual < tol
#     @test primal < tol
#     @test status == :first_order
# end

# @testset "HS7" begin
#     nlp = ADNLPModel(x->log(1 + x[1]^2) - x[2], 
#                      [2.0; 2.0],
#                      x->[(1 + x[1]^2)^2 + x[2]^2 - 4], [0.0], [0.0])

#     stats = with_logger(NullLogger()) do
#       quad_penalty(nlp, nlp.meta.x0)
#     end
#     dual, primal, status = stats.dual_feas, stats.primal_feas, stats.status
#     @test dual < tol
#     @test primal < tol
#     @test status == :first_order
# end

# BENCHMARKING

# using ADNLPModels
# using OptimizationProblems, OptimizationProblems.ADNLPProblems
# using LinearAlgebra
# using SolverCore, SolverBenchmark
# using ADNLPModels, NLPModels
# using OptimizationProblems, OptimizationProblems.ADNLPProblems
# using JSOSolvers

# n = 20
# solvers = solvers = Dict(
#   :qpenal => model -> quad_penalty(model, model.meta.x0),
# )

# ad_problems = (eval(Meta.parse(problem))(;n) for problem ∈ OptimizationProblems.meta[!, :name])

# stats = bmark_solvers(
#   solvers, ad_problems,
#   skipif=prob -> (unconstrained(prob) || get_nvar(prob) > 100 || get_nvar(prob) < 5),
# )

# cols = [:id, :name, :nvar, :neval_cons, :neval_obj, :neval_grad, :neval_hess, :neval_ :iter, :elapsed_time, :status]
# header = Dict(
#   :nvar => "n",
#   :neval_cons => "# cons",
#   :neval_obj => "# f",
#   :neval_grad => "# ∇f",
#   :neval_hess => "# ∇²f",
#   :iter => "iter",
#   :elapsed_time => "t",
# )

# for solver ∈ keys(solvers)
#   pretty_stats(stats[solver][!, cols], hdr_override=header)
# end


# EXERCICE 2

function cv_model(n :: Int)

    domain = (0,1) # set the domain
    partition = n
    model = CartesianDiscreteModel(domain,partition) # set discretization
      
    labels = get_face_labeling(model)
    add_tag_from_tags!(labels,"diri1",[2])
    add_tag_from_tags!(labels,"diri0",[1]) # boundary conditions
  
    order=1
    valuetype=Float64
    reffe = ReferenceFE(lagrangian, valuetype, order)
    V0 = TestFESpace(model, reffe; conformity=:H1, dirichlet_tags=["diri0","diri1"])
    U = TrialFESpace(V0,[0., exp(1)-exp(-2)])
  
    trian = Triangulation(model)
    degree = 2
    dΩ = Measure(trian,degree) # integration machinery
  
    # Our objective function
    w(x) = exp(x[1])
    function f(y)
      ∫((∇(y)⊙∇(y) + 2 * y * y) * w) * dΩ
    end
  
    xin = zeros(Gridap.FESpaces.num_free_dofs(U))
    nlp = GridapPDENLPModel(xin, f, trian, U, V0)
    return nlp
  end

n = 16
stats = ipopt(cv_model(n))
solu = vcat(0, stats.solution, ℯ-ℯ^(-2))
print(solu)


sol = []
for n in [8, 16, 32, 64, 128]
    stats = ipopt(cv_model(n), print_level = 0)
    solu = vcat(0, stats.solution, ℯ-ℯ^(-2))
    push!(sol, solu)
end
sol[1]

x = []
for n in [8, 16, 32, 64, 128]
    r = collect(LinRange(0, 1, n+1))
    push!(x, r)
end

using Plots
plot(x, sol, label=["n=8" "n=16" "n=32" "n=64" "n=128"])

t = x[5]
val = ℯ*ones(length(t))
y = val.^t - val.^(-2*t)
plot!(t, y, label=["x(t)"])

print("Valeur optimale:")
print(ℯ^3-2*ℯ^(-3)+1)
print()
print("\nValeur à n=8")
