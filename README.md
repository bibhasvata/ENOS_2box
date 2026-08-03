# ENOS_2box (v0)
2box inversion script for simultaneous optimisation of emissions-lifetime-isotopes-KIE
COMPLETE TWO-BOX CH4 INVERSION WITH TIME-VARYING EMISSIONS AND ISOTOPES

This script implements a full atmospheric CH4 inversion that optimizes:
- Time-varying emissions E(t) for 5 sources × 2 hemispheres (45 years)
- Time-varying isotopic signatures δ¹³C(t), δD(t) for each source × hemisphere

State vector per year t:
  x(t) = [E_wet_sh(t), E_wet_nh(t), E_agri_sh(t), E_agri_nh(t), ...,
          δ¹³C_wet_sh(t), δ¹³C_wet_nh(t), ..., δD_wet_sh(t), δD_wet_nh(t), ...]

Total state vector: 45 years × (10 emissions + 10 δ¹³C + 10 δD + 6 sinks) = 45 × 36

Cost function: J(x) = (y - F(x))ᵀ S_y⁻¹ (y - F(x)) + (x - x_a)ᵀ S_a⁻¹ (x - x_a)

TWO-STEP APPROACH:
  Step 1: Optimize emissions only with CH4 observations (tight isotope constraints)
  Step 2: Optimize isotopes with full multi-tracer data (tight CH4 constraints)

Smmothing Methodology:
When you have 45 years of CH4 emissions to estimate and you only have sparse atmospheric observations to constrain them, the inversion is mathematically under-determined. Without any additional constraint, the solver can satisfy the observations by producing emission time series that jump wildly from year to year — 40% up in one year, 30% down the next — because many such spiky solutions fit the data equally well. Physically, no real source behaves that way. The smoothing constraint says: prefer solutions where parameters change gradually over time.
The difference operator D
The heart of the smoothing is a matrix called D, built by the build_smoothness_matrix function. It is set up so that when you multiply it by the state vector (the full set of parameters across all years), it computes year-to-year differences for every parameter.
The code supports two orders:
Order 1 (first difference, what is actually used here via smoothness_order = 1): For each consecutive pair of years, D computes x[year+1] − x[year]. If this is large, it means the parameter jumped a lot from one year to the next.
Order 2 (second difference, commented as "usually better"): D computes x[year−1] − 2×x[year] + x[year+1]. This is a discrete version of the second derivative — it measures how much the trajectory is curving or changing direction. Penalising this prevents not just jumps but sharp turns in the time series.
D is a sparse matrix with a very specific pattern. For first differences, each row has a −1 at the current year's parameter column and a +1 at the next year's column, and zeros everywhere else. For second differences, each row has +1, −2, +1 at the previous, current, and next year positions. The matrix is built separately for emissions, δ¹³C, δD, and sinks, because each group of parameters lives at different positions in the state vector.

The smoothness cost term
Rather than baking smoothness into the prior covariance matrix (which the code also shows as an older approach it stepped away from), the active method adds a separate penalty term directly to the cost function:
J_smooth = λ × ‖ D(x − x_prior) / σ ‖²
Breaking this down piece by piece:
x − x_prior is the departure from the prior — how much the inversion has moved away from its starting guess. Smoothing is applied to this departure, not to the raw values, so if the prior itself is spiky, the inversion is not penalised for that spikiness, only for adding new spikiness on top.
D × (x − x_prior) applies the difference operator to those departures. The result is a vector of year-to-year changes (or curvatures) in whatever the inversion is trying to do to the parameters.
Dividing by σ (the prior uncertainty for each parameter type) makes the penalty dimensionless and comparable across very different quantities — methane emissions in Tg/yr, δ¹³C in ‰, and sink anomalies in percent all have wildly different numerical scales, and without this normalisation, a λ tuned for emissions would have no effect on isotopes or vice versa.
Squaring and summing gives a single number. Multiplying by λ controls how strongly to penalise roughness.

Separate λ weights
There is a different λ for each parameter group:
- Emissions: λ = 10 (strongest smoothing — annual emission totals should be fairly continuous)
- δ¹³C isotope signatures: λ = 0.1 (gentle smoothing — these can evolve but not too wildly)
- δD isotope signatures: λ = 0.01 (very mild smoothing — weakest constraint)
- Sinks: λ = 1.0 (moderate)
Higher λ means the solver is penalised more for any year-to-year roughness, so it will produce a smoother posterior even at the cost of fitting the observations slightly less well.

In each Gauss-Newton iteration, the total cost is:
J_total = J_observations + J_prior + J_smooth
The J_observations term pulls the solution toward fitting the atmospheric data. The J_prior term pulls it back toward the prior. The J_smooth term pulls it toward a temporally smooth trajectory.
To minimise J_total, the solver needs its gradient. The gradient of the smoothness term is:
∂J_smooth/∂x = 2λ × D^T × (D(x − x_prior)) / σ²
This is computed analytically in compute_smoothness_penalty and added to the total gradient before the update step. The corresponding Hessian contribution D^T D (scaled by 2λ/σ²) is also added to the left-hand side of the linear system that the solver uses to compute each Newton step, so that the update direction already accounts for smoothness from the very first step.

Before the inversion, the prior emissions may already have a mean year-to-year jump of 15–25%. The diagnostics function print_smoothness_diagnostics compares these statistics before and after. After the inversion with smoothing active, the posterior should show smaller mean and maximum year-to-year changes, without being forced to a constant — real interannual variability driven by strong observational signals (e.g. ENSO-driven wetland flux anomalies) can still appear, because the λ weights are finite and the observations can always override the smoothing if the signal is large enough. The smoothing simply ensures that any variability in the solution is earned by the data, not invented by numerical noise.

Author: Bibhasvata Dasgupta
