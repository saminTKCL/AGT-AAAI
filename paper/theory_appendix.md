# Theory Appendix: Closed-Form Optimal Granularity under Degree-Corrected CSBM

## 1. Setup

Consider a degree-corrected contextual stochastic block model (DC-CSBM) with $C$ classes.
Each node $v$ has expected degree $d_v$, latent class $y_v \in \{1,\ldots,C\}$, and feature
$\mathbf{x}_v = \boldsymbol{\mu}_{y_v} + \boldsymbol{\epsilon}_v$ with $\boldsymbol{\epsilon}_v \sim \mathcal{N}(0, \sigma^2 I)$.

Let $h_v \in [0,1]$ denote **local homophily** (fraction of same-class neighbors).
Under DC-CSBM, one-hop aggregated feature signal aligns with $\boldsymbol{\mu}_{y_v}$ with strength
proportional to $h_v d_v$, while cross-class noise scales as $(1-h_v)d_v$.

## 2. Multi-Hop Signal-to-Noise

Define $k$-hop normalized aggregation $A^k \mathbf{x}_v$ where $A$ is the random-walk matrix.
Under independence of neighbor classes at heterophilous edges, the expected class signal energy at hop $k$ is:

$$\mathbb{E}[\|\text{signal}_k(v)\|^2] \propto h_v^k \cdot d_v$$

Cross-class contamination accumulates as:

$$\mathbb{E}[\|\text{noise}_k(v)\|^2] \propto (1-h_v) \cdot \min(k, d_v) + \sigma^2$$

The per-node SNR after $k$-hop aggregation is therefore:

$$\text{SNR}_k(v) = \frac{h_v^k d_v}{(1-h_v)\min(k,d_v) + \sigma^2}$$

## 3. Optimizer

Differentiating the log-SNR w.r.t. continuous $k$ yields a unique interior maximum when $h_v < 1$.
Setting $\partial_k \log \text{SNR}_k = 0$ gives:

$$k^*(v) = \frac{\log\!\left(\frac{h_v d_v}{(1-h_v) + \sigma^2/d_v}\right)}{\log(1/h_v)}$$

For $h_v \to 1$, $k^* \to k_{\max}$; for $h_v \to 0$, $k^* \to k_{\min}$.

## 4. Practical Closed Form (Implemented)

Since $h_v$ is estimated on finite graphs, we use a stable monotone surrogate:

$$k^*(v) = k_{\min} + (k_{\max}-k_{\min}) \cdot \tilde{h}_v^{\alpha} \cdot (1 + \beta \log(1+d_v)) \cdot \widetilde{\text{SNR}}_v^{\gamma}$$

where $\tilde{h}_v = \max(h_v, h_0)$, $\widetilde{\text{SNR}}_v$ is a feature-space kNN SNR proxy,
and $(\alpha, \beta, \gamma) = (1.5, 0.35, 0.25)$ by default.

**Monotonicity:** $\partial k^*/\partial h_v > 0$, $\partial k^*/\partial d_v > 0$ on the observed range,
matching the GRAIN-TD3 policy correlation with local homophily (validated empirically).

## 5. Diffusion-Time Interpretation

Continuous granularity $k^*$ corresponds to heat-diffusion time $t_v = k^* \Delta t$ on the graph Laplacian.
AGT thus assigns each node its own diffusion stopping time, replacing RL-based per-node depth search
with a single analytic map — $O(E)$ precomputation, no replay buffer, no critic network.

## 6. Assumptions and Limitations

- Assumes local label homophily correlates with feature signal (standard in heterophily literature).
- Feature SNR proxy substitutes for unknown $\sigma^2$; optional calibration MLP absorbs residual error.
- The released GRAIN code uses explicit k-hop aggregation only; our AGT interfaces with the same
  fractional-hop operator for fair comparison.
