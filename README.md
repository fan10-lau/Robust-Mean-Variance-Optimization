# Robust Mean–Variance Optimization (MVO) 

This project is a variation of the MIE377 course project at University of Toronto, which comes with a provided baseline MVO strategy.  
I extend that baseline by incorporating **robust optimization** techniques.  


## Phase 1 - Robust MVO with Box Uncertainty
 
In Phase 1, I implement the Box Uncertainty formulation, which protects against estimation errors in expected returns. 

### Math
- Instead of assuming μ is exact, we allow it to vary within a box set:  

  $$
  \mu_i \in [\hat{\mu}_i - \delta_i, \; \hat{\mu}_i + \delta_i]
  $$

- The robust constraint becomes:

  $$
  \min_{\mu \in U} \mu^T x \geq R
  $$

- This protects against worst-case deviations in returns. 


