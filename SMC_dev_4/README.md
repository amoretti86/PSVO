This README is an ongoing work.

To run the code, run 

`python runner_flag.py`, use default parameters or specify with `--flag_name=flag_value`.



-----

# FFBS

- run forward filtering pass to obtain $x_{1:T}^{1:K}$
- run backward simulation pass to obtatin $\tilde{x}_{1:T}^{1:M}$ and FFBS_log_weights.



use $p(y_{1:T}) = p(y_1) \prod_{t=2}^T p(y_t | y_{1:t-1})$,

where $p(y_t | y_{1:t-1}) = \int p(x_{t-1}|y_{1:t-1}) f(x_t | x_{t-1})g(y_t|x_t) d x_{t-1:t}$



where $x_t, x_{t-1}$ use the backward samples, with the associated backward weights.



the loss in computed as :

$\sum_{t=1}^T \log \sum_{m=1}^M \exp (\text{FFBS_log_weights})$

