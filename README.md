# Privacy Random Variable (PRV) Accountant

A fast algorithm to optimally compose privacy guarantees of differentially private (DP) algorithms to arbitrary accuracy.
Our method is based on the notion of privacy loss random variables to quantify the privacy loss of DP algorithms.
For more details see [[1](https://arxiv.org/abs/2106.02848)].


## Installation

```
pip install prv-accountant
```


## Mechanisms

Currently the following mechanisms are supported:


### Subsampled Gaussian Mechanism

``` python
from prv_accountant import PoissonSubsampledGaussianMechanism
prv = PoissonSubsampledGaussianMechanism(noise_multiplier, sampling_probability)
```
which computes the privacy curve

$$ \delta \left ( \mathcal{N}(0, \sigma^2) \| (1-p) \mathcal{N}(0, \sigma^2) + p \mathcal{N}(1, \sigma^2) \right ), $$

where $p$ is the sampling probability and $\sigma$ is the noise multiplier.
The second argument represents a mixture distribution.


### Gaussian Mechanism

``` python
from prv_accountant import GaussianMechanism
prv = GaussianMechanism(noise_multiplier)
```
which computes the privacy curve

$$ \delta \left ( \mathcal{N}(0, \sigma^2) \| \mathcal{N}(1, \sigma^2) \right ), $$

where $\sigma$ is the noise multiplier.


### Laplace Mechanism

``` python
from prv_accountant import LaplaceMechanism
prv = LaplaceMechanism(mu)
```
which computes the privacy curve

$$ \delta \left ( \textsf{Lap}(0, 1) \| \textsf{Lap}(\mu, 1) \right ). $$


### Pure-DP and Approximate-DP

It is also possible to compose DP guarantees directly
- pure $\varepsilon$-DP guarantees using `prv_accountant.PureDPMechanism(epsilon)`
- approximate $(\varepsilon, \delta)$-DP guarantees using `prv_accountant.ApproximateDPMechanism(epsilon, delta)`


### Custom Mechanisms

It is also possible to add custom mechanisms for the composition computation.
An example can be found in [this notebook](notebooks/laplace-mechanism.ipynb).
All we need is to implement the CDF of the privacy loss distribution.


## Example

### Heterogeneous Composition

It is possible to compose different mechanisms.
The following example will compute the composition of three different mechanism $M^{(a)}, M^{(b)}$ and $M^{(c)}$ composed with themselves $m, n$ and $o$ times, respectively.

An application for such a composition is DP-SGD training with increasing batch size and therefore increasing sampling probability.
After $m+n+o$ training steps, the resulting privacy mechanism $M$ for the whole training process is given by $M = M_1^{(a)} \circ \dots \circ M_m^{(a)} \circ M_1^{(b)} \circ \dots \circ M_n^{(b)} \circ M_1^{(c)} \circ \dots \circ M_o^{(c)}$.

Using the `prv_accountant` we need to create a privacy random variable for each mechanism

```python
from prv_accountant.privacy_random_variables import PoissonSubsampledGaussianMechanism, GaussianMechanism, LaplaceMechanism

prv_a = PoissonSubsampledGaussianMechanism(noise_multiplier=0.8, sampling_probability=5e-3)
prv_b = GaussianMechanism(noise_multiplier=8.0)
prv_c = LaplaceMechanism(mu=0.1)

m = 100
n = 200
o = 100
```

Next, we need to create an accountant instance.
The accountant will take care of most of the numerical intricacies such as finding the support of the PRV and discretisation.
In order to find a suitable domain, the accountant needs to know about the largest number of compositions of each PRV with itself that will be computed.
Larger values of `max_self_compositions` lead to larger domains which can cause slower performance.
In the case of DP-SGD, a reasonable choice of `max_self_compositions` would be the total number of training steps.
Additionally, the desired error bounds for $\varepsilon$ and $\delta$ are required.

```python
from prv_accountant import PRVAccountant

accountant = PRVAccountant(
    prvs=[prv_a, prv_b, prv_c],
    max_self_compositions=[1_000, 1_000, 1_000],
    eps_error=0.1,
    delta_error=1e-10
)
```

Finally, we're ready to compute the composition.
The final bounds and estimates for $\varepsilon$ for the mechanism $M$ are

```python
eps_low, eps_est, eps_up = accountant.compute_epsilon(delta=1e-6, num_self_compositions=[m, n, o])
```


### DP-SGD

For homogeneous DP-SGD (i.e. constant noise multiplier and constant sampling probability) things are even simpler.
We provide a simple command line utility for getting epsilon estimates.

```
compute-dp-epsilon --sampling-probability 5e-3 --noise-multiplier 0.8 --delta 1e-6 --num-compositions 1000
```

Or, use it in python code

```python
from prv_accountant.dpsgd import DPSGDAccountant

accountant = DPSGDAccountant(
    noise_multiplier=0.8,
    sampling_probability=5e-3,
    delta=1e-6,
    eps_error=0.1,
    delta_error=1e-10,
    max_compositions=1000
)

eps_low, eps_estimate, eps_upper = accountant.compute_epsilon(num_compositions=1000)
```

For more examples, have a look in the `notebooks` directory.


## References

[1] [Sivakanth Gopi, Yin Tat Lee, Lukas Wutschitz. (2021). Numerical composition of differential privacy. Advances in Neural Information Processing Systems](https://proceedings.neurips.cc/paper/2021/hash/6097d8f3714205740f30debe1166744e-Abstract.html)


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.


## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
