import numpyro

def reparameterize_nb(loc, scale, eps=1e-6):
    assert (loc is None) == (
        scale is None
    ), "If using the loc/scale NB parameterization, both parameters must be specified"
    rate = scale / loc
    total_count = scale
    return total_count, rate


class NegativeBinomial(numpyro.distributions.GammaPoisson):
    def __init__(
        self,
        loc,
        scale
    ):
        total_count, rate = reparameterize_nb(loc, scale)
        super().__init__(total_count, rate)