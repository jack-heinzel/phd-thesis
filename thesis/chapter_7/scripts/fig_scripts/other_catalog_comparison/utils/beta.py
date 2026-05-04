def mean_variance_to_alpha_beta(mean, variance, loc=0.0, scale=1.0):
    a, c = loc, loc+scale

    a_minus_mean = a - mean
    c_minus_mean = c - mean

    shared_term = (
        (a_minus_mean*c_minus_mean + variance) /
        (scale * variance)
    )

    alpha =  shared_term * a_minus_mean
    beta  = -shared_term * c_minus_mean

    return alpha, beta


def alpha_beta_to_mean_variance(alpha, beta, loc=0.0, scale=1.0):
    a, c = loc, loc+scale
    alpha_plus_beta = alpha + beta

    mean = (alpha*c + beta*a) / alpha_plus_beta

    variance = (
        xpy.square(scale) * alpha*beta /
        (xpy.square(alpha_plus_beta) * (alpha_plus_beta + 1.0))
    )

    return mean, variance
