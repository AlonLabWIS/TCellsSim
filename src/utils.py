import scipy.stats as st


def f_norm(mean=0., scale=1., factor=1., skew=0):
    def inner(x):
        x_standard = (x - mean) / scale
        return factor * 2 * st.norm.pdf(x_standard) * st.norm.cdf(skew * x_standard) / scale

    return inner
