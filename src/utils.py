import scipy.stats as st


def f_norm(mean=0., scale=1., factor=1.):
    def inner(x):
        return factor * st.norm.pdf(x, loc=mean, scale=scale)

    return inner
