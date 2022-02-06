from math import pow, pi, tan, sqrt
from random import uniform


class Mean:
    @classmethod
    def arithmetic(cls, vec):
        try:
            return sum(vec) / len(vec)
        except ZeroDivisionError:
            return 0

    @classmethod
    def lehmer(cls, vec):
        try:
            pow_list = [pow(x, 2) for x in vec]
            return sum(pow_list) / sum(vec)
        except ZeroDivisionError:
            return 0

    @classmethod
    def arithmetic_weighted(cls, differences, s):
        w_denominator = sum(differences)
        w = [differences[k] / w_denominator for k in range(len(s))]
        m = [s[k] * w[k] for k in range(len(s))]
        return sum(m)

    @classmethod
    def lehmer_weighted(cls, differences, s):
        w_denominator = sum(differences)
        m_numerator, m_denominator = 0, 0
        for k in range(len(s)):
            w = differences[k] / w_denominator
            m_numerator += pow(s[k], 2) * w
            m_denominator += s[k] * w
        return m_numerator / m_denominator


class Rand:
    @classmethod
    def cauchy(cls, loc, scale=0.1):
        return loc + (scale * tan(pi * (uniform(0, 1) - 0.5)))

    @classmethod
    def get_random_value(cls, low=0.0, high=1.0):
        return low + (uniform(0, 1) * (high - low))


def distance(a, b):
    return sqrt(pow((a['x'] - b['x']), 2) + pow((a['y'] - b['y']), 2))