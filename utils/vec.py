import math


class Vec:
    def __init__(self, i, j):
        self.i = i
        self.j = j

    @property
    def x(self):
        return self.j

    @x.setter
    def x(self, x):
        self.j = x

    @property
    def y(self):
        return self.i

    @y.setter
    def y(self, y):
        self.i = y

    @property
    def ij(self):
        return self.i, self.j

    def __eq__(self, other):
        return self.i == other.i and self.j == other.j

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.i, self.j))

    def __neg__(self):
        return Vec(-self.i, -self.j)

    def __add__(self, other):
        return Vec(self.i + other.i, self.j + other.j)

    def __sub__(self, other):
        return self + (-other)

    def dist_sq(self, other):
        return (self.i - other.i) ** 2 + (self.j - other.j) ** 2

    def dist(self, other):
        return math.sqrt(self.dist_sq(other))
