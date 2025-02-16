import sys

import numpy as np
import requests
import sympy as sp
from IPython.display import Markdown, display
from sympy import Q

sys.path.append("/content/conda_dir")

welcome = "Packages are loaded. ¡Welcome!"


def Print(message):
    display(Markdown(message))


def print_eq(suffix, eq):
    display(Markdown(suffix + r" $%s$ " % latex(eq)))


euler = (sp.Symbol("e", positive=True), sp.exp(1))


def Rz(phi):
    return sp.Matrix([[sp.cos(phi), -sp.sin(phi), 0], [sp.sin(phi), sp.cos(phi), 0], [0, 0, 1]])


def Rz_2D(phi):
    return sp.Matrix([[sp.cos(phi), -sp.sin(phi)], [sp.sin(phi), sp.cos(phi)]])


def FirstOrderTaylor(h, x, x0):
    dx = x
    evalx = list(zip(x, x0))
    h = h.expand(complex=True)
    happ = h.subs(evalx)
    for xi, dxi in zip(x, dx):
        happ += sp.diff(h, xi).subs(evalx) * dxi
    return sp.simplify(happ)


class Hexagonal:
    lat_vec = sp.Array(
        [
            [sp.Rational(1, 2), sp.sqrt(3) / 2, sp.Integer(0)],
            [sp.Rational(-1, 2), sp.sqrt(3) / 2, sp.Integer(0)],
            [sp.Integer(0), sp.Integer(0), sp.Integer(1)],
        ]
    )

    rec_vec = sp.Array(sp.Matrix(lat_vec).inv().T * 2 * sp.pi)
    K0 = np.array(sp.Matrix((rec_vec[0] - rec_vec[1]) / 3)).flatten()
    K1 = np.array(sp.Matrix(-(rec_vec[0] - rec_vec[1]) / 3)).flatten()


graphene = Hexagonal()


def dot(x, y):
    return sp.simplify((sp.conjugate(x.T) * y)[0])


def norm(x):
    return sp.sqrt(dot(x, x))


def normalize(x):
    return sp.simplify(x) / norm(x)


def delta(x, y):
    if x == y:
        return 1
    return 0


def EigenSystem(H, coord_syst=[]):
    Es = []
    Psis = []
    eigsyst = H.eigenvects()
    for eigv, mul, deg_eigs in eigsyst:
        for eig in deg_eigs:
            Es.append(eigv.subs(coord_syst))
            Psis.append(eig.subs(coord_syst))
    Es = [E for E in Es]
    Psis = [sp.simplify(normalize(sp.simplify(Psi))) for Psi in Psis]
    return Es, Psis


def exps2trig(x):
    return sp.trigsimp(sp.simplify(x).rewrite(sp.sin))


def half_angles(expr, theta):
    half_ang = [
        (1 - sp.cos(theta), 2 * sp.sin(theta / 2) ** 2),
        (1 + sp.cos(theta), 2 * sp.cos(theta / 2) ** 2),
        (2 + 2 * sp.cos(theta), 4 * sp.cos(theta / 2) ** 2),
    ]
    trig_assum = Q.positive(sp.sin(theta / 2)) & Q.positive(sp.cos(theta / 2))

    return exps2trig(sp.refine(sp.simplify(sp.simplify(expr).subs(half_ang)), trig_assum))


grafeno_real = "https://i.pinimg.com/564x/27/73/60/277360f8a8265b0a8008d1d03023a7be.jpg"
grafeno_disp = "https://icn2.cat/images/20190802_spin-communication.jpg"


def ejemplo_grafeno_real():
    return display.Image(requests.get(grafeno_real).content)


def ejemplo_dispositivo_grafeno():
    return display.Image(requests.get(grafeno_disp).content)
