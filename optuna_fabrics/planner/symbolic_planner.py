from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner
from casadi import SX


def symbolic(name, dim):
    return SX.sym(name, dim)

class SymbolicFabricPlanner(ParameterizedFabricPlanner):
    def __init__(self, dof: int, robot_type: str, **kwargs):
        attractor_potential: str = (
            "ca.SX.sym('alpha_attractor', 1) * (ca.norm_2(x) + 1 / 10 * ca.log(1 + ca.exp(-2 * 10 * ca.norm_2(x))))"
        )
        collision_geometry: str = (
            "-sym('k_geo') / (x ** sym('exp_geo')) * xdot ** 2"
        )
        collision_finsler: str = (
            "sym('k_fin')/(x**sym('exp_fin')) * (-0.5 * (ca.sign(xdot) - 1)) * xdot**2"
        )
        base_energy: str = (
            "0.5 * sym('base_inertia') * ca.dot(xdot, xdot)"
        )
        limit_geometry: str = (
            "sym('k_limit_geo') / (x ** sym('exp_limit_geo')) * xdot ** 2"
        )
        limit_finsler: str = (
            "sym('k_limit_fin')/(x**sym('exp_limit_fin')) * (-0.5 * (ca.sign(xdot) - 1)) * xdot**2"
        )
        kwargs['base_energy'] = base_energy
        kwargs['collision_finsler'] = collision_finsler
        kwargs['collision_geometry'] = collision_geometry
        kwargs['limit_geometry'] = limit_geometry
        kwargs['limit_finsler'] = limit_finsler
        super().__init__(
            dof,
            robot_type,
            **kwargs,
        )
