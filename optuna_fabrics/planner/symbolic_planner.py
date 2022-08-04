from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner
from casadi import SX


def symbolic(name, dim):
    return SX.sym(name, dim)

class SymbolicFabricPlanner(ParameterizedFabricPlanner):
    def __init__(self, dof: int, robot_type: str, **kwargs):
        collision_geometry: str = (
            "-sym('k_geo') / (x ** sym('exp_geo')) * xdot ** 2"
        )
        collision_finsler: str = (
            "sym('k_fin')/(x**sym('exp_fin')) * (-0.5 * (ca.sign(xdot) - 1)) * xdot**2"
        )
        limit_geometry: str = (
            "-sym('k_limit_geo') / (x ** sym('exp_limit_geo')) * xdot ** 2"
        )
        limit_finsler: str = (
            "sym('k_limit_fin')/(x**sym('exp_limit_fin')) * (-0.5 * (ca.sign(xdot) - 1)) * xdot**2"
        )
        self_collision_geometry: str = (
            "-sym('k_self_geo') / (x ** sym('exp_self_geo')) * xdot ** 2"
        )
        self_collision_finsler: str = (
            "sym('k_self_fin')/(x** sym('exp_self_fin')) * (-0.5 * (ca.sign(xdot) - 1)) * xdot**2"
        )
        """
        collision_geometry: str = (
            "-0.5 / (x ** 2) * xdot ** 2"
        )
        collision_finsler: str = (
            "0.1/(x**2) * (-0.5 * (ca.sign(xdot) - 1)) * xdot**2"
        )
        self_collision_geometry: str = (
            "-0.5 / (x ** 1) * xdot ** 2"
        )
        self_collision_finsler: str = (
            "0.1/(x**2) * (-0.5 * (ca.sign(xdot) - 1)) * xdot**2"
        )
        limit_geometry: str = (
            "-0.5/ (x ** 1) * xdot ** 2"
        )
        limit_finsler: str = (
            "0.4/(x**1) * (-0.5 * (ca.sign(xdot) - 1)) * xdot**2"
        )
        """
        base_energy: str = (
            "0.5 * sym('base_inertia') * ca.dot(xdot, xdot)"
        )
        damper_beta: str = (
            "0.5 * (ca.tanh(-sym('alpha_b') * (ca.norm_2(x) - sym('radius_shift'))) + 1) * sym('beta_close') + sym('beta_distant') + ca.fmax(0, sym('a_ex') - sym('a_le'))"
        )
        damper_eta: str = (
            "0.5 * (ca.tanh(-0.5 * sym('ex_lag') * (1 - sym('ex_factor')) - 0.5) + 1)"
        )


        kwargs['base_energy'] = base_energy
        kwargs['damper_beta'] = damper_beta
        kwargs['damper_eta'] = damper_eta
        kwargs['collision_finsler'] = collision_finsler
        kwargs['collision_geometry'] = collision_geometry
        kwargs['self_collision_finsler'] = self_collision_finsler
        kwargs['self_collision_geometry'] = self_collision_geometry
        kwargs['limit_geometry'] = limit_geometry
        kwargs['limit_finsler'] = limit_finsler
        super().__init__(
            dof,
            robot_type,
            **kwargs,
        )
