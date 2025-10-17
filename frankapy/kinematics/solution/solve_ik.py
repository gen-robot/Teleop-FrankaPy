"""
Solves the basic IK problem.
"""

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as onp
import pyroki as pk
from typing import Union


def solve_ik(
    robot: pk.Robot,
    target_link_name: str,
    target_wxyz: onp.ndarray,
    target_position: onp.ndarray,
    initial_guess: onp.ndarray = None,
) -> onp.ndarray:
    """
    Solves the basic IK problem for a robot.

    Args:
        robot: PyRoKi Robot.
        target_link_name: String name of the link to be controlled.
        target_wxyz: onp.ndarray. Target orientation.
        target_position: onp.ndarray. Target position.

    Returns:
        cfg: onp.ndarray. Shape: (robot.joint.actuated_count,).
    """
    assert target_position.shape == (3,) and target_wxyz.shape == (4,)
    target_link_index = robot.links.names.index(target_link_name)
    cfg = _solve_ik_jax(
        robot,
        jnp.array(target_link_index),
        jnp.array(target_wxyz),
        jnp.array(target_position),
        jnp.array(initial_guess) if initial_guess is not None else None,
    )
    assert cfg.shape == (robot.joints.num_actuated_joints,)
    return onp.array(cfg)

@jdc.jit
def _solve_ik_jax(
    robot: pk.Robot,
    target_link_index: jax.Array,
    target_wxyz: jax.Array,
    target_position: jax.Array,
    initial_guess: jax.Array = None,
) -> jax.Array:
    """Single IK solve with optional initial guess."""
    joint_var = robot.joint_var_cls(0)
    factors = [
        pk.costs.pose_cost_analytic_jac(
            robot,
            joint_var,
            jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(target_wxyz), target_position
            ),
            target_link_index,
            pos_weight=50.0,
            ori_weight=10.0,
        ),
        pk.costs.limit_cost(
            robot,
            joint_var,
            weight=100.0,
        ),
    ]
    
    # Add initial guess cost if provided
    if initial_guess is not None:
        factors.append(
            pk.costs.rest_cost(
                joint_var,
                initial_guess,
                weight=35.0,
            )
        )
    
    # Set up initial values
    if initial_guess is not None:
        initial_vals = jaxls.VarValues.make((joint_var.with_value(initial_guess),))
    else:
        initial_vals = None
    
    sol = (
        jaxls.LeastSquaresProblem(factors, [joint_var])
        .analyze()
        .solve(
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
            initial_vals=initial_vals,
        )
    )
    return sol[joint_var]


def solve_batch_ik_with_continuity(
    robot: pk.Robot,
    target_link_name: str,
    target_wxyz_sequence: Union[onp.ndarray, jnp.ndarray],
    target_position_sequence: Union[onp.ndarray, jnp.ndarray],
    initial_guess: Union[onp.ndarray, jnp.ndarray] = None,
) -> onp.ndarray:

    target_position_sequence = jnp.array(target_position_sequence)
    target_wxyz_sequence = jnp.array(target_wxyz_sequence)
    positions_transposed = jnp.swapaxes(target_position_sequence, 0, 1)  # [B, T, 3]
    wxyzs_transposed = jnp.swapaxes(target_wxyz_sequence, 0, 1)  # [B, T, 4]
    
    target_link_index = robot.links.names.index(target_link_name)
    
    T, B = target_position_sequence.shape[:2]
    if initial_guess is None:
        initial_guess = jnp.tile(
            jnp.array(robot.joint_var_cls(0).default_factory()), (B, 1)
        )
    else:
        initial_guess = jnp.atleast_2d(jnp.array(initial_guess))
        if initial_guess.shape[0] == 1 and B > 1:
            initial_guess = jnp.repeat(initial_guess, B, axis=0)

    @jdc.jit
    def _solve_sequence_with_scan_jit(
        initial_guess,
        poses,
        robot,
        target_link_index,
    ):
        def ik_step(prev_solution, pose_data):
            position, wxyz = pose_data
            solution = _solve_ik_jax(
                robot,
                target_link_index,
                wxyz,
                position,
                prev_solution
            )
            return solution, solution

        _, all_solutions = jax.lax.scan(ik_step, initial_guess, poses)
        return all_solutions

    batched_solver = jax.vmap(
        _solve_sequence_with_scan_jit, 
        in_axes=(0, (0, 0), None, None)
    )

    all_solutions = batched_solver(
        initial_guess, 
        (positions_transposed, wxyzs_transposed),
        robot,
        jnp.array(target_link_index)
    )

    solutions_transposed = jnp.swapaxes(all_solutions, 0, 1) # return to [T, B,...]
    
    return onp.array(solutions_transposed)