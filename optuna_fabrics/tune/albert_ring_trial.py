import numpy as np
import quaternionic

from optuna_fabrics.tune.albert_trial import AlbertTrial
from optuna_fabrics.tune.ring_trial import RingTrial
from mpscenes.goals.goal_composition import GoalComposition
from mpscenes.obstacles.sphere_obstacle import SphereObstacle

def generate_random_orientation(mean, rotation=0.0, tilting=0.0):
    """
    Generate random orientation of feasible reaching tasks.

    params
    mean:list
        mean of quaternion, good: mean = [0.0, 0.707, 0.0, 0.0]
    rotation: float
        amount of rotation around z axis
    tilting: float
        amount of tilting
    """
    lower_limit = np.array([0, 0, -rotation, -tilting])
    upper_limit = np.array([0, 0, rotation, tilting])
    orientation = np.array(mean) + np.random.uniform(lower_limit, upper_limit, 4)
    return orientation.tolist()

def generate_random_position():
    return np.random.uniform(np.array([0.0, 0.0, 1.2]), np.array([1.0, 1.2, 1.4]), 3)

class AlbertRingTrial(AlbertTrial, RingTrial):

    def __init__(self, weights=None):
        super().__init__(weights=weights)
        self._sub_goal_0_links = ['origin', 'panda_hand']
        self._sub_goal_1_links = ['panda_link7', 'panda_hand']

    def shuffle_env(self, env):
        # Definition of the goal.
        mean = [0.0, 0.707, 0.0, 0.0]
        goal_orientation = generate_random_orientation(mean, rotation=0.1, tilting=0.1)
        goal_dict = {
            "subgoal0": {
                "m": 3,
                "w": 0.0,
                "is_primary_goal": True,
                "indices": [0, 1, 2],
                "parent_link": self._sub_goal_0_links[0],
                "child_link": self._sub_goal_0_links[1],
                "desired_position": generate_random_position().tolist(),
                "high": [0.7, 0.2, 0.7],
                "low": [0.5, -0.2, 0.6],
                "epsilon": 0.05,
                "type": "staticSubGoal",
            },
            "subgoal1": {
                "m": 3,
                "w": 3.0,
                "is_primary_goal": False,
                "indices": [0, 1, 2],
                "parent_link": self._sub_goal_1_links[0],
                "child_link": self._sub_goal_1_links[1],
                "desired_position": [0.107, 0, 0],
                "angle": goal_orientation,
                "epsilon": 0.05,
                "type": "staticSubGoal",
            }
        }
        goal = GoalComposition(name="goal", content_dict=goal_dict)
        env.add_goal(goal)
        # Definition of the obstacle.
        radius_ring = 0.31
        obstacles = []
        ring_orientation = generate_random_orientation(goal_orientation, rotation=0.1, tilting=0.1)
        #goal_orientation = np.random.random(4).tolist()
        rotation_matrix_ring = quaternionic.array(ring_orientation).to_rotation_matrix
        whole_position = goal.is_primary_goalGoal().position()
        for i in range(self._obstacle_resolution + 1):
            angle = i/self._obstacle_resolution * 2.*np.pi
            origin_position = [
                0.0,
                float(radius_ring * np.cos(angle)),
                float(radius_ring * np.sin(angle)),
            ]
            position = np.dot(np.transpose(rotation_matrix_ring), origin_position) + whole_position
            static_obst_dict = {
                "dim": 3,
                "type": "sphere",
                "geometry": {"position": position.tolist(), "radius": 0.1},
            }
            obstacles.append(SphereObstacle(name="staticObst", content_dict=static_obst_dict))
        for obst in obstacles:
            env.add_obstacle(obst)
        return env, obstacles, goal

