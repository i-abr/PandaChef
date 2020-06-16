import time
import numpy as np
import math
import gym
from gym.spaces import Box
import pybullet as bullet_client
import pybullet_data as pd

useNullSpace            = 1
ikSolver                = 0
pandaEndEffectorIndex   = 12 #8
pandaNumDofs            = 7

ul = np.array([0.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
#upper limits for null space (todo: set them to proper range)
ll = np.array([-0.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
#joint ranges for null space (todo: set them to proper range)
jr = ul-ll
#restposes for null space
jointPositions=[0., 0.258, 0., -2.24, 0., 2.66, 0., 0.02, 0.02]
rp = jointPositions

init_pizza_pose = np.array([0.8, 0., 0.3])

class PandaChefEnv(object):
    def __init__(self, render=False, time_step = 0.002, frame_skip=5):
        self._time_step = time_step
        self._frame_skip = frame_skip
        self._render = render
        if render:
            bullet_client.connect(bullet_client.GUI)
        else:
            bullet_client.connect(bullet_client.DIRECT)
        bullet_client.setAdditionalSearchPath(pd.getDataPath())
        bullet_client.setTimeStep(time_step)
        bullet_client.setGravity(0., 0., -9.81)
        flags = bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        bullet_client.loadURDF("plane.urdf", np.array([0.,0.,0.]), flags=flags, useFixedBase=True)
        self.robot_id = bullet_client.loadURDF("./franka_panda_chef/panda_chef.urdf", np.array([0,0,0]), useFixedBase=True, flags=flags)
        self.pizza_id = bullet_client.loadURDF('./objects/sphere_with_restitution.urdf', init_pizza_pose, flags=flags)
        self.low_bnds = np.array([0.4, 0.1, -0.5])
        self.high_bnds = np.array([0.8, 0.7, 0.5])
        self._set_cmd = (self.high_bnds+self.low_bnds)/2.0
        self.action_scale = np.array([0.1,0.1,0.1])
        self.action_space = Box(low=np.array([-1,-1., -1.]), high=np.array([1., 1., 1.]))
        self.reset()

    def reset(self):
        index=0
        for j in range(bullet_client.getNumJoints(self.robot_id)):
            bullet_client.changeDynamics(self.robot_id, j, linearDamping=0, angularDamping=0.1)
            info = bullet_client.getJointInfo(self.robot_id, j)
            jointName = info[1]
            jointType = info[2]
            if (jointType == bullet_client.JOINT_PRISMATIC):
                bullet_client.resetJointState(self.robot_id, j, jointPositions[index])
                index=index+1
            if (jointType == bullet_client.JOINT_REVOLUTE):
                bullet_client.resetJointState(self.robot_id, j, jointPositions[index])
                index=index+1
        self.__prevPose = bullet_client.getLinkState(self.robot_id, pandaEndEffectorIndex)
        bullet_client.resetBasePositionAndOrientation(self.pizza_id, init_pizza_pose, [0.,0.,0.,1.])
        bullet_client.resetBaseVelocity(self.pizza_id, np.zeros(3), np.zeros(3))
    def step(self, action):
        # dcmd = self.action_space.sample()
        dcmd = np.clip(action, -1,1)
        new_cmd = np.clip(self._set_cmd + dcmd * self.action_scale, self.low_bnds, self.high_bnds)
        new_cmd = 0.5 * new_cmd + 0.5 * self._set_cmd
        pos = [new_cmd[0], 0, new_cmd[1]]
        orn = bullet_client.getQuaternionFromEuler([0.,new_cmd[2], np.pi])
        jointPoses = bullet_client.calculateInverseKinematics(self.robot_id,
                        pandaEndEffectorIndex, targetPosition=pos, targetOrientation=orn,
                        lowerLimits=ll,
                        upperLimits=ul,
                        jointRanges=jr,
                        restPoses=rp,
                        maxNumIterations=5)
        for i in range(pandaNumDofs):
            bullet_client.setJointMotorControl2(
                self.robot_id, i, bullet_client.POSITION_CONTROL, jointPoses[i], force=1 * 240.)
        for _ in range(self._frame_skip):
            bullet_client.stepSimulation()
        current_pos = bullet_client.getLinkState(self.robot_id, pandaEndEffectorIndex)
        self._set_cmd = new_cmd
