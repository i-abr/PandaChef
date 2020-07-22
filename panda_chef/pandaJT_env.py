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
jointPositions=[0., 0.158, 0., -2.24, 0., 2.66, 0., 0.02, 0.02]
rp = jointPositions

# init_pizza_pose = np.array([0.8, 0., 0.3+0.5])
init_pizza_pose = np.array([0.8, 0., 0.3])

pan_default_orn = np.array(bullet_client.getQuaternionFromEuler([0.,0., np.pi]))
target_flip_orn = np.array(bullet_client.getQuaternionFromEuler([0., -np.pi,0.]))
jnt_ctrl_idx = [1,3,5]

class PandaChefEnv(object):
    def __init__(self, render=False, time_step = 1./60.,frame_skip=1):
        self._time_step = time_step
        self._frame_skip = frame_skip
        self._render = render
        if render:
            bullet_client.connect(bullet_client.GUI)
            bullet_client.configureDebugVisualizer(bullet_client.COV_ENABLE_GUI, 0)
        else:
            bullet_client.connect(bullet_client.DIRECT)
        bullet_client.setAdditionalSearchPath(pd.getDataPath())
        bullet_client.setTimeStep(time_step)
        bullet_client.setGravity(0., 0., -9.81)
        flags = bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        self.floor_id = bullet_client.loadURDF("plane.urdf", np.array([0.,0.,0.]), flags=flags, useFixedBase=True)
        self.robot_id = bullet_client.loadURDF("./franka_panda_chef/panda_chef.urdf", np.array([0,0,0]), useFixedBase=True, flags=flags)
        self.pizza_id = bullet_client.loadURDF('./objects/sphere_with_restitution.urdf', init_pizza_pose, flags=flags)
        self.low_bnds = np.array([0.4, 0.1, -0.5])
        self.high_bnds = np.array([0.8, 0.7, 0.5])
        self._set_cmd = (self.high_bnds+self.low_bnds)/2.0
        self._past_ee_pos = None
        self.action_scale = np.array([2.,2.,2.])
        # self.action_scale = np.array([120,120,22])
        self.action_space = Box(low=np.array([-1,-1., -1.]), high=np.array([1., 1., 1.]))
        self.reset()

        obs = self.get_obs()
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=obs.shape)

        self._max_flip_reward = 0.

    def reset(self):
        self._set_cmd = (self.high_bnds+self.low_bnds)/2.0
        self._max_flip_reward = 0.
        bullet_client.changeDynamics(self.pizza_id, -1, linearDamping=0.1, angularDamping=0.2)
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
        return self.get_obs()

    def get_reward(self, ee_state, pizza_state, action):
        ee_pos = np.array(ee_state[4])
        ee_orn = ee_state[5]
        pizza_pos, pizza_orn = pizza_state[0]
        pizza_pos = np.array(pizza_pos)
        pizza_linear_vel, pizza_angular_vel = pizza_state[1]
        pan_reward = -np.linalg.norm(ee_orn-pan_default_orn)**2
        catch_rew = -0.1*np.linalg.norm(ee_pos-pizza_pos)**2
        catch_rew += np.clip(pizza_pos[2]-ee_pos[2], -np.inf, 0.)
        flip_rew = -np.linalg.norm(target_flip_orn-pizza_orn)**2
        if flip_rew > self._max_flip_reward:
            self._max_flip_reward = flip_rew
        jnt_reward = 0.
        for ctrl_idx in jnt_ctrl_idx:
            th, thdot, thFx, thJx = bullet_client.getJointState(self.robot_id, ctrl_idx)
            jnt_reward += -(th-rp[ctrl_idx])**2 -0.001*(thdot)**2
        return catch_rew + flip_rew + jnt_reward

    def get_obs(self):
        ee_state    = bullet_client.getLinkState(self.robot_id, pandaEndEffectorIndex)
        pizza_config  = bullet_client.getBasePositionAndOrientation(self.pizza_id)
        pizza_vel   = bullet_client.getBaseVelocity(self.pizza_id)
        ee_pos = np.array(ee_state[4])
        if self._past_ee_pos is None:
            self._past_ee_pos = ee_pos
        self._curr_ee_pos = ee_pos
        ee_orn = np.array(ee_state[5])
        pizza_pos = np.array(pizza_config[0])
        pizza_orn = np.array(pizza_config[1])
        pizza_linear_vel = np.array(pizza_vel[0])
        pizza_angular_vel = np.array(pizza_vel[1])
        jnts = []
        jnt_vels = []
        for ctrl_idx in jnt_ctrl_idx:
            th, thdot, thFx, thJx = bullet_client.getJointState(self.robot_id, ctrl_idx)
            jnts.append(th)
            jnt_vels.append(thdot)
        obs = np.concatenate([jnts, jnt_vels, ee_pos, pizza_pos, ee_orn, pizza_orn,  pizza_linear_vel, pizza_angular_vel])
        return obs

    def step(self, action):
        # dcmd = self.action_space.sample()
        cmd = np.clip(action, -1,1)*self.action_scale
        ctrl_idx = 0
        for i in range(pandaNumDofs):
            if i in jnt_ctrl_idx:
                bullet_client.setJointMotorControl2(
                    self.robot_id, i, bullet_client.VELOCITY_CONTROL, targetVelocity=cmd[ctrl_idx], force=47)
                # bullet_client.setJointMotorControl2(
                #     self.robot_id, i, bullet_client.TORQUE_CONTROL, force=cmd[ctrl_idx])
                ctrl_idx += 1
            # else:
            #     bullet_client.setJointMotorControl2(
            #         self.robot_id, i, bullet_client.POSITION_CONTROL, jointPositions[i], force=2 * 240.)
        for _ in range(self._frame_skip):
            bullet_client.stepSimulation()
        ee_state    = bullet_client.getLinkState(self.robot_id, pandaEndEffectorIndex)
        pizza_config  = bullet_client.getBasePositionAndOrientation(self.pizza_id)
        pizza_vel   = bullet_client.getBaseVelocity(self.pizza_id)
        reward      = self.get_reward(ee_state, (pizza_config, pizza_vel), action)
        obs = self.get_obs()
        done = False
        # if len(bullet_client.getContactPoints(self.floor_id, self.pizza_id))>0:
        #     reward += -1
        # if np.abs(obs[0])>0.5 or np.abs(obs[2]) > 0.5:
        # bullet_client.addUserDebugLine(self._past_ee_pos, self._curr_ee_pos, lifeTime=0, lineColorRGB=[0,1,0])
        # self._past_ee_pos = self._curr_ee_pos
        return obs, reward, done, {}
