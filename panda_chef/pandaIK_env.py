import time
import numpy as np
import math
import gym
from gym.spaces import Box
import pybullet as bullet_client
import pybullet_data as pd
import os
from copy import deepcopy
dir_path = os.path.dirname(os.path.realpath(__file__))

useNullSpace            = 1
ikSolver                = 0
pandaEndEffectorIndex   = 12 #8
pandaNumDofs            = 7

ul = np.array([1.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
#upper limits for null space (todo: set them to proper range)
ll = np.array([-1.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
#joint ranges for null space (todo: set them to proper range)
jr = ul-ll
#restposes for null space
jointPositions=[0., 0., 0., -2.5, 0., 2.5, 0., 0.02, 0.02]
rp = jointPositions

# positio and orientation
init_pizza_pose = np.array([0.7, 0., 0.2])
init_pizza_state = np.array([0.7, 0.2, 0.])
zero_ee_pose    = (np.array([0.7, 0., 0.13]), np.array([0.,0., np.pi]))

init_pancake_rot = bullet_client.getQuaternionFromEuler([0.,np.pi,np.pi])

def draw_coordinate(id, **kwargs):
    bullet_client.addUserDebugLine([0,0,0],[0.1,0,0],[1,0,0],parentObjectUniqueId=id, lineWidth=5, **kwargs)
    bullet_client.addUserDebugLine([0,0,0],[0,0.1,0],[0,1,0],parentObjectUniqueId=id, lineWidth=5, **kwargs)
    bullet_client.addUserDebugLine([0,0,0],[0,0,0.1],[0,0,1],parentObjectUniqueId=id, lineWidth=5, **kwargs)

class PandaChefEnv(object):
    def __init__(self, render=False, time_step = 1/100., frame_skip=1):
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
        bullet_client.loadURDF("plane.urdf", np.array([0.,0.,0.]), flags=flags, useFixedBase=True)
        self.robot_id = bullet_client.loadURDF(dir_path+'/urdf/panda_chef.urdf',
                                        np.array([0,0,0]), useFixedBase=True, flags=flags)
        self.pancake_id = bullet_client.loadURDF(dir_path+'/urdf/sphere_with_restitution.urdf',
                                        init_pizza_pose, init_pancake_rot, flags=flags)

        # setting up the bounds for the action space
        self._set_cmd = deepcopy(zero_ee_pose)
        self._bnds = (np.array([0.1]*3), np.array([0.4]*3))

        self.low_bnds = (self._set_cmd[0] - self._bnds[0], self._set_cmd[1] - self._bnds[1])
        self.high_bnds = (self._set_cmd[0] + self._bnds[0], self._set_cmd[1] + self._bnds[1])

        self.action_scale = (np.array([0.05, 0., 0.05]), np.array([0., 0.2, 0.]))
        self.action_space = Box(low=-np.ones(6), high=np.ones(6))

        # bullet_client.addUserDebugLine([0.4, 0., 0.1], [0.9, 0., 0.1])
        # bullet_client.addUserDebugLine([0.9, 0., 0.1], [0.9, 0., 0.7])
        # bullet_client.addUserDebugLine([0.9, 0., 0.7], [0.4, 0., 0.7])
        # bullet_client.addUserDebugLine([0.4, 0., 0.7], [0.4, 0., 0.1])

        # set up the coordinate sys for the pizza
        # bullet_client.addUserDebugText("baseLink", [0,0,0.05],
        #                     textColorRGB=[1,0,0],textSize=1.5,parentObjectUniqueId=self.pancake_id)
        # created some coordinate systems for debugging
        draw_coordinate(self.pancake_id)
        draw_coordinate(self.robot_id)

        # TODO: create coordinate for the target where you want the pancake to end up


        self.reset()

        obs = self.get_obs()
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=obs.shape)

    def reset(self):
        self._filt_cmd = 0.
        self.t = 0
        self._set_cmd = deepcopy(zero_ee_pose)
        index=0
        # bullet_client.changeDynamics(self.robot_id, j, linearDamping=0, angularDamping=0.1, restitution=0.)
        for j in range(bullet_client.getNumJoints(self.robot_id)):
            bullet_client.changeDynamics(self.robot_id, j, linearDamping=0, angularDamping=0.1, restitution=0.)
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

        bullet_client.resetBasePositionAndOrientation(self.pancake_id, init_pizza_pose, init_pancake_rot)
        bullet_client.resetBaseVelocity(self.pancake_id, np.zeros(3), np.zeros(3))
        return self.get_obs()

    def get_reward(self, ee_state, pizza_state, action):
        ee_pos = np.array(ee_state[4])
        ee_orn = ee_state[5]
        pizza_pos, pizza_orn = pizza_state[0]
        pizza_pos = np.array(pizza_pos)
        pizza_linear_vel, pizza_angular_vel = pizza_state[1]
        pizza_linear_vel = np.array(pizza_linear_vel)
        catch_rew = -np.sum((ee_pos-pizza_pos)**2)
        flip_rew = -np.clip(pizza_angular_vel[1],-5,5) #* (0.95**self.t)
        lin_vel_penalty = np.sum(pizza_linear_vel**2)
        return catch_rew + flip_rew - 1e-3*np.sum((action)**2) - lin_vel_penalty

    def get_obs(self):
        ee_state    = bullet_client.getLinkState(self.robot_id, pandaEndEffectorIndex, computeLinkVelocity=1)
        pizza_config  = bullet_client.getBasePositionAndOrientation(self.pancake_id)
        pizza_vel   = bullet_client.getBaseVelocity(self.pancake_id)
        ee_pos = np.array(ee_state[4])
        ee_orn = np.array(ee_state[5])
        ee_linvel = np.array(ee_state[6])
        ee_angvel = np.array(ee_state[7])
        pizza_pos = np.array(pizza_config[0])
        pizza_orn = np.array(pizza_config[1])
        pizza_linear_vel = np.array(pizza_vel[0])
        pizza_angular_vel = np.array(pizza_vel[1])
        # obs = np.concatenate([pizza_pos-ee_pos, pizza_orn-ee_orn,ee_pos,
        #             pizza_pos, ee_orn, ee_linvel, ee_angvel, pizza_orn,  pizza_linear_vel, pizza_angular_vel])
        obs = np.concatenate([ee_pos-pizza_pos, pizza_orn, pizza_linear_vel, pizza_angular_vel])

        return obs

    def step(self, action):
        dcmd = np.clip(action, -1,1)
        new_pos = np.clip(self._set_cmd[0] + dcmd[:3] * self.action_scale[0], self.low_bnds[0], self.high_bnds[0])
        new_orn = np.clip(self._set_cmd[1] + dcmd[3:] * self.action_scale[1], self.low_bnds[1], self.high_bnds[1])
        new_quat_orn = bullet_client.getQuaternionFromEuler(new_orn)

        # get IK EE pose
        jointPoses = bullet_client.calculateInverseKinematics(self.robot_id,
                        pandaEndEffectorIndex, targetPosition=new_pos, targetOrientation=new_quat_orn,
                        lowerLimits=ll, upperLimits=ul, jointRanges=jr,restPoses=rp,
                        maxNumIterations=5)
        # make each motor go to correct joint pose
        for i in range(pandaNumDofs):
            bullet_client.setJointMotorControl2(
                self.robot_id, i, bullet_client.POSITION_CONTROL, jointPoses[i], force=70.)
        # step the simulation forward
        for _ in range(self._frame_skip):
            bullet_client.stepSimulation()

        # get the state for reward calculatin
        ee_state    = bullet_client.getLinkState(self.robot_id, pandaEndEffectorIndex)
        pizza_config  = bullet_client.getBasePositionAndOrientation(self.pancake_id)
        pizza_vel   = bullet_client.getBaseVelocity(self.pancake_id)
        reward      = self.get_reward(ee_state, (pizza_config, pizza_vel), action)
        # self._set_cmd = new_cmd.copy()
        obs = self.get_obs()
        done = False
        # if obs[2]<-0.05 or np.abs(obs[0])>0.5 or np.abs(obs[2]) > 0.3:
        #     done = True
        if np.abs(obs[0]) > 0.3: #or obs[2] > 0.6 or obs[2] < -0.3:
            done = True
        self.t += 1
        return obs, reward, done, {}


if __name__ == '__main__':
    env = PandaChefEnv(render=True)
    env.reset()
    while True:
        env.step(env.action_space.sample())
        time.sleep(0.01)
