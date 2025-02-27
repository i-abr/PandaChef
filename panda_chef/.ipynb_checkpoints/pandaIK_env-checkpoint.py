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
    def __init__(self, render=False, time_step = 1/200., frame_skip=1):
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

        self.action_scale = (np.array([0.05, 0., 0.05]), np.array([0., 0.7, 0.]))
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
        draw_coordinate(self.robot_id, parentLinkIndex=pandaEndEffectorIndex)

        # TODO: create coordinate for the target where you want the pancake to end up


        self.reset()

        rew, obs = self.get_rew_obs()
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
        rew, obs = self.get_rew_obs()
        return obs

    def get_rew_obs(self):
        pan_full_state  = bullet_client.getLinkState(self.robot_id, pandaEndEffectorIndex, computeLinkVelocity=1)
        cake_pose       = bullet_client.getBasePositionAndOrientation(self.pancake_id)
        cake_full_vel   = bullet_client.getBaseVelocity(self.pancake_id)

        pan_position    = np.array(pan_full_state[4])
        pan_orientation = np.array(pan_full_state[5])
        pan_linear_velocity  = np.array(pan_full_state[6])
        pan_angular_velocity = np.array(pan_full_state[7])

        cake_position    = np.array(cake_pose[0])
        cake_orientation = np.array(cake_pose[1])
        cake_linear_velocity  = np.array(cake_full_vel[0])
        cake_angular_velocity = np.array(cake_full_vel[1])

        # TODO: unsure if any benefit if velocities can be taken relative to each other
        obs = np.concatenate([pan_position-cake_position,
                                pan_orientation-cake_orientation,
                                pan_linear_velocity, cake_linear_velocity,
                                pan_angular_velocity, cake_angular_velocity])

        catch_cost       = np.sum((pan_position - cake_position)**2) + np.sum((pan_orientation-cake_orientation)**2)
        pan_vel_penalty  = np.sum((pan_linear_velocity)**2) + np.sum((pan_angular_velocity)**2)
        cake_vel_penalty = np.sum((cake_linear_velocity)**2) + np.sum((cake_angular_velocity)**2)
        rew = -catch_cost -1e-3 * pan_vel_penalty -1e-4*cake_vel_penalty
        return rew, obs

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
                self.robot_id, i, bullet_client.POSITION_CONTROL, jointPoses[i], force=40.)
        # step the simulation forward
        for _ in range(self._frame_skip):
            bullet_client.stepSimulation()

        rew, obs = self.get_rew_obs()
        done = False
        # if obs[2]<-0.05 or np.abs(obs[0])>0.5 or np.abs(obs[2]) > 0.3:
        #     done = True
        if np.abs(obs[0]) > 0.3: #or obs[2] > 0.6 or obs[2] < -0.3:
            done = False
        self.t += 1
        return obs, rew, done, {}


if __name__ == '__main__':
    env = PandaChefEnv(render=True)
    env.reset()
    while True:
        env.step(env.action_space.sample())
        time.sleep(0.01)
