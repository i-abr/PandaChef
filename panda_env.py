import time
import numpy as np
import math

useNullSpace = 1
ikSolver = 0
pandaEndEffectorIndex = 12 #8
pandaNumDofs = 7

ul = np.array([0.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
#upper limits for null space (todo: set them to proper range)
ll = np.array([-0.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
#joint ranges for null space (todo: set them to proper range)
jr = ul-ll
#restposes for null space
jointPositions=[0., 0.258, 0., -2.24, 0., 2.66, 0., 0.02, 0.02]
rp = jointPositions

class PandaSim(object):
    def __init__(self, bullet_client):
        self.t = 0
        self.bullet_client = bullet_client
        #print("offset=",offset)
        flags = self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        # legos=[]
        # self.bullet_client.loadURDF("tray/traybox.urdf", [0+offset[0], 0+offset[1], -0.6+offset[2]], [-0.5, -0.5, -0.5, 0.5], flags=flags)
        # legos.append(self.bullet_client.loadURDF("lego/lego.urdf",np.array([0.1, 0.3, -0.5])+self.offset, flags=flags))
        # legos.append(self.bullet_client.loadURDF("lego/lego.urdf",np.array([-0.1, 0.3, -0.5])+self.offset, flags=flags))
        # legos.append(self.bullet_client.loadURDF("lego/lego.urdf",np.array([0.1, 0.3, -0.7])+self.offset, flags=flags))
        self.bullet_client.loadURDF("plane.urdf", np.array([0.,0.,0.]), flags=flags, useFixedBase=True)
        # sphereId = self.bullet_client.loadURDF("sphere_small.urdf",np.array( [0, 0.3, -0.6])+self.offset, flags=flags)
        # self.bullet_client.loadURDF("sphere_small.urdf",np.array( [0, 0.3, -0.5])+self.offse/t, flags=flags)
        # self.bullet_client.loadURDF("sphere_small.urdf",np.array( [0, 0.3, -0.7])+self.offset, flags=flags)
        # orn=[-0.707107, 0.0, 0.0, 0.707107]#p.getQuaternionFromEuler([-math.pi/2,math.pi/2,0])
        # eul = self.bullet_client.getEulerFromQuaternion([-0.5, -0.5, -0.5, 0.5])
        self.robot_id = self.bullet_client.loadURDF("./franka_panda_chef/panda_chef.urdf", np.array([0,0,0]), useFixedBase=True, flags=flags)
        index = 0
        for j in range(self.bullet_client.getNumJoints(self.robot_id)):
            self.bullet_client.changeDynamics(self.robot_id, j, linearDamping=0, angularDamping=0.1)
            info = self.bullet_client.getJointInfo(self.robot_id, j)
            jointName = info[1]
            jointType = info[2]
            if (jointType == self.bullet_client.JOINT_PRISMATIC):
                self.bullet_client.resetJointState(self.robot_id, j, jointPositions[index])
                index=index+1
            if (jointType == self.bullet_client.JOINT_REVOLUTE):
                self.bullet_client.resetJointState(self.robot_id, j, jointPositions[index])
                index=index+1
        self.__prevPose = self.bullet_client.getLinkState(self.robot_id, pandaEndEffectorIndex)
    def reset(self):
        pass

    def step(self):
        t = self.t
        self.t += 1./60.
        pos = [0.6 + 0.2 * math.sin(1.5 * t),   0.,  0.24]
        th = 0.2 * math.sin(2.5*t)
        # orn = self.bullet_client.getQuaternionFromEuler([math.pi,th,0.])
        orn = self.bullet_client.getQuaternionFromEuler([0.,th,math.pi])
        jointPoses = self.bullet_client.calculateInverseKinematics(self.robot_id,
                        pandaEndEffectorIndex, targetPosition=pos, targetOrientation=orn,
                        lowerLimits=ll,
                        upperLimits=ul,
                        jointRanges=jr,
                        restPoses=rp, maxNumIterations=5)
        for i in range(pandaNumDofs):
            self.bullet_client.setJointMotorControl2(
                    self.robot_id, i, self.bullet_client.POSITION_CONTROL, jointPoses[i], force=5 * 240.)
        current_pos = self.bullet_client.getLinkState(self.robot_id, pandaEndEffectorIndex)
        # self.bullet_client.addUserDebugLine(self.__prevPose[4], current_pos[4], lineColorRGB=[0,1,0], lifeTime=1)
        # self.__prevPose = current_pos
