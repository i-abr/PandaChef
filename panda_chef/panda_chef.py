import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer




xml_path = 'franka_panda.xml'

model = load_model_from_path(xml_path)
sim = MjSim(model)
viewer = MjViewer(sim)

act_mid = np.mean(model.actuator_ctrlrange, axis=1)
act_rng = 0.5 * (model.actuator_ctrlrange[:,1]-model.actuator_ctrlrange[:,0])

sim.data.qpos[0] = act_mid[0]
sim.data.qpos[1] = act_mid[1]
sim.data.qpos[2] = act_mid[2]

sim.data.ctrl[0] = act_mid[0]
sim.data.ctrl[1] = act_mid[1]
sim.data.ctrl[2] = act_mid[2]

sim.forward()
if __name__=='__main__':

    while True:
        sim.step()
        viewer.render()
