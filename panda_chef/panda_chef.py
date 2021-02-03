import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer




xml_path = 'simple_flipper.xml'

model = load_model_from_path(xml_path)
sim = MjSim(model)
viewer = MjViewer(sim)

act_mid = np.mean(model.actuator_ctrlrange, axis=1)
act_rng = 0.5 * (model.actuator_ctrlrange[:,1]-model.actuator_ctrlrange[:,0])


if __name__=='__main__':

    while True:
        sim.step()
        viewer.render()
