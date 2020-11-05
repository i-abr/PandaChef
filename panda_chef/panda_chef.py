import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer




xml_path = 'franka_panda.xml'

model = load_model_from_path(xml_path)
sim = MjSim(model)
viewer = MjViewer(sim)

if __name__=='__main__':

    while True:
        sim.step()
        viewer.render()
