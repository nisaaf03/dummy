from stable_baselines import GAIL
from stable_baselines.gail import ExpertDataset
' ############################ CEK ENVIRONMENTNYA YANG DIPAKEEEEEEEEEEEEEE ############################# '
from env_1joint_norew_targetasobs_v2_5 import ExcaRobo 

SIM_ON = 0
'# Load the expert dataset'
dataset = ExpertDataset(expert_path='env_1joint_norew_targetasobs_v25_16jul_5eps.npz', traj_limitation=-1, verbose=1)
env = ExcaRobo(SIM_ON)
### TRAINING MODEL ###
model = GAIL('MlpPolicy', env, dataset, verbose=1)
# # Note: in practice, you need to train for 1M steps to have a working policy
if __name__ == '__main__': 
#   # freeze_support()
  model.learn(total_timesteps=100000)
  model.save("env_1joint_norew_targetasobs_v25_16jul_5eps_100rb") #### JANGANLUPA GANTI NAMA MODEL ###1joint_targetasobs_randomtarget_30k
  # del model # remove to demonstrate saving and loading
