
import torch

checkpoint_old_file = 'kitti.pth'
checkpoint_new_file = 'new_kitti.pth'
state_dict = torch.load(checkpoint_old_file)
torch.save(state_dict, checkpoint_new_file, _use_new_zipfile_serialization=False)