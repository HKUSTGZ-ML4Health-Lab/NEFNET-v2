import os
import torch
from collections import OrderedDict


class CheckPointer:
    """
    class handling model save and load
    save model state dict, optimizer, scheduler, iteration
    """
    def __init__(self, model, optimizer=None, scheduler=None, save_dir=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir

    def save(self, name, **kwargs):
        if self.save_dir is None:
            return
        save_data = {}
        if self.optimizer is not None:
            save_data['optimizer'] = self.optimizer.state_dict()
        if self.scheduler is not None:
            save_data['scheduler'] = self.scheduler.state_dict()
        save_data['model'] = self.model.state_dict()
        save_data.update(kwargs)

        save_path = os.path.join(self.save_dir, "{}.pkl".format(name))
        print('Saving checkpoint to {}'.format(save_path))
        torch.save(save_data, save_path)

        self.record_last_checkpoint(save_path) ## pay attention

    def load(self, resume_iter=None, best_valid=False):
        """
        load model from specific checkpoint, or last checkpoint or best valid point for test or not load
        :param resume_iter: resume from specific checkpoint
        :param  best_valid: whether to resume from best valid point
        :return:
        """
        if self.has_checkpoint():
            if resume_iter is None or resume_iter == '':
                if best_valid:
                    save_path = os.path.join(self.save_dir, 'best_valid.pkl')
                else:
                    with open(os.path.join(self.save_dir, 'last_checkpoint'), 'r') as f:
                        save_path = f.read().strip()
            # resume from specific checkpoint
            else:
                save_path = resume_iter
            print('Loading model from {}'.format(save_path))
            checkpoint = torch.load(save_path)
            self._load_model_state_dict(checkpoint.pop('model'))
            if 'optimizer' in checkpoint and self.optimizer is not None:
                print('Loading optimizer from {}'.format(save_path))
                self.optimizer.load_state_dict(checkpoint.pop('optimizer'))
            if 'scheduler' in checkpoint and self.scheduler is not None:
                print('Loading scheduler from {}'.format(save_path))
                self.scheduler.load_state_dict(checkpoint.pop('scheduler'))

            # return extra params
            return checkpoint
        else:
            print('No checkpoint, Return model')
            return {}

    def load_nefnet2(self, path = '/home/zhanzhehui_min/Engineering_project/ECG_Panorama/codes/Test1/best_valid.pkl'):
        """
        load model from specific file
         Adapting between models with different architectures by assigning parameters of identical structures
        :return:
        """
        nefnet2_state_dict = torch.load(path)
        loaded_model_state_dict = nefnet2_state_dict['model']
        loaded_model_state_dict = self.strip_prefix_if_present(loaded_model_state_dict, prefix='module.')
        new_state_dict = self.model.state_dict()
        new_state_dict = self.strip_prefix_if_present(new_state_dict, prefix='module.')
        print('loaded from {}'.format(path))
        for key in loaded_model_state_dict:
            if key in new_state_dict:
                new_state_dict[key] = loaded_model_state_dict[key]
        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.load_state_dict(new_state_dict)
        else:
            self.model.load_state_dict(new_state_dict)

    def _load_model_state_dict(self, loaded_model_state_dict):
        # if the state_dict comes from a model that was wrapped in a
        # DataParallel or DistributedDataParallel during serialization,
        # remove the "module" prefix before performing the matching
        loaded_model_state_dict = self.strip_prefix_if_present(loaded_model_state_dict, prefix='module.')

        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.load_state_dict(loaded_model_state_dict)
        else:
            self.model.load_state_dict(loaded_model_state_dict)

    @staticmethod
    def strip_prefix_if_present(state_dict, prefix):
        keys = sorted(state_dict.keys())
        if not all(key.startswith(prefix) for key in keys):
            return state_dict
        stripped_state_dict = OrderedDict()
        for key, value in state_dict.items():
            stripped_state_dict[key.replace(prefix, "")] = value
        return stripped_state_dict

    def record_last_checkpoint(self, last_checkpoint_path):
        with open(os.path.join(self.save_dir, 'last_checkpoint'), 'w') as f:
            f.write(last_checkpoint_path)

    def has_checkpoint(self):
        return os.path.exists(os.path.join(self.save_dir, 'last_checkpoint'))
