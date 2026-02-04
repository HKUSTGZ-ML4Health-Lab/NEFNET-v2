from .Tianchi import EcgTianChiInterval,EcgTianChi_finetune
from .CPSC2018 import CPSC2018, CPSC2018_finetune,CPSC2018_c
from .ChinaDB import ChinaDB, ChinaDB_finetune
from .PTBXL import PTBVXL, PTBVXL_finetune
from .Panobench import Panobench
from torch.utils.data import ConcatDataset

def build_dataset(cfg, phase):
    if cfg.DATA.dataset == 'Tianchi':
        return EcgTianChiInterval(cfg, phase)
    elif cfg.DATA.dataset == 'CPSC2018':
        return CPSC2018(cfg, phase)
    elif cfg.DATA.dataset == 'ChinaDB':
        return ChinaDB(cfg, phase)
    elif cfg.DATA.dataset == 'PTBXL':
        return PTBVXL(cfg, phase)
    elif cfg.DATA.dataset == 'Panobench':
        return Panobench(cfg,phase)
    elif cfg.DATA.dataset == 'all':
        chinadb_dataset = ChinaDB(cfg, phase=phase)
        cpsc2018_dataset = CPSC2018(cfg, phase=phase)
        tianchi_dataset = EcgTianChiInterval(cfg, phase=phase)
        ptbxl_dataset = PTBVXL(cfg, phase=phase)
        all_dataset = ConcatDataset([chinadb_dataset, tianchi_dataset, cpsc2018_dataset, ptbxl_dataset]) # incart_dataset, ptb_dataset, code_dataset
        return all_dataset
    else:
        raise NotImplementedError("{} is not support".format(cfg.DATA.dataset))

