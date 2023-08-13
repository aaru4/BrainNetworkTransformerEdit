from source.models.transformer import GraphTransformer
from source.models.brainnetcnn import BrainNetCNN
from source.models.fbnetgen import FBNETGEN
from source.models.BNT import BrainNetworkTransformer
from omegaconf import DictConfig


def model_factory(config: DictConfig):
    if config.model.name in ["LogisticRegression", "SVC"]:
        return None
    return eval(config.model.name)(config).cuda()
