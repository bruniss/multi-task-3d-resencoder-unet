from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, BCELoss
from losses.losses import masked_cosine_loss, BCEWithLogitsLossLabelSmoothing, BCEDiceLoss, BCEWithLogitsLossZSmooth

LOSS_FN_MAP = {
    "BCEDiceLoss": BCEDiceLoss,
    "BCEWithLogitsLossLabelSmoothing": BCEWithLogitsLossLabelSmoothing,
    "BCEWithLogitsLossZSmooth": BCEWithLogitsLossZSmooth,
    "BCEWithLogitsLoss": BCEWithLogitsLoss,
    "BCELoss":BCELoss,
    "CrossEntropyLoss": CrossEntropyLoss,
    "MSELoss": MSELoss,
    "masked_cosine_loss": masked_cosine_loss,
}

def get_loss_fn(loss_name):
    return LOSS_FN_MAP[loss_name]
