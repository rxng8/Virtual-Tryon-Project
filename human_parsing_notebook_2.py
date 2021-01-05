# %%

"""
This notebook is for the atr dataset training
"""

from PIL import Image

from core.dataset import ATRDataset
from core.utils import *
from core.models import get_unet_model_for_human_parsing

# Config
IMG_SHAPE = (256, 192, 3)
BATCH_SIZE = 16
OUTPUT_CHANNELS = len(ATRDataset.LABEL2INT)

# %%

base = "./dataset/ICCV15_fashion_dataset(ATR)/humanparsing"
ds = ATRDataset(base)
steps_per_epoch = ds.steps_per_epoch

# %%

model = get_unet_model_for_human_parsing(IMG_SHAPE, OUTPUT_CHANNELS)

# %%

train_ds = ds.get_tf_train_batch_dataset()
test_ds = ds.get_tf_test_batch_dataset()


# %%

history = model.fit(
    train_ds,
    steps_per_epoch=steps_per_epoch, 
    epochs=1,
    validation_data=test_ds,
    validation_steps=10
)

# %%
