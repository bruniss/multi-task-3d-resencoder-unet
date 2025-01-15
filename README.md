this repository contains training and inference scripts for a dynamically created 3d u-net with residual encoders/decoders with squeeze and excitation blocks. 

the model can learn both segmentation and regression tasks simultaneously, and can accept an arbitrary number of inputs and target labels. currently the only supported data format is zarr. 

there is only 1 model type currently implemented, but adding additional support is trivial and something i will do soon. currently the model only supports a shared encoder path with separate decoder heads/paths. 

___
## purpose
the primary focus of creating this was to enable efficient multi-task learning with 3d unets. i am not a particularly skilled or experience programmer nor am i a machine learning expert by any stretch of the imagination, no guarantees for performance!

### main features
**self configuring encoder/decoder paths**: provided a config with inputs and channels, the model will automatically create the required paths and select some defaults for loss

**automatic patch finding**: provided with a reference labeled zarr array, the dataset will find patches that meet the requirements for both volumetric size within the patch and label density within it. this allows for passing gigantic but relatively sparsely labeled datasets and only performing training on the areas which are densely labeled. this list is then saved to a json so you will not have to run this computation again, provided your patch size has not changed.

**surface normal rotations/flips**: includes augmentations to apply proper rotation and flipping for surface normals along with other targets

**a clear dataset and train path**: all data is stored in dictionaries where the key is the name of your provided target. this makes extending or modifying relatively easy, and even someone without much python experience should be able to follow what is happening in each stage.

___
## configuration
the primary method for configuring the model is through a yaml file provided as an argument. an example config is in the tasks folder. the config has the following parameters (do not use the type hinting, its only there to let you know what to provide):

most of these fields are optional. this is an example yaml file 

```yaml
# YOU DO NOT HAVE TO PROVIDE VALUES IN MOST OF THESE FIELDS

tr_params:
  model_name: ink             # required [str] this is the name your model will checkpoint will be saved as
  vram_max: 22000             # required if autoconfigure is true, else optional-- the amount in MB you want the model to use
  autoconfigure: false        # optional, true -- if true, the network will attempt to set some reasonable defaults based on your vram_max
  patch_size: [14, 128, 128]  # optional [list] patch size for training
  batch_size: 12              # optional [int] batch size for training
  tr_val_split: 0.90          # optional[float] the percentage of your total dataset used for training, with the other part being used for val
  ignore_label: null          # [NOT YET IMPLEMENTED] if you have an ignore label, you can set it here and loss will not be computed against it
  loss_only_on_label: false   # [NOT YET IMPLEMENTED]
  dilate_label: false         # optional, false [bool] if true, will apply a small dilation to your labels using a 3x3 spherical kernel -- will skip tasks named "normals"
  initial_lr: 0.001           # optional initial learning rate
  weight_decay: 0.0           # optional, 0.0001 [float]
  max_steps_per_epoch: 500    # optional, 500 [int] the number of batches seen by the model each epoch
  max_val_steps_per_epoch: 25 # optional, 25 [int] the number of batches seen in validation each epoch
  max_epoch: 500              # optional, 500 [int] the maximum number of epochs to train for
  gradient_accumulation: 1    # optional, 1 [int] if 1, no accumulation, if >1 will accumulate this many 'batches' each batch to similate larger batch size.
  load_weights_only: false    # optional, false [bool] if true, will not load the optimizer, scheduler, or epoch state from the model -- set true to fine-tune
  ckpt_out_base: "/mnt/raid_hdd/models/ink" # [str] the path the model checkpoint is saved to
  # checkpoint_path: "/mnt/raid_hdd/models/ink/ink_157.pth" # optional, None [str] if provided, will load the provided checkpoint and begin training from there
  optimizer: "AdamW" # optional, AdamW [str] the optimizer to use during training. currently only AdamW and SGD are provided.
  num_dataloader_workers: 12 # optional, 4 [int]
  tensorboard_log_dir: "/home/sean/Desktop/tensorboard/ink" # [str] the path the tensorboard logs will be stored to

model_config:

  # set the feature maps and the number of levels you would like in your network -- the number of feature maps
  # must be equal to the number of levels. note that the feature maps extend through the entire encoder/decoder
  # stage, and so will add a significant amount of memory overhead as these numbers get larger. typically
  # feature maps double each layer, but you can put whatever numbers you want here.

  f_maps: [64, 128, 256, 512, 768] # optional, [list] or [int] -- if an int is provided will automatically double each layer

  # you can set the basic block from the following list:
  #     DoubleConv, ResNetBlock, ResNetBlockSE, nnUNetStyleResNetBlockSE
  # if you specify a squeeze and excitation block, you can provide a se_module , or it will fall back to scse
  basic_module: "nnUNetStyleResNetBlockSE" # optional

  # valid options here are:
  #    1. cse - `Channel Squeeze and Excitation <https://arxiv.org/abs/1709.01507>`_
  #    2. sse - `Spatial Squeeze and Excitation <https://arxiv.org/abs/1803.02579>`_
  #    3. scse - `Channel and Spatial Squeeze and Excitation <https://arxiv.org/abs/1803.02579>`_
  se_module: 'sse' # optional [str]

  # these two control how much your data is downsampled/upsampled in the network. if 2 is set, the size is reduced by half each layer in the encoder
  # this means that if you set your f_maps to have 5 layers, and your z_axis can't be divided by two 5 times, you'll get an error, you can set this
  # here to 1, and not downsample at all in that dimension
  pool_kernel_size: [1, 2, 2] # optional, 2 [tuple] or [int] , the amount of downsampling done per level in encoder. the decoder will use the same config , (z, y, x)

dataset_config:
  # the dataset config is where we build the targets data and the labeled data. you can use any number of inputs and labels.
  # the dataset is configured to search a zarr volume of labels for valid patches. we consider a valid patch to be one that meets the
  # criteria set by min_bbox_percent and min_labeled_ratio. the z, y, x starting positions of these are saved to a json so we only
  # have to compute this once. on large volumes, this can take a long time.
  # ref label:
      # used for the valid patch finding. the name of these refers to the volume within volume paths. in the below example
      # "ink" is referring to the ink volume, defined one line above. you can provide any number of volume paths.

  min_bbox_percent: 0.20 # optional, 0.97 [float] a percentage of the patch size that must be encompassed by a bbox containing all the labels in the patch
  min_labeled_ratio: 0.20 # optional, 0.15 [float] a percentage of the above bbox that must contain labeled data (ie: the density of the labels)
  use_cache: true # optional, True [bool] whether to store a patch position cache. strongly recommended
  cache_file: "/home/sean/Desktop/s1_segments/14_128_128_inkcache_2.json" # optional, './' the location to store the cache
  volume_paths: # these are your input volumes. each must have input volume path, target volume path (from your targets defined in targets),
    - input: "/home/sean/Desktop/s1_segments/0901.zarr/layers.zarr"
      ink: "/home/sean/Desktop/s1_segments/0901.zarr/inklabels.zarr"
      ref_label: "ink"

    - input: "/home/sean/Desktop/s1_segments/0926.zarr/layers.zarr"
      ink: "/home/sean/Desktop/s1_segments/0926.zarr/inklabels.zarr"
      ref_label: "ink"

    - input: "/home/sean/Desktop/s1_segments/1321.zarr/layers.zarr"
      ink: "/home/sean/Desktop/s1_segments/1321.zarr/inklabels.zarr"
      ref_label: "ink"

    - input: "/home/sean/Desktop/s1_segments/1619.zarr/layers.zarr"
      ink: "/home/sean/Desktop/s1_segments/1619.zarr/inklabels.zarr"
      ref_label: "ink"

    - input: "/home/sean/Desktop/s1_segments/3336.zarr/layers.zarr"
      ink: "/home/sean/Desktop/s1_segments/3336.zarr/inklabels.zarr"
      ref_label: "ink"

    - input: "/home/sean/Desktop/s1_segments/4423.zarr/layers.zarr"
      ink: "/home/sean/Desktop/s1_segments/4423.zarr/inklabels.zarr"
      ref_label: "ink"

    - input: "/home/sean/Desktop/s1_segments/5753.zarr/layers.zarr"
      ink: "/home/sean/Desktop/s1_segments/5753.zarr/inklabels.zarr"
      ref_label: "ink"

    - input: "/home/sean/Desktop/s1_segments/51002.zarr/layers.zarr"
      ink: "/home/sean/Desktop/s1_segments/51002.zarr/inklabels.zarr"
      ref_label: "ink"

  targets:
    # the targets provided here are how the model is configured. for each entry here, the model will construct an additional
    # decoder path, with the specified number of channels.
    ink:
      channels: 1            # required [int] the number of channels in your input -- your input should be in shape z, y, x, c
      activation: "none"     # optional, none [str] the activation type you would like your model to perform during training, options are: Sigmoid , Softmax, None.
                             # be careful you do not introduce a double activation if your loss function also applies one

      weight: 1              # optional, 1 [float] the weight applied to the task, as a percentage. this is multiplied by the loss value during training, so 1 is 100%
                             # if you have multiple tasks, these do not need to equal 1. this is the default. ex: weight: 0.75

      loss_fn: "BCEDiceLoss" # optional, 'BCEDiceLoss' [str] the loss you would like to use from the loss_fn_map in train.py. to add losses, simply add to the mapping in train.py
      loss_kwargs:           # optional, none - any keyword arguments you would like to pass to your loss function, each on its own line
        alpha: 0.5
        beta: 0.5

inference_params:
  patch_size: [14, 256, 256] # optional - patch size to use for inference. this can be different than what you trained with, but might hurt performance
  batch_size: 8              # optional - this typically can be larger for inference than training
  checkpoint_path: "/mnt/raid_hdd/models/ink/ink_352.pth" # the checkpoint to load
  num_dataloader_workers: 12
  input_path: "/home/sean/Desktop/s1_segments/3852/3852.zarr/layers.zarr" # [str] input volume, CURRENTLY MUST BE ZARR
  input_format: "zarr" # not currently implemented, all data is saved to zarr in the output dir as predictions.zarr
  output_dir: "/home/sean/Desktop/s5_segments/20241030152031"
  output_format: "zarr" # currently does nothing , can only be zarr
  output_type: "np.uint8" # currently does nothing
  output_targets: ["ink"] # required
  load_all: true # optional, false - if true, will load entire input_path volume into memory
  overlap: 0.1   # optional, .01 - the percentage of overlap each patch should have -- this increases your total patch count, so be careful to not go too high

  targets:   # the targets for inference must match a target that will exist in the dictionary your model outputs , and the channels must be equal
    ink:     # to the number of channels your model outputs
      channels: 1
      activation: "none"
      weight: 1
```
