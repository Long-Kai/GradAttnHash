# Gradient Attention Hashing

This is the source code for paper "Accelerate Learning of Deep HashingWith Gradient Attention" (ICCV 2019)

The code is implemented on PyTorch Version 0.4.1

## Citation
If you use this code for your research, please consider cite the paper:

    
    @inproceedings{huang2019accelerate,
      title={Accelerate Learning of Deep Hashing With Gradient Attention},
      author={Huang, Long-Kai and Chen, Jianda and Pan, Sinno Jialin},
      booktitle={Proceedings of the IEEE International Conference on Computer Vision},
      pages={5271--5280},
      year={2019}
    }
    
      
## Datasets

The datasets used in experiments are CIFAR-10, Imagenet and NUSWIDE-81. 
To run the code, it needs to download the images of the datasets and decompress them into corresponding folds in [./data](/data).

## Training

Run [./run_training.py](/run_training.py) to start training.

The training parameters can be configured in [./run_training.py](/run_training.py) 

To save the model for evalution, change  `tr_config["save_model"] = False` to `True`.


