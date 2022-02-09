## Supervision meets Self-supervision: A Deep Multitask Network for Colorectal Cancer Histopathological Analysis, MISP 2022 

Aritra Marik, **Soumitri Chattopadhyay** and Pawan Kumar Singh, **“Supervision meets Self-supervision: A Deep Multitask Network for Colorectal Cancer Histopathological Analysis”**, _Accepted for Oral Presentation at_ **Intl. Conf. on Machine Intelligence and Signal Processing (MISP), 2022.**

## Requirements
To install the required dependencies run the following in command prompt:
`pip install -r requirements.txt`

## Running the codes:
Required directory structure:

```
+-- data
|   +-- .
|   +-- train
|   +-- val
+-- dataset.py
+-- main.py
+-- model.py
+-- Network_modules.py
+-- requirements.txt
```
Then, run the code using the command prompt as follows:

`python main.py --data_dir "./path/to/data"`

**Hyperparameters (available as arguments):**
- `--num_epochs` : number of training epochs. Default = 200
- `--learning_rate` : learning rate for training. Default = 0.01
- `--batchsize` : batch size for training. Default = 32
- `--margin` : triplet loss margin value > 0. Default = 0.2
- `--lambd` : loss balancing factor. Default = 10

<!-- ## Citation
If you find this article useful in your research, please consider citing:
```
@InProceedings{marik2022supervision,
author = {Aritra Marik and Soumitri Chattopadhyay and Pawan Kumar Singh},
title = {Supervision meets Self-supervision: A Deep Multitask Network for Colorectal Cancer Histopathological Analysis},
booktitle = {International Conference on Machine Intelligence and Signal Processing (MISP)},
year = {2022}
} -->
