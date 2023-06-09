# 🚀  Installation
```bash
# clone project
git clone https://github.com/hnagn2003/detectaction
cd detectaction
```
## download example data
Download [here](https://drive.google.com/drive/folders/1L17MLCVUfiJcl-xfXtS3i3arPmwVv-zV)

Then put it to data/kinetics400_tiny
## create conda environment
```bash
conda create -n myenv python=3.9
conda activate myenv
conda install pytorch torchvision -c pytorch
pip install -U openmim
mim install mmengine
mim install mmcv
mim install mmdet  # optional
mim install mmpose  # optional
git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2
pip install -v -e .
cd ..

# install requirements
pip install -r requirements.txt
```

## Project Structure

The directory structure of new project looks like this:

```
│
├── configs                   <- Hydra configs
│   ├── callbacks                <- Callbacks configs
│   ├── data                     <- Data configs
│       ├── recog.yaml
│   ├── debug                    <- Debugging configs
│   ├── experiment               <- Experiment configs
│   ├── extras                   <- Extra utilities configs
│   ├── hparams_search           <- Hyperparameter search configs
│   ├── hydra                    <- Hydra configs
│   ├── local                    <- Local configs
│   ├── logger                   <- Logger configs
│   ├── model                    <- Model configs
│       ├── recog.yaml
│   ├── paths                    <- Project paths configs
│   ├── trainer                  <- Trainer configs
│   │
│   ├── eval.yaml             <- Main config for evaluation
│   └── train.yaml            <- Main config for training
│
├── data                   <- Project data
│   ├── kinetics400_tiny
│
├── logs                   <- Logs generated by hydra and lightning loggers
│
├── notebooks              <- Jupyter notebooks. Naming convention is a number (for ordering),
│                             the creator's initials, and a short `-` delimited description,
│                             e.g. `1.0-jqp-initial-data-exploration.ipynb`.
│
├── scripts                <- Shell scripts
│
├── src                    <- Source code
│   ├── data                     <- Data scripts
│       ├── recog_datamodule.py 
│   ├── models                   <- Model scripts
│       ├── components
│           ├──recognizerCf.py
│       ├── recog_module.py
│   ├── utils                    <- Utility scripts
│   │
│   ├── eval.py                  <- Run evaluation
│   └── train.py                 <- Run training
│
├── tests                  <- Tests of any kind
│
├── .env.example              <- Example of file for storing private environment variables
├── .gitignore                <- List of files ignored by git
├── .pre-commit-config.yaml   <- Configuration of pre-commit hooks for code formatting
├── .project-root             <- File for inferring the position of project root directory
├── environment.yaml          <- File for installing conda environment
├── Makefile                  <- Makefile with commands like `make train` or `make test`
├── pyproject.toml            <- Configuration options for testing and linting
├── requirements.txt          <- File for installing python dependencies
├── setup.py                  <- File for installing project as a package
└── README.md
```

<br>

## Config structure
## Training
```bash

#train with config
python -m src.train trainer=gpu data.num_workers=10 data.batch_size=2 data.pin_memory=true trainer.devices=1  trainer.max_epochs=10

# train with data config
python -m src.train trainer=gpu data.num_workers=10 data.batch_size=2 data.pin_memory=true trainer.devices=1 data.ann_file_train="kinetics_tiny_train_video.txt" data.data_root_train="data/kinetics400_tiny/" data.ann_file_val="kinetics_tiny_val_video.txt" data.data_root_val="data/kinetics400_tiny/"

```


