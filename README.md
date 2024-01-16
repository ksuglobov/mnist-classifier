# mnist-classifier

mnist-classifier is a simple MNIST image classifier using the LeNet-5 neural
network.

## Install environment and run pre-commit

```bash
mkdir test_env
virtualenv test_env
source ./test_env/bin/activate

poetry install

pre-commit install

pre-commit run -a
```

# Usage

There are two modes of operation of the program: train and infer - for training
and inference accordingly. The entry point to the program is `commands.py`.

Help for all program:

```bash
python3 commands.py -h
```

Output:

```
Simple MNIST image classifier using the LeNet-5 neural network

positional arguments:
  {train,infer}  Mode of operation
    train        Run training
    infer        Run inference
```

## Train mode

Help for train mode:

```bash
python3 commands.py train -h
```

Output:

```
usage: commands.py train [-h] [--log_level {debug,info,warning}] [--path2model_dir PATH2MODEL_DIR]
                         [--n_epochs N_EPOCHS]

options:
  -h, --help            show this help message and exit
  --log_level {debug,info,warning}
                        Logging level (default: info)
  --path2model_dir PATH2MODEL_DIR
                        Path to directory for saving model
  --n_epochs N_EPOCHS   Number of epochs (default: 5)
```

## Infer mode

Help for infer mode:

```bash
python3 commands.py infer -h
```

Output:

```
usage: commands.py infer [-h] [--log_level {debug,info,warning}] [--path2model PATH2MODEL]
                         [--path2output_dir PATH2OUTPUT_DIR]

options:
  -h, --help            show this help message and exit
  --log_level {debug,info,warning}
                        Logging level (default: info)
  --path2model PATH2MODEL
                        Path to saved model
  --path2output_dir PATH2OUTPUT_DIR
                        Path to directory for output file
```
