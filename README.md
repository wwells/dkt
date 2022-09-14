# Deep Knowledge Tracing Implementation

This repo borrows heavily from two implementations of the DeepKnowledgeTracing Model in Python
* https://github.com/lccasagrande/Deep-Knowledge-Tracing
* https://github.com/mmkhajah/dkt

The following papers were used as primary inspiration:

* http://stanford.edu/~cpiech/bio/papers/deepKnowledgeTracing.pdf
* https://home.cs.colorado.edu/~mozer/Research/Selected%20Publications/reprints/KhajahLindseyMozer2016.pdf


TODO:
* try and run in separate / distributed machines
* generate a transform pipeline that we can use in front of a predict function
* expand the read/load/transform data functions so we can use a much larger distributed dataset
* configure GPU usage

# Data Format

We generally used the data format found in [mmkhajah's implementation](https://github.com/mmkhajah/dkt) which is a space separated text file
where the first column is the student id, the second is the skills_id, and the third is a boolean indicating whether the student got the question right.

Where we deviate from that repo is there is no split file.   We handle the split programmatically.

There is a smaller toy dataset that is a subset of the larger assistments.txt available that only has 5k observations to aid in rapid testing.

# Usage

```sh
$ python3 run_dkt.py --help
usage: Khan DKT Prototype [-h] [--dataset DATASET] [--lstm_units LSTM_UNITS] [--dropout_rate DROPOUT_RATE] [--epochs EPOCHS] [--test_split TEST_SPLIT]
                          [--val_split VAL_SPLIT]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     the path to the data on disk

Model arguments.:
  --lstm_units LSTM_UNITS
                        number of units of the LSTM layer.
  --dropout_rate DROPOUT_RATE
                        fraction of the units to drop.

Training arguments.:
  --epochs EPOCHS       number of epochs to train.
  --test_split TEST_SPLIT
                        fraction of data to be used for testing (0, 1).
  --val_split VAL_SPLIT
                        fraction of data to be used for validation (0, 1).
```
# Setup

Create and activate a virtual env with python 3.8.   Pip install the requirements.

> NOTE:   You'll want to make sure you have python 3.8 ready to use.

```
python3.8 -m venv ~/.venv/dkt
source ~/.venv/dkt/bin/activate
pip3 install .
```

## Managing Python Libraries

Libraries are pinned using `pip-compile`.   If you need to increment any libraries you can:
* change them in the requirements.in
* run `pip-compile` after installing `pip-tools`

Then when you run `pip3 install .` the setup.py for khan-deepkt reads the generated requirements.txt

## Google Colab Use

Commands for use in a collab notebook cell are as below:

Install python 3.8
```
# Setup python 3.8

!sudo apt-get update -y
!sudo apt-get install python3.8

#change alternatives
!sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1
!sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 2

#check python version
!python --version
```

Mount Gdrive
```
from google.colab import drive
drive.mount('/content/gdrive')
```

Clone this repo into the gdrive
```
%cd gdrive/My Drive/
! git clone https://github.com/wwells/dkt.git
```

Install python requirements
```
%cd dkt
!ls
!pip install -r requirements.txt
```

# Datasets

We have included the skill builder version of the [Assistments 2009-2010 dataset](https://sites.google.com/site/assistmentsdata/home/assistment-2009-2010-data/skill-builder-data-2009-2010) which is one of the datasets evaluated in the paper.