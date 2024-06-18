# dyad-single-encounters-article

This repository contains the data and code for the article ["Ecological Data Reveal Imbalances in Collision Avoidance Due to Groups' Social Interaction"](https://arxiv.org/abs/2406.06084) by Adrien Gregorj, Zeynep Yücel, Francesco Zanlugo and Takayuki Kanda.

## Installation

To install the required packages, run the following command in the terminal:

```bash
pip install -r requirements.txt
```

The `pedestrians_social_binding` is installed from the source code in the [ATC - DIAMOR - Pedestrians](https://github.com/hbulab/atc-diamor) repository. It will be cloned and installed in the `pedestrians_social_binding` folder.
Refer to the README file in the `pedestrians_social_binding` folder for more information on how to get and prepare the data.

## Usage

You can run the python scripts in the `src` folder to reproduce the results of the article. Some scripts (`00`, `01`, `02`, `03`) compute some intermediate results (stored in pickle format in `data/pickle`). We already provide these files, but you can recompute them by running the scripts.

The `04` script computes the results of the article. It will take a long time to run.

The `05` script generates figures and results for the toy trajectories used in the article.

The `06` script generates figures used in the article to illustrate the different kinds of deviation measures used.
