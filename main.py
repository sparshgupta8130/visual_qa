import torch
import warnings
from cfg import *
from models import *
from eval import test
from load_data import get_embeds


def main():
    data_dir = config['data_dir']
    dataset = config['dataset']

    # TODO: Initialize Dataloader

    if config['train'] is True:
        pass
        # TODO: Initialize Model and Model config

        # TODO: Print Model and Number of Parameters

        # TODO: Call train from trainer
    else:
        pass
        # TODO: Load model for testing

        # TODO: Print Model and Number of Parameters

        # TODO: Get the test results


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
