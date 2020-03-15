import argparse
import torch

parser = argparse.ArgumentParser(description='Set hyperparameters')
parser.add_argument('--lr', default=0.0001, help='Learning Rate')
parser.add_argument('--gamma', default=0.99, help='TODO')
parser.add_argument('--tau', default=1, help='TODO')
parser.add_argument('--seed', default=1, help='TODO')
parser.add_argument('--num_processes', default=16, help='TODO')
parser.add_argument('--num_steps', default=20, help='TODO')
parser.add_argument('--max_epoch', default=10000, help='TODO')
parser.add_argument('--env', default='ENV_NAME', help='TODO')

if __name__ == "__main__":

    # Set hyperparameters
    params = parser.parse_args()

    # Set seed
    torch.manual_seed(params.seed)

    # Create environment