# INM707 Deep Learning

This repo contains various different approaches to train an agent to play the [Atari 2600 Boxing](https://en.wikipedia.org/wiki/Boxing_(Atari_2600)) game.

Models considered here are a Vanilla DQN using Replay Memory and MSE temporal difference loss, and a Duelling DQN using Prioritised Replay Memory, Huber loss, and a separate "target network".

## Installation

This project requires [OpenAI Baselines](https://github.com/openai/baselines). Follow installation steps in linked project - can skip installation of tensorflow as this is already included in the project dependencies.

## boxing-dqn

This contains notebooks and dependencies for the two DQN models trained for the project report

## scripts

This contains various scripts and notebooks written as part of our learning for the project, which are not relevant for the final writeup.