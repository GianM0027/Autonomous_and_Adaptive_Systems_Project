# Autonomous and Adaptive Systems Course Project

## Introduction
This repository contains the implementation for the Autonomous and Adaptive Systems Course Project course project. The work focuses on the application of Proximal Policy Optimization (PPO) to a selection of tasks from the Procgen Benchmark. 
The Procgen Benchmark, developed by OpenAI, comprises 16 procedurally generated game-like environments designed to test sample efficiency and generalization in reinforcement learning.

## Project Overview
The project aims to replicate and explore the PPO approach, closely following the methodology described by OpenAI. It discusses the implementation details and the rationale behind certain deviations from the original Procgen publication. 
The experiments demonstrate that the agent learns and generalizes well, yielding results that align with those reported in the initial studies.

## Installation
To run the experiments, you need to install the required dependencies. Execute the following command to install the necessary packages:
```bash
pip install -r requirements.txt
```

This step ensures all dependencies are correctly installed and avoids potential conflicts.

## Repository Contents
- **Notebooks**: This directory contains Jupyter notebooks that document the training experiments.
- **Test Agent**: This directory includes scripts to test the trained models using different configurations. Adjust the `GAME` parameter within the script to test different games. The trained models are saved in the `weights` folder.
