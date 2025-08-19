# Autonomous and Adaptive Systems Course Project

## Introduction
This repository contains the implementation for the Autonomous and Adaptive Systems Course Project course project. The work focuses on the application of Proximal Policy Optimization (PPO) to a selection of tasks from the Procgen Benchmark.  
The Procgen Benchmark, developed by OpenAI, comprises 16 procedurally generated game-like environments designed to test sample efficiency and generalization in reinforcement learning.

## Project Overview
The project aims to replicate and explore the PPO approach, closely following the methodology described by OpenAI. It discusses the implementation details and the rationale behind certain deviations from the original Procgen publication.  
The experiments demonstrate that the agent learns and generalizes well, yielding results that align with those reported in the initial studies.

## Training Results
I evaluated the training method on two Procgen environments: Coinrun and Fruitbot.

The agents successfully learn to complete levels with high accuracy. The plots below report the average sum of rewards per episode (red) and the average episode length (blue).

<img width="1990" height="1050" alt="plots" src="https://github.com/user-attachments/assets/ca735b71-5b36-4fa6-aa54-1bc540da4c9c" />


The following videos present the trained agents playing in previously unseen test environments:

<div align="center"> <table> <tr> <td align="center"><b>Coinrun</b></td> <td align="center"><b>Fruitbot</b></td> </tr> <tr> <td align="center"> <img src="https://github.com/user-attachments/assets/779870fe-5ca2-4000-ab4a-d1fb7af1fd39" /> </td> <td align="center"> <img src="https://github.com/user-attachments/assets/c10263ce-ae24-4796-ad03-9f9605b8e553" /> </td> </tr> </table> </div>





## Installation
To run the experiments, you need to install the required dependencies. Execute the following command to install the necessary packages:
```bash
pip install -r requirements.txt
```

Once the requirements are installed, the training can be performed by running `main_notebook.ipynb`.
