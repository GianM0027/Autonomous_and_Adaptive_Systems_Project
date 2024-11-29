# Autonomous and Adaptive Systems Course Project

## Introduction
This repository contains the implementation for the Autonomous and Adaptive Systems Course Project course project. The work focuses on the application of Proximal Policy Optimization (PPO) to a selection of tasks from the Procgen Benchmark.  
The Procgen Benchmark, developed by OpenAI, comprises 16 procedurally generated game-like environments designed to test sample efficiency and generalization in reinforcement learning.

## Project Overview
The project aims to replicate and explore the PPO approach, closely following the methodology described by OpenAI. It discusses the implementation details and the rationale behind certain deviations from the original Procgen publication.  
The experiments demonstrate that the agent learns and generalizes well, yielding results that align with those reported in the initial studies.

## Training Results
The results of the training are showcased in the videos below. These videos highlight the performance of the trained agents in two specific Procgen environments: **Coinrun** and **Fruitbot**.

### Coinrun Environment
This video demonstrates the agent successfully navigating through the Coinrun environment after training:
<video src="recordings/Coinrun.mp4" controls width="640" height="360"></video>

### Fruitbot Environment
This video showcases the agent interacting with the Fruitbot environment and achieving competent gameplay:
<video src="recordings/Fruitbot.mp4" controls width="640" height="360"></video>

These videos provide visual evidence of the agent's learning capabilities and its ability to generalize across procedurally generated levels.

## Installation
To run the experiments, you need to install the required dependencies. Execute the following command to install the necessary packages:
```bash
pip install -r requirements.txt
