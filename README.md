# Cross-sensor nighttime lights image calibration for DMSP/OLS and SNPP/VIIRS with Residual U-net

By Dmitry Nechaev<sup>1</sup>, Mikhail Zhizhin<sup>2,3,*</sup>, Alexey Poyda<sup>1, 4</sup>, Tilottama Ghosh<sup>2</sup>, Feng-Chi Hsu<sup>2</sup> and Christopher Elvidge<sup>2</sup>

<sup>1</sup>	Moscow Institute of Physics and Technology; nechaev.dv@phystech.edu

<sup>2</sup>	Earth Observation Group, Payne Institute, Colorado School of Mines; mzhizhin@mines.edu

<sup>3</sup>	Space Research Institute, Russian Academy of Sciences

<sup>4</sup>	NRC “Kurchatov Institute”; poyda_aa@nrcki.ru

<sup>*</sup>	Correspondence: mzhizhin@mines.edu

### Table of Contents

0. [Introduction](#Introduction)
1. [Usage](##Usage)
2. [Citation](##Citation)

### Introduction

This repository contains all the data, scripts, research notebooks, plots, weights, etc. for the "Cross-sensor nighttime lights image calibration for DMSP/OLS and SNPP/VIIRS with Residual U-net" paper. Master branch has the script that can be used for the generation of DMSP-like imagery from VIIRS imagery. The notebooks branch has all the research related data.

### Usage
This script currently supports .npz and .envi formats!

Check the dependencies in the requirements.txt, you need to install tensorflow-addons.

Run "python generation.py --help" to check how to use it.

In case of any questions contact me on nechaev.dv@phystech.edu

### Citation
If you use the data or model in your research, please cite:

Nechaev, D.; Zhizhin, M.; Poyda, A.; Ghosh, T.; Hsu, F.-C.; Elvidge, C. Cross-Sensor Nighttime Lights Image Calibration for DMSP/OLS and SNPP/VIIRS with Residual U-Net. Remote Sens. **2021**, 13, 5026. https://doi.org/10.3390/rs13245026


