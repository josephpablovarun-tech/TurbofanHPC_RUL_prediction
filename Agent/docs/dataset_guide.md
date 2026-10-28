---
doc_id: DG
title: FD001 Dataset Guide
revision: "1.0"
---

## Overview of FD001
FD001 is one subset of the NASA C-MAPSS turbofan engine degradation simulation data. It contains 100 training trajectories that run until failure and 100 test trajectories that stop some time before failure. All engines operate under one condition (sea level) and share one fault mode, high pressure compressor (HPC) degradation. The data are simulated, not recorded from real aircraft.

## Source and citation
The files come from the NASA Prognostics Center of Excellence data set repository, with its included readme. The simulation is described in A. Saxena, K. Goebel, D. Simon and N. Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", Proceedings of the 1st International Conference on Prognostics and Health Management (PHM08), Denver, 2008.

## Columns
Each row is one operating cycle of one engine. The files have 26 whitespace separated columns: unit number, time in cycles, three operational settings and 21 sensor measurements. This project names them engine_id, cycle, setting_1 to setting_3 and sensor_1 to sensor_21. The dataset readme gives no sensor names or units, so this project uses these generic identifiers and does not assert what a sensor physically measures.

## Engine identifiers
The engine_id is unique only within a dataset split. Training engine 24 and test engine 24 are different engines, so every request names the dataset split as well as the engine.

## Remaining useful life target
Remaining useful life (RUL) is counted in operating cycles. For training engines it equals the engine's final recorded cycle minus the current cycle, with no cap. Test engines stop before failure, and a separate file supplies the RUL at each test engine's last observed cycle. The last observed test cycle is not the failure cycle.

## What HPC degradation means here
HPC stands for high pressure compressor. FD001 simulates degradation that originates in the HPC and grows until failure. The data identify no blade, component or physical cause, so this project's models estimate remaining cycles only. They do not diagnose a fault location or prescribe maintenance.

## Cycles are not calendar time
One cycle is one operating cycle of the simulation. Cycles cannot be converted to flight hours or calendar days without an external assumption about how the engine is used.

## Inputs used by the saved models
The saved models use 12 of the 21 sensors (sensor_2, sensor_3, sensor_4, sensor_7, sensor_8, sensor_11, sensor_12, sensor_13, sensor_15, sensor_17, sensor_20 and sensor_21), the cycle count, and features computed from those sensors' past observations. The three operational settings and the other nine sensors are not model inputs.

## Location of the HPC
HPC stands for high pressure compressor. In a turbofan it is part of the compressor section, downstream of the low pressure compressor and upstream of the combustor. The dataset readme gives no sensor locations, so this project does not map any sensor to the HPC or to any other part.

## Engine-level target, not component life
The FD001 target is the number of operating cycles of the whole simulated engine until its failure endpoint, under HPC degradation. The dataset supplies no separate blade, bearing or other component life labels, so the models cannot report a component life.

