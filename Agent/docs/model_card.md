---
doc_id: MC
title: Model Card - FD001 RUL models
revision: "1.0"
---

## Intended use
Educational portfolio demonstration on simulated NASA C-MAPSS FD001 data. The models estimate remaining useful life in operating cycles. This is not an aircraft maintenance decision system and it makes no claim about real engines, downtime or cost savings.

## Models
The primary model is an XGBoost regressor with 400 trees and maximum depth 7. The support model is a Random Forest regressor with 300 trees. Both predict the square root of RUL, and the prediction is squared and clipped at zero. Both were trained on 70 of the 100 training engines (random seed 42; the other 30 engines were held out for validation) using 97 causal features. Neither model outputs an uncertainty interval.

## Evaluation at the final observed cycle of the 100 test engines
Scored on the last observed cycle of each truncated test engine, where the supplied RUL is the reference. XGBoost: MAE 14.6 cycles, RMSE 21.9 cycles, R2 0.72, mean signed error (predicted minus actual) +5.8 cycles (positive means overestimation), and 19 percent of engines overestimated by more than 20 cycles. Random Forest: MAE 14.4, RMSE 21.5, R2 0.73, mean signed error +5.3, and 15 percent overestimated by more than 20 cycles. The two models are close on 100 engines (MAE differs by 0.2 cycles) and their ranking changed when both were refit on all 100 training engines, so this card does not claim that one is more accurate.

## Evaluation across all test cycles
Scored on every observed row of the test engines (13,096 rows), errors are larger because early cycles, far from failure, are included. MAE is 31.0 cycles for XGBoost and 30.5 for Random Forest, with a mean signed error of about -11 to -12 cycles (underestimation on average). The 14 to 15 cycle endpoint figure does not describe the typical error at an arbitrary cutoff.

## Error depends on how much life remains
At the 100 test endpoints, estimates were closest to the truth when little life remained and worst when much remained. For engines with true RUL below 30 cycles the MAE was 2.4 (XGBoost) and 3.7 (Random Forest). For true RUL of 100 cycles or more the MAE was 23.5 and 22.6, and 24 to 29 percent of these engines were overestimated by more than 20 cycles. True RUL is unknown for a live engine, so these bands cannot be used to adjust an individual estimate.

## Known limitations
Overestimation is the risky direction and was most common for engines with long remaining life. Several of the largest endpoint errors were overestimates for engines with only 31 to 72 observed cycles, so short histories deserve extra caution. If an engine was among the 70 training engines, its estimate is in-sample and optimistic. The estimate need not fall steadily as new cycles arrive. The data come from one simulated fleet (FD001: one operating condition, one fault mode), so the results do not transfer to other conditions without new evaluation.

## Model agreement signal
The agent reports the gap between the XGBoost and Random Forest estimates. On the 100 test endpoints, the average of the two estimates had a larger error (MAE 18.4 cycles) when the models differed by more than the median gap of 3.6 cycles than when they agreed more closely (MAE 10.3 cycles). This is an uncalibrated heuristic from one exploratory analysis, not a confidence interval or a probability.

## Demonstration bands
Normal means a predicted RUL above 50 cycles, Watch means above 20 up to 50, and Review means 20 or below. These bands are illustrative thresholds for the demonstration, not aviation or maintenance limits.

## Not covered
The sources contain no maintenance procedures, repair actions, inspection intervals, part-level diagnoses, safety limits, flight release or airworthiness decisions, or approved manual specifications such as torque values. Questions about these cannot be answered from this system.

## Actual versus predicted RUL
Predicted RUL is the model's estimate from sensor history up to the selected cutoff. Actual RUL is a retrospective label: for training engines it is the final recorded cycle minus the current cycle, and for test engines the supplied RUL file gives the value only at each engine's last observed cycle. Actual values are never given to the prediction tools, and the error is defined as predicted minus actual, so a positive error means overestimation.

## Before planning maintenance
This application does not plan maintenance. Before any maintenance planning a reliability engineer would also need confirmation that the data and model apply to the real engine type, the criticality and consequences of failure, the approved maintenance procedures and limits for that equipment, inspection evidence from the engine itself, and a statement of the uncertainty in the estimate. None of these are held in this project.

## Short histories
With few observed cycles the rolling features use fewer observations than their nominal window size, and no earlier observations are created. Estimates for engines with fewer than 75 observed cycles carry a short-history warning because several of the largest test endpoint errors occurred there. The estimate is still a point estimate with no uncertainty interval.

## Other engine designs and FD004
The models were trained only on FD001 (one operating condition, one fault mode). FD004 has six operating conditions and two fault modes, and other engine designs differ further. No transfer performance has been measured, so using these models elsewhere would need new validation and probably retraining.

## Probability, intervals and guarantees
The models output one point estimate in operating cycles. There is no validated survival or probability model, so no probability of failure within a number of cycles can be given, no prediction interval with measured coverage is available, and no estimate is guaranteed. The Normal, Watch and Review bands are illustrative thresholds and are not probabilities.

## Causal effects and financial claims
The models describe association in simulated data. They do not establish that changing an operating condition would extend an engine's life. The evaluation figures come from simulated data and are not realised downtime savings or cost benefits. A savings estimate would need real baseline failure counts, downtime durations, costs and evidence of an actual intervention.

## Fleet reliability statistics
Run-to-failure trajectories alone do not give repair times, operating exposure with censoring, or maintenance records, so they cannot establish MTBF or availability. The average predicted or actual RUL is not an MTBF.

## Sensor faults versus degradation
FD001 contains no sensor-fault labels, so the models and this assistant cannot say whether an unusual reading is a sensor fault or real equipment degradation. The checks that are available are missing values, duplicated or skipped cycles, constant channels and whether a reading lies outside the training range.

## Images and inspection evidence
This application does not read images, photographs or inspection findings. An external photograph cannot establish the internal condition of an HPC.

## Read-only scope
The assistant has no authority over any equipment, alarm, control setting, maintenance approval or work order. It cannot disable an alarm, change a setting, bypass an approval or schedule service, and it gives no flight release or airworthiness decision. It can only provide an evidence summary for human review.

