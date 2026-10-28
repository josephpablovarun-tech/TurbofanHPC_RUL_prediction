---
doc_id: SG
title: Sensor and Feature Glossary
revision: "1.0"
---

## Sensor names and units
The 21 sensor channels are called sensor_1 to sensor_21. The dataset documentation lists them only as numbered measurements. Names and physical units have not been verified in this project, so none are shown and no sensor is described as measuring a particular quantity.

## Sensors used by the model
The saved models read 12 sensors: sensor_2, sensor_3, sensor_4, sensor_7, sensor_8, sensor_11, sensor_12, sensor_13, sensor_15, sensor_17, sensor_20 and sensor_21. They were chosen in this project's exploratory analysis. Choosing a sensor for the model is not a statement that the sensor is the cause of degradation.

## Trend summaries
The trend tool reports, for each chosen sensor over a window of recent observed cycles, the first and last value, the minimum and maximum, the window mean, the change from first to last, and the least squares slope per cycle. By default it shows the four sensors whose least squares trend over the window is largest relative to that engine's own spread over its observed history. This ranks sensors by recent movement, not by importance to the model or to any failure mechanism.

## Model input features
Each model input is computed only from the engine's current and earlier cycles. The features are rolling means over the last 5, 10 and 20 observations, rolling standard deviations over the last 5 and 10 observations, an exponentially weighted mean, the offset from the engine's early level (its first five cycles), and the cycle count. Rolling windows count observations, not calendar cycles. Early in an engine's history a window holds fewer observations than its nominal size.

## Reading a trend responsibly
A rising or falling sensor trend describes what was observed. It does not identify a cause, a failing part or a maintenance action, and a sensor that moves little can still matter to the model.
