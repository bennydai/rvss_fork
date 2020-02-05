# Data Fusion Notes
## Rigid Body Transformation
Since the problem is constrained in a 2D world environment, we consider P to be the form of:
```python
P = np.array([[np.cos(theta), -np.sin(theta), x]
              [np.sin(theta), np.cos(theta), y]
              [0, 0, 1]])
```

## Covariance Value
The code contains a set value of 5 as the error. Probably reconsider that value to make SLAM more accurate. It's called process noise. Attempt to have a play with the measurement variance rather the motion variance.

## Kalman Filter
It provides a covariance matrix as part of the SLAM process.
