# ExperimentWalkRideClassifier

Usage:
```python
walk_detect = WalkDetect("model.joblib")
is_walk = walk_detect.check_is_walk("motion.csv", gps_file="gps.csv")
print(f"Is walk {is_walk}")
```
