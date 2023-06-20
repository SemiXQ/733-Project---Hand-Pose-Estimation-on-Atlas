# VC Lab Course Project with Huawei: Hand Pose Estimation on Atlas
This project is to convert existing models to achieve a performance of FPS>=20 on Huawei's Atlas 200 DK

We have modified the original pipelines of the model in simple2D and retrained it. The prep_utils.py in ./simple2D/utils is from the original model.

Eventually, the whole workflow with handx model is able to achieve 27 FPS with a stable performance.

### To run the model in ./handx:
```python runhandx.py```

### To run the model in ./simple2D:
```python runsimple2d.py```

### The result of handx model:
![handx result](https://github.com/SemiXQ/733-Project---Hand-Pose-Estimation-on-Atlas/blob/master/handx%20result.gif)
