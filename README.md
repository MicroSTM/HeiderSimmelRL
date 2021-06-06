# Generating Heider-Simmel animations used for experiments

## Requirements
- Python 3.5
- Numpy >= 1.14.2
- PyTorch 0.3.1
- [pybox2d](https://github.com/pybox2d/pybox2d)
- pygame
- OpenCV >= 3.4.2

## HH

### Style 1, 100% AD
```
python -m experiments.gen_social --exp-name Blocking_v1 --checkpoints 2000 16000 --action-freq 1
```

### Style 2, 100% AD
```
python -m experiments.gen_social --exp-name Blocking_v2 --checkpoints 34400 16000 --action-freq 1
```


### Change AD
Change the argument ``action-freq`` from 1, 2, 5, 10, to 15 for 100% AD, 50% AD, 20% AD, 10% AD, and 7% AD respectively


## HO

### Style 1
```
python -m experiments.gen_social --exp-name Blocking_v0 --checkpoints 0 16000
```

### Style 2
```
python -m experiments.gen_social --exp-name Blocking_v0 --checkpoints 0 46000 
```

### Style 3
```
python -m experiments.gen_social --exp-name Blocking_v0 --checkpoints 0 16000 --restitution 1
```

## OO

### collision
```
python -m experiments.gen_physical --exp-name collision
```

### rod
```
python -m experiments.gen_physical --exp-name rod
```

### rope
```
python -m experiments.gen_physical --exp-name rope
```

### spring
```
python -m experiments.gen_physical --exp-name rope
```
