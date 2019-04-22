# Generate games

```
python pyrat.py -d 0 -md 0 --rat AIs/manh.py --python AIs/manh.py --nonsymmetric --nodrawing --tests 1000 --synchronous --save
```

### Complete the utils.py file


# Generate the dataset

```
python generate_dataset.py
```


# Complete the train.py file to train a classifier

```
python train.py
```

# Test in pyrat

copy save.pkl, utils.py to the pyrat root folder and supervised to the pyrat root/AIs folder

```
python pyrat.py -d 0 -md 0 --rat AIs/manh.py --python AIs/supervised.py --nonsymmetric --nodrawing --tests 1000 --synchronous --save
```
