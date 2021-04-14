# AI Programming with Python Project
## _Image classification '102 flower categories' using  models_

> Note: `you may execute it as follows.` for you train ve pradict execute.

```sh
open the file map
1 - cd /home/workspace/ImageClassifier
```
```sh
execute train.py with parameters
2 - python train.py --save_dir 'result' --arch 'vgg13' --GPU GPU flowers
```

- ✨As a result of this process, we get a model.'checkpoint.pth' inside the result folder. ✨
```sh
Finally, we can check the model with parameters.
3 - python predict.py image_dir 'flowers' load_dir 'result' --GPU GPU 
```
- ✨I print the results of all pictures in the reference folder on the screen. ✨

## Following arguments mandatory or optional for train.py

- 'data_dir'. 'ensure data directory. must argument', type = str
- '--save_dir'. 'ensure saving directory. Optional argument', type = str
- '--arch'. 'Vgg13 can be used if this argument specified, otherwise Alexnet will be used', type = str
- '--lrn'. 'Learning rate, default value 0.001', type = float
- '--hidden_units'. 'Hidden units in Classifier. Default value is 2048', type = int
- '--epochs'. 'Number of epochs', type = int
- '--GPU'. "Option to use GPU", type = str

## Following arguments mandatory or optional for predict.py

- 'image_dir'. 'ensure path to image. must argument', type = str
- 'load_dir'. 'ensure path to checkpoint. must argument', type = str
- '--top_k'. 'Top K most likely classes. Optional', type = int
- '--category_names'. 'Mapping of categories to real names. JSON file name to be ensured. Optional', type = str
- '--GPU'. "Option to use GPU. Optional", type = str


