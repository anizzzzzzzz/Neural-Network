
### Prerequisities
Need python 3.6 or any.

### Installation
```sh
$ git clone https://github.com/anizzzzzzzz/Neural-Network.git
$ cd Neural-Network/
```

### Creating Virtual Environment
To create a virtual environment, go to your project's directory and run virtualenv.

#### On macOS and Linux:
``` 
python3 -m virtualenv venv
``` 
#### On windows:
``` 
py -m virtualenv venv
```
Note : The second argument is the location to create the virtualenv. Generally, you can just create this in your project and call it venv.
[ virtualenv will create a virtual Python installation in the venv folder.]

### Activating Virtual Environment
Before you can start installing or using packages in your virtualenv, you'll need to activate it.

#### On macOS and Linux:
```
source venv/bin/activate
```

#### On windows:
```
.\venv\Scripts\activate
```

#### Confirming virtualenv by checking location

##### On macOS and Linux:
```
which python
```
Output : .../venv/bin/python

##### On windows:
```
where python
```
Output : .../venv/bin/python.exe


### Installing packages with pip
```
pip install -r requirements.txt
```

#### For MultilayerANN
##### Download Data
The link, [MNIST Handwritten Digits Data](http://yann.lecun.com/exdb/mnist/index.html), points you to the MNIST database of handwritten digits, available from this page, has a training set of 60,000 examples, and a test set of 10,000 examples.

Download all four zipped data and unzip them and place it into
```
[path_of_project]/Neural-Network/src/MultilayerAnn/data
```

* Run LoadData.py to load the handwritting data and save it into numpy zip files for efficient loading later. It will create 'mnist_scaled.npz' file inside directory.
* Train the model by executing TrainData.py file. After training completion, model will be saved in model directory along with images of cost and training/validation accuracy.
* Run PredictData.py for testing the test-data.numpy zip files 