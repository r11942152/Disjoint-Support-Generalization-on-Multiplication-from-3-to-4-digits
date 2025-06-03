## Step 1: Setup Environment

Ensure you have Python 3.10 or above installed on your system. You can check your Python version by running:

```
python --version
$ Python 3.10.12
```

If you need to update Python, visit the [official Python website](https://www.python.org/downloads/).

Next, it's recommended to create a virtual environment:

```
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

Then, install the required packages:

```
pip install -r requirements.txt
```

## Step 2: Generate Dataset

Run the following command to set up the necessary datasets:

```
python3 data_generate.py
python3 iid_test_data_generate.py
python3 ood_test_data_generate.py
```

## Step 3: Train Model

Run the following command to train the model
```
python3 training.py
```
