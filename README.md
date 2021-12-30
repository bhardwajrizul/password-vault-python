# password-vault-python

## Step 0
#### Install all the required libraries

```sh
import sqlite3, hashlib
from tkinter import *
from tkinter import simpledialog
from functools import partial
import uuid
import pyperclip
import base64
import os
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
from cryptography.fernet import Fernet
import dill
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import warnings
import random

#Model
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score 
from sklearn.metrics import confusion_matrix,classification_report
```

## Step 1
#### Run create_model.py using any editor or the command below 

```sh
python3 create_model.py
```

#### It will create a few Files Required for password_vault.py to run

## Step 2
#### Run password_valut.py
```sh
python3 password_vault.py
```
