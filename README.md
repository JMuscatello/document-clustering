# document-clustering

A short python script that assigns a label to text items in a .csv file using unsupervised machine learning methods. 

## Methods
Currently K-Means clustering and LDA (Latent Dirichlet Allocation) methods are implemented. K-means uses (reduced dimensionality) TF-IDF vectors as features to reperesent each piece of text. LDA also uses TF-IDF vectors in calculating the topic distribution for each item, and a label is assigned according to the topic with maximum probability.

## Installation

Set up and activate a virtual environment using:
```
python3 -m venv venv
source venv/bin/activate
```

Install dependencies from `requirements.txt`
```
pip install -r requirements.txt
```

Install NLTK packages in python.
Open a python shell and run the following
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```
## Usage

The script can be run using:
```
python text_clustering -f <path to file>
```
This will fit a LDA model and create a new csv file `output.csv` containing the original contents and a new 'cluster' column containg the integer cluster value.
Other options include:
```
-t 'lda' or 'kmeans'
-n <number of clusters>
-c <csv column name>
```
For a full list of options run:
```
python text_clustering.py --help
```

