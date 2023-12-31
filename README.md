# Graph Trasformer applicato all’activity prediction su reti di Petri​ 
This work is a Big Data Analytics and Machine Learning course project. <br>
This project extends the Graph Transformer architecture already developed in a [previous study](https://github.com/graphdeeplearning/graphtransformer) in order to perform activity prediction in graph structured data.
Starting from a dataset of graphs representing the execution of a process model, the goal of this work is to predict the next activity of the process. <br>
In order to do so, this work tries to apply the Graph Transformer architecture to the BPI12 dataset.

### Requirements
To run our application you need to have installed:
* [Python 3.7](https://www.python.org/downloads/release/python-370/)
  
* Conda
  ```
  # For Linux
  curl -o ~/miniconda.sh -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh

  # For OSX
  curl -o ~/miniconda.sh -O https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
  ```
  
* A set of libraries that you can configure inside a Conda Environment by using the [env.yml](https://github.com/chiaragii/gtransformers/blob/main/env.yml)
  ```
  # Install python environment
  conda env create -f env.yml   

  # Activate environment
  conda activate graph_transformer
  ```

### Reproduce results
To run our program copy and paste the following command in your terminal:
```
git clone https://github.com/chiaragii/gtransformers.git
cd gtransformers

python main_BPI_graph_classification.py
```

You can change the configuration parameters in the *GraphsTransformer.json* file inside the config folder.

### Documentation
In this section you can find the documentation of our project: [Documentation](https://github.com/chiaragii/gtransformers/blob/main/docs/Graph%20Trasformer%20applicato%20all%E2%80%99activity%20prediction%20su%20reti%20di%20Petri%E2%80%8B%20.pdf)

### Contributors
| Contributor name | Contacts |
| :-------- | :------- | 
| `Gobbi Chiara`     | chiaragobbi2001@gmail.com | 
| `Moretti Alice`     | morettialice@outlook.it | 
