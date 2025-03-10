# Fine-Tuning BERT with Masked Language Modeling

## Overview
This repository demonstrates how to fine-tune BERT (Bidirectional Encoder Representations from Transformers) for the task of Masked Language Modeling (MLM). The model is trained on a custom dataset, specifically the Harry Potter series, to improve its ability to predict missing words within a sentence. This method can be applied to various natural language processing (NLP) tasks, such as enhancing search results and recommendation systems.

The tutorial provided walks you through the steps needed to fine-tune BERT on the Harry Potter dataset, from setting up the environment to training and evaluating the model. By the end of this guide, you will have fine-tuned a BERT model capable of predicting masked words in sentences.

## Requirements
* Python 3.x  
* PyTorch  
* Hugging Face Transformers library  
* Google Colab or a GPU-enabled environment  

## Running the Code
To run this tutorial, simply clone the repository. 

```bash
git clone https://github.com/andjela-r/NLP-Tutorials.git
```

### Setting Up the Environment 
For installing and setting up Poetry: [Poetry Documentation](https://python-poetry.org/docs/)

After installing and setting up Poetry, run

```bash
poetry install --no-root
```
to install all necessary dependencies.


Feel free to contribute, suggest improvements, or report any issues. Happy coding!
