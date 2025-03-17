# Autocomplete Tool

This is a Streamlit-based tool that uses n-gram models to provide autocomplete suggestions based on text input. The tool is powered by a set of custom n-gram models built from a dataset of queries and generated outputs.

## Requirements
* Python 3.x  
* Streamlit
* pandas
* nltk
* transformers
 

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

## How It Works
- Data Loading & Preprocessing: The script loads a dataset containing queries and generated outputs, filters it, and processes the text by tokenizing and cleaning the sentences.  
- N-gram Model: The tool constructs n-gram models (from 1-gram to 5-gram) from the dataset.  
- Autocomplete Suggestions: Based on the input text, the tool suggests the most likely next words using n-gram probability estimation.  
- Streamlit Interface: The input text is entered through a Streamlit interface, and the tool displays autocomplete suggestions as you type.  

## Running the App
To run the Streamlit app:

```bash
streamlit run app.py --server.fileWatcherType=none
```
The app will open in your default browser where you can type a text, and it will provide suggestions for the next word based on n-gram models.

## Key Functions
- **load_data**: Loads and filters the dataset.
- **prepare_sentences**: Cleans and prepares the sentences for tokenization.
- **tokenize_sentences**: Tokenizes sentences into words.
- **split_data**: Splits the data into training and test sets.
- **count_n_grams**: Counts occurrences of n-grams in the data.
- **estimate_probability**: Calculates the probability of a word given its previous context.
- **suggest_a_word**: Suggests the next word based on the current context.

<br>  
Feel free to contribute, suggest improvements, or report any issues. Happy coding!