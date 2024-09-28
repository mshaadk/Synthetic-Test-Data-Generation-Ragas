# Synthetic Test Data Generation using Ragas
This project focuses on generating synthetic test data using Ragas with the help of pre-trained models like LLaMA3 for data generation and critique. The project is implemented in Google Colab, which makes it accessible and easy to use for running machine learning experiments in the cloud. We use embeddings from HuggingFace's BAAI/bge-small-en model and domain-specific document loaders, such as PubMedLoader, for medical content.

## Overview
The purpose of this project is to create synthetic test data to benchmark machine learning models using the Ragas testset generator. The synthetic test data is essential for simulating scenarios where real-world data may be unavailable or too sensitive to use. This project leverages the LLaMA3 model for both test data generation and critique, providing a robust framework for creating diverse test sets.

## Features
- **Document Loading**: Loads domain-specific documents (e.g., medical research papers) using PubMedLoader.
- **Embeddings**: Uses HuggingFace embeddings model BAAI/bge-small-en.
- **Synthetic Data Generation**: Creates synthetic test sets with simple, multi-context, and reasoning questions.
- **Data Critique**: The generated data is critiqued by the LLaMA3 model for quality assurance.

## Google Colab Setup
### Step 1: Open the Colab Notebook
Start by opening Google Colab and create a new notebook or open the existing one for the project.

### Step 2: Install Required Libraries
Run the following code to install all the necessary dependencies.

```python
# Warning control
import warnings
warnings.filterwarnings('ignore')

!pip install ragas langchain_community langchain_groq sentence_transformers xmltodict -q
```

### Step 3: Set Up API Keys and Environment Variables
In your Google Colab notebook, use the google.colab module to access userdata and store your API key. You will need the Groq API Key to run this project.

```python
import os
from google.colab import userdata

os.environ["GROQ_API_KEY"] = userdata.get('GROQ_API_KEY')
```

Make sure to provide your GROQ_API_KEY when prompted by Colab.

### Step 4: Initialize Models and Load Data
Load documents and initialize embeddings with HuggingFaceBgeEmbeddings.

```python
from langchain_community.document_loaders import PubMedLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_groq import ChatGroq

# Initialize models
data_generation_model = ChatGroq(model="llama3-8b-8192", groq_api_key=os.environ["GROQ_API_KEY"])
critic_model = ChatGroq(model="llama3-8b-8192", groq_api_key=os.environ["GROQ_API_KEY"])

# Setup embeddings
model_name = "BAAI/bge-small-en"
embeddings = HuggingFaceBgeEmbeddings(model_name=model_name, model_kwargs={"device": "cpu"}, encode_kwargs={"normalize_embeddings": True})

# Load documents
loader = PubMedLoader("cancer", load_max_docs=5)
documents = loader.load()
```

### Step 5: Generate Synthetic Test Data
Generate test sets based on different question categories such as simple, multi-context, and reasoning.

```python
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context

# Set distributions for test data generation
distributions = {simple: 0.5, multi_context: 0.4, reasoning: 0.1}

# Generate the test data
generator = TestsetGenerator.from_langchain(data_generation_model, critic_model, embeddings)
testset = generator.generate_with_langchain_docs(documents, 5, distributions)

# Convert to DataFrame for analysis
test_df = testset.to_pandas()
test_df.head()
```

### Step 6: Save Generated Data
Once the synthetic test data is generated, you can save it as a CSV file or process it further.

```python
test_df.to_csv('synthetic_test_data.csv', index=False)
```

## Collaborations
This project is open to collaborations! Whether you are a researcher, developer, or data scientist, we welcome contributions in the form of:

- **Feature Enhancements**: Adding new test scenarios or models.
- **Bug Fixes**: Help improve the reliability of the synthetic data generation process.
- **Documentation**: Improve the clarity and depth of documentation.
  
## How to Collaborate
1. Fork the repository.
2. Make a new branch:

```bash
git checkout -b feature-branch
```

3. Commit your changes:
```bash
git commit -m "Add new feature"
```

4. Push the changes to your branch:
```bash
git push origin feature-branch
```

5. Open a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE.txt) file for more details.
