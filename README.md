# DetectiveMangle 🕵️‍♂️🐻

Welcome to **DetectiveMangle**, a charming, friendly, and witty bear detective that helps you uncover the information you need from your uploaded documents. Dressed in a classic Sherlock Holmes costume, Mangle is always ready to solve your document mysteries.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Image Credit](#image-credit)
- [Contributing](#contributing)
- [License](#license)

## Introduction

DetectiveMangle is an AI-powered tool that uses advanced natural language processing to help you get insights from your documents. Upload a document and ask Mangle questions to get clear, concise answers, summaries, and key information.

## Features

- **Answer Questions:** Ask Mangle anything about your uploaded documents.
- **Summarize Content:** Get quick summaries of sections or entire documents.
- **Find Key Information:** Locate important details hidden in your files.

## Installation

To run DetectiveMangle locally, follow these steps:

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/detective-mangle.git
    cd detective-mangle
    ```

2. **Create and activate a virtual environment:**

    ```bash
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Streamlit app:**

    ```bash
    streamlit run app.py
    ```

## Usage

1. **Upload Your Document:**
   Use the file uploader to add a .txt, .pdf, or .docx file.

2. **Ask Mangle:**
   Type your question in the chat, and Mangle will fetch the answer for you.

3. **Enjoy the Insights:**
   Whether it's summarizing a section or finding specific information, Mangle is here to help!

## How It Works

1. **File Upload:**
   Users can upload documents in .txt, .pdf, or .docx formats.

2. **Embedding the Document:**
   The document content is read, split into chunks, and embedded using OpenAI embeddings. These embeddings are cached for faster retrieval.

3. **Retrieving Information:**
   Mangle uses a retrieval model to find relevant chunks based on the user's question.

4. **Generating Responses:**
   The language model (LLM) generates responses based on the retrieved information and the user's question.

## Image Credit

*Image of Mangle in a Sherlock Holmes costume by [Yurang](https://www.instagram.com/yurang_official/?hl=kom).*

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or new features.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
