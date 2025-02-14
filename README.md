# Healthcare Assistant Chatbot

## Overview
This project aims to develop a voice-based AI chatbot designed to assist users with healthcare-related queries. The chatbot will answer questions about symptoms, treatments, medications, and general health advice. This README outlines the plan of action, dataset preparation, model fine-tuning, and deployment steps.

---

## Plan of Action

### Step 1: Dataset Preparation
- **Objective**: Prepare a clean and well-structured dataset for training.
- **Tasks**:
  - Parse and combine multiple XML files into a single JSON file.
  - Clean and preprocess the text data (e.g., remove null or invalid entries, normalize text).
  - Perform exploratory data analysis (EDA) to understand dataset characteristics.
- **Outcome**: A ready-to-use dataset for fine-tuning the model.

### Step 2: Model Fine-Tuning
- **Objective**: Fine-tune a pre-trained language model on the healthcare dataset.
- **Tasks**:
  - Load a pre-trained DialoGPT model using Hugging Face Transformers.
  - Tokenize the dataset and align inputs with the model’s expectations.
  - Fine-tune the model for 1–2 hours using the Trainer API.
- **Outcome**: A fine-tuned model capable of answering healthcare-related queries.

### Step 3: Speech Integration
- **Objective**: Enable voice-based interaction with the chatbot.
- **Tasks**:
  - Use Google Cloud Speech-to-Text for converting voice input to text.
  - Use Google Cloud Text-to-Speech for converting chatbot responses to voice.
- **Outcome**: A fully functional voice-based chatbot.

### Step 4: Deployment
- **Objective**: Deploy the chatbot as a web application.
- **Tasks**:
  - Build a simple Flask-based web interface.
  - Test the chatbot with sample inputs.
- **Outcome**: A user-friendly interface for interacting with the chatbot.

---

## Dataset Details

### Dataset Sources
The dataset used for this project is derived from **MedQuAD** (Medical Question Answering Dataset). MedQuAD contains question-answer pairs related to medical topics, making it ideal for training a healthcare assistant chatbot.

### Combining Files
- **Initial Format**:
  - The original dataset was distributed across multiple XML files.
  - Each XML file contained structured data with fields like `Question`, `Answer`, and metadata.
- **Conversion Process**:
  - **Step 1**: Parse each XML file using Python's `xml.etree.ElementTree` module.
  - **Step 2**: Extract relevant fields (`Question`, `Answer`) and store them in a Python dictionary.
  - **Step 3**: Convert the dictionary into a single JSON file for easier handling during training.
  - **Step 4**: Optionally, convert the JSON file into a CSV file for further analysis.

Example of the final JSON structure:
```json
[
    {"question": "What are the symptoms of leukemia?", "answer": "Symptoms include fatigue, frequent infections..."},
    {"question": "How is diabetes treated?", "answer": "Diabetes is treated with insulin or oral medications..."}
]
```
## Exploratory Data Analysis (EDA)
Before fine-tuning the model, an exploratory data analysis was conducted to understand the dataset's characteristics:
- **QA Pairs**: Counted the total number of question-answer pairs.
- **Question Types**: Analyzed the types of questions (e.g., symptoms, treatments, medications).
- **Answer Length**: Calculated the average length of answers to ensure they fit within the model's token limits.
- **Keyword Frequency**: Identified frequently occurring medical terms (e.g., "symptoms," "treatment").
- **Missing Entries**: Checked for missing or invalid questions/answers and filtered them out.

---

## Preprocessing Steps

### Filtering
- Removed records with missing or invalid entries (e.g., null values, incomplete QA pairs).
- Reduced the dataset size slightly to improve processing speed without significantly affecting quality.

### Normalization
- Lowercased all text.
- Removed special characters and unnecessary whitespace.
- Tokenized the text for model compatibility.

### Splitting
- Divided the dataset into training and validation sets (e.g., 80% training, 20% validation).

---

## Model Fine-Tuning

### Fine-Tuning with DialoGPT

#### Tokenization
- Tokenized the training and validation datasets using the Hugging Face tokenizer.
- Aligned inputs with the model’s expectations by adding `labels` that match the `input_ids`.

#### Fine-Tuning
- Used the `Trainer` API to fine-tune the DialoGPT model.
- Applied **PEFT (LoRA)** for efficient fine-tuning.

#### Saving the Model
- Saved the fine-tuned model checkpoint for future use.

---

## Problem Encountered

During the fine-tuning process, I encountered the following issue:

### Validation Loss Not Logged
- **Description**: The validation loss was not logged during training (`No log`), which prevented us from evaluating the model's performance effectively.
- **Impact**: Without validation metrics, it was challenging to assess whether the model was learning correctly.

### Model Repetition
- **Description**: When testing the model with sample questions, it repeated the input question instead of generating meaningful responses.
- **Example**:
  - Input: `"What are the symptoms of diabetes?"`
  - Output: `"What are the symptoms of diabetes?"`

### Images of the Problem
Below are two images illustrating the issue:

![Image](https://github.com/user-attachments/assets/3c94cf85-46a4-4d8e-9d08-ee0efb99febe)
*Figure 1: Validation loss was not logged during training.*

![Image](https://github.com/user-attachments/assets/735967f0-23a2-44c6-9e4a-293cb49e42e8)
*Figure 2: The model repeated the input question instead of generating a meaningful response.*

# Future Research

To make this project even better, I have a few ideas for what I can work on next:

- **Try Different Models**:  
  I want to explore models like **ClinicalBERT**, which are specifically made for healthcare tasks. These might work better than the one I used now.

- **Improve Fine-Tuning**:  
  I need to learn more about how to fine-tune models effectively. Maybe tweaking things like the learning rate or training time will help solve the problems I ran into.



# Conclusion

This project was a great learning experience for me. I built a healthcare assistant chatbot that can answer questions about symptoms, treatments, and medications using a big dataset and some cool NLP techniques.  

Of course, I ran into a few challenges—like issues with logging validation loss and the model repeating questions—but these are all part of the process. I’ve learned so much along the way, and I feel like I’ve set up a good foundation.  

With a bit more work, I believe I can turn this into a really helpful tool that people can use to get quick answers about their health. I’m excited to keep improving it!
