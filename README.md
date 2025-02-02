# Fabric Inspection Chatbot

## 📌 Project Overview
A powerful AI-driven chatbot designed to **query and extract information from large PDF documents** efficiently. This bot leverages **deepseek-r1-distill-llama-70b** **Llama 3**, **Groq API**, **Pinecone** for vector storage, and **Streamlit** for an interactive user interface.

## 🚀 Features
- **Natural Language Processing**: Uses `deepseek-r1-distill-llama-70b` for text understanding and Summarizing.
- **Embeddings with Llama 3.2**: Enhances search performance.
- **Vector Database**: Utilizes `Pinecone` for fast and scalable search.
- **Environment Variables**: API keys and other configurations are securely stored in a `.env` file.

## 🛠️ Installation
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/zanishbilal/chatbot.git
cd chatbot
```

### **2️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3️⃣ Set Up Environment Variables**
Create a `.env` file and add your API keys:
```env
GROQ_API_KEY="your_groq_api_key"
PINECONE_API_KEY="your_pinecone_api_key"
PINECONE_INDEX="your_index_name"
PINECONE_ENV="us-east-1"
```

### **4️⃣ Run the Application**
```bash
python app.py
```

## 🛠️ Usage
- Upload larg pdf.
- Chat with the bot to get insight analysis.
- Retrieve past inspections from Pinecone.

## ⚙️ Configuration
Modify `config.py` to adjust settings like model type, embedding parameters, and vector search preferences.

## 🏗️ Project Structure
```
📂 chatbot

|-- 📂 Rag
    │--- app.py
    |--- document_preprocessing.py
    |--- query_handling.py
    │-- 📂 dataset
    │-- 📂 report
│-- requirements.txt
│-- .env
│-- README.md
```

## 📝 License
This project is open-source and available under the MIT License.

## 🤝 Contributing
Pull requests are welcome! Please follow best coding practices and ensure proper documentation.

