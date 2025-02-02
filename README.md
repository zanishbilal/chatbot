# Fabric Inspection Chatbot

## 📌 Project Overview
This project is a chatbot designed to assist with fabric inspection tasks. It integrates **Groq** for language processing and **Pinecone** for vector storage, making it efficient for retrieving relevant information based on fabric-related queries.

## 🚀 Features
- **Natural Language Processing**: Uses `deepseek-r1-distill-llama-70b` for text understanding.
- **Embeddings with Llama 3.2**: Enhances search performance.
- **Vector Database**: Utilizes `Pinecone` for fast and scalable search.
- **Environment Variables**: API keys and other configurations are securely stored in a `.env` file.

## 🛠️ Installation
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/yourusername/fabric-inspection-chatbot.git
cd fabric-inspection-chatbot
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
- Upload fabric images.
- Chat with the bot to get fabric analysis.
- Retrieve past inspections from Pinecone.

## ⚙️ Configuration
Modify `config.py` to adjust settings like model type, embedding parameters, and vector search preferences.

## 🏗️ Project Structure
```
📂 fabric-inspection-chatbot
│-- 📂 models
│-- 📂 utils
│-- app.py
│-- requirements.txt
│-- .env
│-- README.md
```

## 📝 License
This project is open-source and available under the MIT License.

## 🤝 Contributing
Pull requests are welcome! Please follow best coding practices and ensure proper documentation.

