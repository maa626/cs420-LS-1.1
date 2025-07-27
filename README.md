[![Open in Codespaces](https://classroom.github.com/assets/launch-codespace-2972f46106e565e64193e422d61a12cf1da4916b45550586e14ef0a7c637dd04.svg)](https://classroom.github.com/open-in-codespaces?assignment_repo_id=19971552)

# RAG Chat Bot with LangGraph

A Retrieval Augmented Generation (RAG) application that can answer questions about PDF documents using LangChain and OpenAI.

## Setup Instructions

### 1. Install Dependencies
```bash
npm install
```

### 2. Environment Variables
Create a `.env` file in the project root with:
```env
OPENAI_API_KEY=your-openai-api-key-here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-api-key-here
LANGCHAIN_PROJECT=langgraph-chat-bot-rag
```

### 3. Add Your PDF Document
Place your PDF file in the project root and name it `document.pdf`

Or update the path in `TAG.ts`:
```typescript
const pdfPath = "./your-document-name.pdf";
```

### 4. Run the Application
```bash
npm start
```

## Usage

Once running, you can:
- Ask questions about your PDF content
- Type `chunks` to see all document chunks
- Type `exit` to quit

## Features

- 📄 PDF document processing
- 🔍 Semantic search through document chunks
- 🤖 AI-powered question answering
- 👁️ Full pipeline visibility (see retrieved chunks and prompts)
- 🚫 Strict context-only responses

## Example Questions

Depending on your PDF content, you can ask:
- "What is this document about?"
- "Summarize the main points"
- "What does it say about [specific topic]?"

## Assignment Submission

- Follow the tutorial and submit the working code to GitHub
- Push your code to GitHub
- Submit the link to your GitHub repository for this assignment
