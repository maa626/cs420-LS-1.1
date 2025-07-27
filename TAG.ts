
// Import all dependencies at the top
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { Document } from "@langchain/core/documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { Annotation, StateGraph } from "@langchain/langgraph";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import "cheerio";
import 'dotenv/config';
import { pull } from "langchain/hub";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import * as readline from 'readline';

//LLM Model
const llm = new ChatOpenAI({
  model: "gpt-4o-mini",
  temperature: 0
});

//Embedding Model
const embeddings = new OpenAIEmbeddings({
  model: "text-embedding-3-large"
});

//Vector Store
const vectorStore = new MemoryVectorStore(embeddings);

// Main async function to handle all async operations
async function main() {
  try {
    console.log("Starting RAG application...");

    // Load and chunk contents of blog
    const pTagSelector = "p";
    const cheerioLoader = new CheerioWebBaseLoader(
      "https://lilianweng.github.io/posts/2023-06-23-agent/",
      {
        selector: pTagSelector
      }
    );

    console.log("Loading documents from website...");
    const docs = await cheerioLoader.load();
    console.log(`Loaded ${docs.length} documents`);
    console.log(`First document preview: ${docs[0]?.pageContent?.slice(0, 200)}...`);

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000, chunkOverlap: 200
    });
    const allSplits = await splitter.splitDocuments(docs);
    console.log(`Split into ${allSplits.length} chunks`);
    
    // Optional: Show first few chunks for inspection
    console.log("\n📄 First 3 chunks preview:");
    allSplits.slice(0, 3).forEach((chunk, index) => {
      console.log(`\nChunk ${index + 1}:`);
      console.log(`Length: ${chunk.pageContent.length} characters`);
      console.log(`Preview: ${chunk.pageContent.slice(0, 100)}...`);
    });

    // Index chunks
    console.log("Adding documents to vector store...");
    await vectorStore.addDocuments(allSplits);
    console.log("Documents indexed successfully");

    // Define prompt for question-answering
    console.log("Loading RAG prompt template...");
    const promptTemplate = await pull<ChatPromptTemplate>("rlm/rag-prompt");
    console.log("Prompt template loaded");

    // Alternative: Use a custom strict prompt
    const strictPromptTemplate = ChatPromptTemplate.fromTemplate(`
You are an assistant for question-answering tasks. Use ONLY the following pieces of retrieved context to answer the question. 

STRICT RULES:
- If the answer is not in the provided context, say "I cannot answer this question based on the provided article content."
- Do not use any external knowledge beyond what's in the context (exept as complementary)
- Stay focused on the article content only
- Use three sentences maximum and keep the answer concise

Context: {context}

Question: {question}

Answer based ONLY on the context above:`);

    console.log("Custom strict prompt template created");

    // Define state for application
    const InputStateAnnotation = Annotation.Root({
      question: Annotation<string>,
    });

    const StateAnnotation = Annotation.Root({
      question: Annotation<string>,
      context: Annotation<Document[]>,
      answer: Annotation<string>,
    });

    // Define application steps
    const retrieve = async (state: typeof InputStateAnnotation.State) => {
      console.log("\n🔍 RETRIEVAL STEP");
      console.log("Question:", state.question);
      const retrievedDocs = await vectorStore.similaritySearch(state.question, 4); // Get top 4 chunks
      console.log(`Retrieved ${retrievedDocs.length} relevant document chunks:`);
      
      // Show details of retrieved chunks
      retrievedDocs.forEach((doc, index) => {
        console.log(`\n--- Chunk ${index + 1} ---`);
        console.log(`Content Preview: ${doc.pageContent.slice(0, 150)}...`);
        console.log(`Full Length: ${doc.pageContent.length} characters`);
        console.log(`Source: ${doc.metadata?.source || 'N/A'}`);
        console.log(`--- End Chunk ${index + 1} ---`);
      });
      
      return { context: retrievedDocs };
    };

    const generate = async (state: typeof StateAnnotation.State) => {
      console.log("\n🤖 GENERATION STEP");
      console.log("Generating answer using retrieved context...");
      console.log(`Using ${state.context.length} chunks for context`);
      
      const docsContent = state.context.map(doc => doc.pageContent).join("\n");
      console.log(`Total context length: ${docsContent.length} characters`);
      
      // Use the strict prompt template for better control
      const messages = await strictPromptTemplate.invoke({ question: state.question, context: docsContent });
      
      // Show the actual prompt being sent to the LLM
      console.log("\n📝 PROMPT BEING SENT TO LLM:");
      console.log("=" .repeat(50));
      console.log(messages.messages[0].content);
      console.log("=" .repeat(50));
      
      console.log("Sending to LLM for answer generation...");
      
      const response = await llm.invoke(messages);
      console.log("✅ Answer generated successfully");
      
      return { answer: response.content };
    };

    // Compile application and test
    console.log("Compiling graph...");
    const graph = new StateGraph(StateAnnotation)
      .addNode("retrieve", retrieve)
      .addNode("generate", generate)
      .addEdge("__start__", "retrieve")
      .addEdge("retrieve", "generate")
      .addEdge("generate", "__end__")
      .compile();
    console.log("Graph compiled successfully");

    // Create readline interface for user input
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout
    });

    // Function to ask questions interactively
    const askQuestion = () => {
      rl.question('\nEnter your question (or "exit" to quit, "chunks" to see all chunks): ', async (userInput) => {
        if (userInput.toLowerCase() === 'exit') {
          console.log("Goodbye!");
          rl.close();
          return;
        }

        if (userInput.toLowerCase() === 'chunks') {
          console.log(`\n📄 All ${allSplits.length} chunks in the vector store:`);
          allSplits.forEach((chunk, index) => {
            console.log(`\n=== Chunk ${index + 1} ===`);
            console.log(`Length: ${chunk.pageContent.length} characters`);
            console.log(`Content: ${chunk.pageContent.slice(0, 200)}...`);
            console.log(`Metadata:`, chunk.metadata);
          });
          askQuestion();
          return;
        }

        if (userInput.trim() === '') {
          console.log("Please enter a valid question.");
          askQuestion();
          return;
        }

        try {
          console.log("\n🔍 Processing your question...");
          const result = await graph.invoke({
            question: userInput
          });
          
          console.log("\n=== RESULT ===");
          console.log("❓ Question:", result.question);
          console.log("💡 Answer:", result.answer);
          console.log("=============");
          
          // Ask for next question
          askQuestion();
        } catch (error) {
          console.error("❌ Error processing question:", error);
          askQuestion();
        }
      });
    };

    // Start the interactive session
    console.log("\n🎉 RAG Application is ready!");
    console.log("You can now ask questions about the article on autonomous agents.");
    console.log("📖 Article: https://lilianweng.github.io/posts/2023-06-23-agent/");
    console.log("\n💡 Commands:");
    console.log("  - Ask any question about the article");
    console.log("  - Type 'chunks' to see all document chunks");
    console.log("  - Type 'exit' to quit");
    askQuestion();

  } catch (error) {
    console.error("Error in main function:", error);
    console.error("Stack trace:", error.stack);
  }
}

// Run the main function
main();
