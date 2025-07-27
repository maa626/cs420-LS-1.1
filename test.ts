import { ChatOpenAI } from "@langchain/openai";
import 'dotenv/config';

console.log("Testing basic setup...");

async function test() {
  try {
    const llm = new ChatOpenAI({
      model: "gpt-4o-mini",
      temperature: 0
    });
    
    console.log("LLM created successfully!");
    
    const response = await llm.invoke("Hello, just testing if this works!");
    console.log("Response:", response.content);
    
  } catch (error) {
    console.error("Error:", error);
  }
}

test();
