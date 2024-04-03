import {Client} from "langsmith";
import {ChatOpenAI} from "@langchain/openai"
import { runOnDataset } from "langchain/smith";
import { evalConfig } from "./functions.js";
import * as dotenv from 'dotenv';

/*
Issue: submission is undefined
*/

// Step 1: Create a dataset
// Inputs are provided to your model, so it know what to generate
const datasetInputs = [
    {question: "a rap battle between Captain Kirk and Captain Picard"},
    {question: "a rap battle between Cardcaptor Sakura and Ash Ketchum"}
    // ... add more as desired
];

// Outputs are provided to the evaluator, so it knows what to compare to
// Outputs are optional but recommended.
const datasetOutputs = [
    {must_mention: ['Spock', 'The prime directive']},
    {must_mention: ['Kero-chan, Charizard']}
];

dotenv.config();
const LANGCHAIN_API_KEY = process.env.LANGCHAIN_API_KEY;

if (!LANGCHAIN_API_KEY) {
    throw new Error('Missing LANGCHAIN_API_KEY in .env file');
}

// needs a langchain api key
const client = new Client({ apiKey: LANGCHAIN_API_KEY });
const datasetName = "Rap Battle Dataset 3";

// Storing inputs in a dataset lets us
// run chains and LLMs over a shared set of examples.
const dataset = await client.createDataset(datasetName, {description: "Rap battle prompts"});

await client.createExamples({
    inputs: datasetInputs,
    outputs: datasetOutputs,
    datasetId: dataset.id
});

// Step 2: Define system
// custum function, runnable, agent or LLM model
const llm = new ChatOpenAI({modelName: "gpt-3.5-turbo", temperature: 0});

// Step 3: Evaluate
  await runOnDataset(llm, datasetName, {
  evaluationConfig: evalConfig,
  // You can manually specify a project name
  // or let the system generate one for you
  // projectName: "chatopenai-test-1",
  projectMetadata: {
    // Experiment metadata can be specified here
    version: "1.0.0",
  },
});



