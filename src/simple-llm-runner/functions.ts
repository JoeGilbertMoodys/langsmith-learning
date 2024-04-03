import { EvaluatorInputFormatter, RunEvalConfig, runOnDataset } from "langchain/smith";
import { Run, Example } from "langsmith";
import { EvaluationResult } from "langsmith/evaluation";

// You can define any custom evaluator as a function
// The 'run' contains the system outputs (and other trace information).
// The 'example' contains the dataset inputs and outputs.
const mustMention = async ({
    run,
    example,
  }: {
    run: Run;
    example?: Example;
  }): Promise<EvaluationResult> => {
    // Check whether the prediction contains the required phrases
    const mustMention: string[] = example?.outputs?.must_contain ?? [];
    // Assert that the prediction contains the required phrases
    const score = mustMention.every((phrase) =>
      run?.outputs?.output.includes(phrase)
    );
    return {
      key: "must_mention",
      score: score,
    };
  };

  const formatEvaluatorInputs: EvaluatorInputFormatter = function ({
    rawInput, // dataset inputs
    rawPrediction, // model outputs
    rawReferenceOutput, // dataset outputs
    run: Run,
}) {
    return {
        input: rawInput.input,
        prediction: rawPrediction?.output,
        reference: `Must mention: ${rawReferenceOutput?.must_mention ?? [].join(", ")}`,
        submission: Run
    };
};

export const evalConfig: RunEvalConfig = {
    // Custom evaluators can be user-defined RunEvaluator's
    customEvaluators: [mustMention],
    // Prebuilt evaluators
    evaluators: [
      {
        evaluatorType: "labeled_criteria",
        criteria: "helpfulness",
        feedbackKey: "helpfulness",
        // The off-the-shelf evaluators need to know how to interpret the data
        // in the dataset and the model output.
        formatEvaluatorInputs
      },
      {
        evaluatorType: "criteria",
        criteria: {
          cliche: "Are the lyrics cliche?"
        },
        feedbackKey: "is_cliche",
        formatEvaluatorInputs
      },
    ],
  };