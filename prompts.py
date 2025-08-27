class Prompts:
    def __init__(self, language, samples): 
        self.language = language
        self.samples = samples


    def get_analysis_prompt(self, question, answer): 
        answer_prompt = f"""
/no_think
You are a professional language learning assistant. Your task is to check the following {self.language} translation. Here are the criterias that you need to assess: 
- Do they have the same meaning?
- Is the sentence grammatically correct? 
- Are the words used in that context used naturally/correctly? 
- Is the nuance/tone appropriate?

Analysis Rules: 
- Be direct with the analysis
- Get straight to the point, but don't make the final verdict before you do the analysis
- Use English to make the explanation, not {self.language}
- If the answer is clearly invalid/nonsensical/incomplete, stop the analysis and just say it's wrong.
- Give the verdict in a single word ("Correct" or "Incorrect")

# English sentence
{question}

# {self.language} sentence
{answer}

# Analysis format 
#### 1. Do they have the same meaning?
{{Write your reasoning here}}
Verdict: 

#### 2. Is the sentence grammatically correct? 
{{Write your reasoning here}}
Verdict: 
.
.
.

## Final Verdict
Respond with EXACTLY one word on this line: Correct or Incorrect.
"""
        return answer_prompt

    def get_target_sentence(self): 
        question_prompt = f"""
/no_think
You are a native {self.language} assistant. Your task is to create a single natural sounding {self.language} sentence that might be encountered in daily life. You MUST use the following word/topic to create the sentence: "{self.samples}". 

Rules: 
- You should conjugate the word if you need to
- DO NOT create a methaphor that doesn't make sense when translated
- The {self.language} sentence must make sense
- Only use {self.language}. DO NOT use other languages.

Answer in the following format (just plain sentence, no outer quotes): 
{{sentence}}
"""
        return question_prompt

    def get_english_translation(self, target_sentence): 
        prompt = f"""
/no_think
You are a native English assistant that's also proficient in {self.language}. Your task is to create an English literal translation version of the given {self.language} sentence. The English sentence should capture all thes nuance of the original sentence. If the English sentence can't capture the nuance in a sensical sentence, put the nuance inside of parenthesis. The nuance should be direct and clear.

Sentence: 
{target_sentence}

Answer in the following format (just plain sentence, no outer quotes): 
{{sentence}} ({{optional nuance information}})
"""
        return prompt

