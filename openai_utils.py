import openai
import pandas as pd
from tabulate import tabulate
import json
import logging

def calculate_message_cost(model, prompt_tokens, completion_tokens):
    if model == "gpt-4o":
        cost = (prompt_tokens / 1_000_000 * 5.00) + (completion_tokens / 1_000_000 * 15.00)
    elif model == "gpt-4o-mini":
        cost = (prompt_tokens / 1_000_000 * 0.150) + (completion_tokens / 1_000_000 * 0.600)
    else:
        cost = 0  # Default to 0 if the model is not recognized
    return cost

# Global dictionary to store token usage
token_usage = {
    "gpt-4o-mini": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    "gpt-4o": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    "total": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
}

# List to store individual call data
token_usage_log = []
# Global variable to store costs
message_costs = []


def chat_with_model(api_key, conversation_history=None, user_message_content=None, model=None, response_format=None, system_message_content=None):
    openai.api_key = api_key

    # Initialize conversation history if not provided
    if conversation_history is None:
        conversation_history = []

    # Add system message if provided
    if system_message_content:
        system_message = {"role": "system", "content": system_message_content}
        conversation_history.insert(0, system_message)  # Insert at the beginning

    # Add user message if provided
    if user_message_content:
        user_message = {"role": "user", "content": user_message_content}
        conversation_history.append(user_message)
    
    # Set default model if not provided
    if model is None:
        model = "gpt-4o-mini"

    # Set response format
    response_format_option = {"type": "json_object"} if response_format == "json" else None

    # Log the full API request
    logging.info(f"\n{'='*50}\nFull API Request:\n{json.dumps(conversation_history, indent=2)}\nModel: {model}\nResponse Format: {response_format_option}\n{'='*50}")

    try:
        completion = openai.chat.completions.create(
            model=model,
            messages=conversation_history,
            response_format=response_format_option
        )
        
        assistant_message = completion.choices[0].message
        conversation_history.append({"role": assistant_message.role, "content": assistant_message.content})

        usage_stats = completion.usage
        message_cost = calculate_message_cost(model, usage_stats.prompt_tokens, usage_stats.completion_tokens)

        # Log the full API response
        logging.info(f"\n{'='*50}\nFull API Response:\n{json.dumps(completion.model_dump(), indent=2)}\n{'='*50}")

        token_usage_log.append({
            'model': model,
            'prompt': user_message_content[:50] if user_message_content else '',
            'prompt_tokens': usage_stats.prompt_tokens,
            'completion_tokens': usage_stats.completion_tokens,
            'total_tokens': usage_stats.total_tokens,
            'cost': message_cost
        })

        logging.info(f"Token usage - Prompt: {usage_stats.prompt_tokens}, "
                     f"Completion: {usage_stats.completion_tokens}, "
                     f"Total: {usage_stats.total_tokens}, "
                     f"Cost: ${message_cost:.6f}")

        return assistant_message.content, usage_stats, message_cost

    except Exception as e:
        logging.error(f"Error in API call: {str(e)}")
        return None, None, 0
    
def print_token_usage_summary():
    if not token_usage_log:
        logging.info("No token usage data available.")
        return

    df = pd.DataFrame(token_usage_log)
    
    # Add a 'Prompt Number' column
    df['Prompt Number'] = range(1, len(df) + 1)
    
    # Reorder columns
    df = df[['Prompt Number', 'model', 'prompt', 'prompt_tokens', 'completion_tokens', 'total_tokens']]
    
    # Create a summary table
    summary_table = tabulate(df, headers='keys', tablefmt='pretty', showindex=False)
    
    # Calculate totals
    totals = df[['prompt_tokens', 'completion_tokens', 'total_tokens']].sum()
    total_row = pd.DataFrame([['TOTAL', '', '', totals['prompt_tokens'], totals['completion_tokens'], totals['total_tokens']]], 
                             columns=df.columns)
    
    # Add total row to the summary
    summary_table += "\n" + tabulate(total_row, headers='keys', tablefmt='pretty', showindex=False)
    
    logging.info("\nToken Usage Summary:")
    logging.info("\n" + summary_table)
    
    # Calculate costs
    gpt4o_usage = df[df['model'] == 'gpt-4o'].sum()
    gpt4o_mini_usage = df[df['model'] == 'gpt-4o-mini'].sum()
    
    gpt4o_cost = (gpt4o_usage['prompt_tokens'] / 1_000_000 * 5.00) + (gpt4o_usage['completion_tokens'] / 1_000_000 * 15.00)
    gpt4o_mini_cost = (gpt4o_mini_usage['prompt_tokens'] / 1_000_000 * 0.150) + (gpt4o_mini_usage['completion_tokens'] / 1_000_000 * 0.600)
    total_cost = gpt4o_cost + gpt4o_mini_cost
    
    logging.info(f"\nEstimated Costs:")
    logging.info(f"GPT-4o: ${gpt4o_cost:.4f}")
    logging.info(f"GPT-4o mini: ${gpt4o_mini_cost:.4f}")
    logging.info(f"Total Cost: ${total_cost:.4f}")

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Your OpenAI API key (replace with your actual key)
    api_key = "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"

    # Initialize conversation history
    conversation_history = []

    # Test with both models
    models = ["gpt-4o-mini", "gpt-4o"]

    for model in models:
        logging.info(f"\nTesting with {model}")
        
        # Test a few interactions
        for i in range(3):
            user_message = f"Tell me a fun fact about the number {i+1}"
            response, usage_stats, cost = chat_with_model(api_key, conversation_history, user_message, model, "text")
            message_costs.append({"model": "gpt-4o-mini", "cost": cost})
            
            if response:
                logging.info(f"User: {user_message}")
                logging.info(f"Assistant: {response}")
                logging.info(f"Cost for this message: ${cost:.6f}")
            else:
                logging.error("Failed to get a response")

        # Clear conversation history for the next model
        conversation_history = []

    # Print token usage summary
    print_token_usage_summary()

if __name__ == "__main__":
    main()