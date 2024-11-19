#!/usr/bin/env python3

import argparse
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt
import os
import json
from datetime import datetime

class Chat:
    def __init__(self, model_path, device=None, generation_params=None):
        self.console = Console()
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # HF gen params
        self.default_gen_params = {
            'max_new_tokens': 256,
            'min_new_tokens': 10,
            'num_beams': 10,
            'no_repeat_ngram_size': 3,
            'temperature': 0.7,
            'do_sample': True,           # Enable sampling
            'length_penalty': 1.0,       # Prefer longer sequences
            'early_stopping': True       # Stop when valid ending found
        }
        
        if generation_params:
            self.default_gen_params.update(generation_params)

        # Load model + tokenizer    
        self.console.print(f"Loading model...", style="yellow")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        self.history = []
        
    def generate_answer(self, question):
        inputs = self.tokenizer(
            question,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **self.default_gen_params,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer
    
    def save_history(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.history, f, indent=2)
        self.console.print(f"\nChat history saved to {filename}", style="green")
    
    def format_qa(self, question, answer):
        return f"""### Question\n{question}\n### Answer\n{answer}"""
    
    def start(self):
        self.console.print("\nAsk SummerBot questions about your child's health!", style="bold blue")
        self.console.print("Type 'quit' or 'exit' to end the session", style="dim")
        self.console.print("Type 'save' to save the chat history", style="dim")
        
        while True:
            try:
                question = Prompt.ask("\n[bold blue]Question")
                
                # Handle commands
                if question.lower() in ['quit', 'exit']:
                    if self.history:
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        filename = f'qa_history_{timestamp}.json'
                        self.save_history(filename)
                    break
                    
                if question.lower() == 'save':
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f'qa_history_{timestamp}.json'
                    self.save_history(filename)
                    continue
                
                # Generate answer
                self.console.print("\nThinking...", style="yellow")
                answer = self.generate_answer(question)
                
                # Store in history
                self.history.append({
                    'question': question,
                    'answer': answer,
                    'timestamp': datetime.now().isoformat()
                })
                
                md = Markdown(self.format_qa(question, answer))
                self.console.print(md)
                
            except KeyboardInterrupt:
                self.console.print("\nExiting...", style="yellow")
                if self.history:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f'qa_history_{timestamp}.json'
                    self.save_history(filename)
                break
                
            except Exception as e:
                self.console.print(f"\nError: {str(e)}", style="bold red")

def main():
    parser = argparse.ArgumentParser(description='Interactive QA Model CLI')
    parser.add_argument(
        '--model_path',
        type=str,
        default='qa_model_finetuned',
        help='Path to the fine-tuned model'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        help='Device to run the model on (default: auto-detect)'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=256,
        help='Maximum number of tokens to generate'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Temperature for generation (higher = more random)'
    )
    
    args = parser.parse_args()
    
    # Check if model path exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model path '{args.model_path}' does not exist!")
        print("Please provide the correct path to your fine-tuned model.")
        return
    
    # Custom generation parameters from command line
    generation_params = {
        'max_new_tokens': args.max_tokens,
        'temperature': args.temperature,
    }
    
    # Initialize and run interface
    chat = Chat(args.model_path, args.device, generation_params)
    chat.start()

if __name__ == "__main__":
    main()