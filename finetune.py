import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSeq2SeqLM,  # Changed to seq2seq model
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
import pandas as pd

class DirectQADataset(Dataset):
    def __init__(self, questions, answers, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.questions = questions
        self.answers = answers
        self.max_length = max_length

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        # tokenize question
        question_encoding = self.tokenizer(
            self.questions[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # tokenize answer
        answer_encoding = self.tokenizer(
            self.answers[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': question_encoding['input_ids'].flatten(),
            'attention_mask': question_encoding['attention_mask'].flatten(),
            'labels': answer_encoding['input_ids'].flatten(),
        }

def train_qa_model(
    model, 
    train_dataloader, 
    val_dataloader=None,
    epochs=10,
    lr=2e-5,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0, # since it's a finetune can skip warmup phase
        num_training_steps=total_steps
    )
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}')
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            train_loss += loss.item()
            
            # Backward
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_dataloader)
        print(f"Average training loss: {avg_train_loss}")
        
        # Validation
        if val_dataloader:
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc='Validation'):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    val_loss += outputs.loss.item()
            
            avg_val_loss = val_loss / len(val_dataloader)
            print(f"Validation loss: {avg_val_loss}")

def generate_answer(model, tokenizer, question, device, max_length=128):
    model.eval()
    inputs = tokenizer(
        question,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    # small seq2seq model for local training
    model_name = "google/flan-t5-small"  # "facebook/bart-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    df = pd.read_csv('qa.csv')
    
    questions = df.question.tolist()
    answers = df.answer.tolist()
    
    dataset = DirectQADataset(questions, answers, tokenizer)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    train_qa_model(model, dataloader, epochs=3)
    
    model.save_pretrained("qa_model_finetuned")
    tokenizer.save_pretrained("qa_model_finetuned")
    
    # Test
    test_question = "What foods are safe for my newborn to eat?"
    answer = generate_answer(model, tokenizer, test_question, device='cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Q: {test_question}")
    print(f"A: {answer}")

if __name__ == "__main__":
    main()