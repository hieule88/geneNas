# import torch
# from datasets import load_dataset
# from transformers import AutoTokenizer
# dataset = load_dataset('glue', 'mrpc', split='train')
# tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
# dataset = dataset.map(lambda e: tokenizer(e['sentence1'], truncation=True, padding='max_length'), batched=True)

# dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
# data = next(iter(dataloader))
# print(**data)
