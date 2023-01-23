test_sentences = df_test['Sentence'].values
test_labels = df_test['Label'].values

test_input_ids, test_attention_masks = [], []


test_sentences = ["[CLS] " + sentence + " [SEP]" for sentence in test_sentences]

tokenized_test_sentences = [tokenizer.tokenize(sent) for sent in test_sentences]


test_input_ids = [tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_test_sentences]


test_input_ids = pad_sequences(test_input_ids, 
                               maxlen=128, 
                               dtype="long",
                               truncating="post", 
                               padding="post")


for sequence in test_input_ids:
  mask = [float(i>0) for i in sequence]
  test_attention_masks.append(mask)
  
  
  batch_size = 32  
test_input_ids = torch.tensor(test_input_ids)
test_attention_masks = torch.tensor(test_attention_masks)
test_labels = torch.tensor(test_labels)
prediction_data = TensorDataset(test_input_ids, test_attention_masks, test_labels)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

print('Predicting labels for {:,} test sentences...'.format(len(test_input_ids)))


model.eval()

predictions , true_labels = [], []

for batch in prediction_dataloader:
  batch = tuple(t.to(device) for t in batch)
  b_input_ids, b_input_mask, b_labels = batch
  with torch.no_grad():
      outputs = model(b_input_ids, token_type_ids=None, 
                      attention_mask=b_input_mask)

  logits = outputs[0]

  logits = logits.detach().cpu().numpy()
  label_ids = b_labels.to('cpu').numpy()

  predictions.append(logits)
  true_labels.append(label_ids)

  
print ('Test Accuracy: {:.2%}'.format(flat_accuracy(logits, label_ids)))
#df_test['Sentence']
