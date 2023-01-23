def get_finance_train():
  df_train = pd.read_csv("finance_train.csv")
  return df_train
def get_finance_test():
  df_test = pd.read_csv("finance_test.csv")
  return df_test

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

print ("Train and Test Files Loaded as train.csv and test.csv")

LABEL_MAP = {0 : "negative", 1 : "neutral", 2 : "positive"}
NONE = 4 * [None]
RND_SEED=2020

def plot_confusion_matrix(y_true,y_predicted):
  cm = metrics.confusion_matrix(y_true, y_predicted)
  print ("Plotting the Confusion Matrix")
  labels = ["Negative","Neutral","Positive"]
  df_cm = pd.DataFrame(cm,index =labels,columns = labels)
  fig = plt.figure(figsize=(14,12))
  res = sns.heatmap(df_cm, annot=True,cmap='Blues', fmt='g')
  plt.yticks([0.5,1.5,2.5], labels,va='center')
  plt.title('Confusion Matrix - TestData')
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.show()
  plt.close()
  
df_train = get_finance_train()
df_test = get_finance_test()
df_train.head()
sentences = df_train["Sentence"].values
labels = df_train['Label'].values
sentences[0]

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased",do_lower_case=True)

sentences_with_special_tokens = []
for i in sentences:
  s = "[CLS] "
  s2 = " [SEP]"
  sentences_with_special_tokens.append(s+i+s2)
sentences_with_special_tokens[0]


tokenized_texts = []
for i in sentences_with_special_tokens:
  tokenized_texts.append(tokenizer.tokenize(i))

input_ids = []
for i in tokenized_texts:
  input_ids.append(tokenizer.convert_tokens_to_ids(i))
  
  

input_ids = pad_sequences(input_ids, 
                          maxlen=128,
                          dtype="long",
                          truncating="post", 
                          padding="post")

attention_masks = []

for j in input_ids:
  mask = [float(i>0) for i in j]
  attention_masks.append(mask)
  
  
  X_train, X_val, y_train, y_val = train_test_split(input_ids,labels,test_size=0.15,random_state = RND_SEED) 
train_masks, validation_masks, _, _ = train_test_split(attention_masks,input_ids,test_size=0.15,random_state=RND_SEED)

train_inputs = torch.tensor(np.array(X_train));
validation_inputs = torch.tensor(np.array(X_val));
train_masks = torch.tensor(np.array(train_masks));
validation_masks = torch.tensor(np.array(validation_masks));
train_labels = torch.tensor(np.array(y_train));
validation_labels = torch.tensor(np.array(y_val));

batch_size = 32
train_data = TensorDataset(train_inputs, train_masks, train_labels);
train_sampler = RandomSampler(train_data);
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size);
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels);
validation_sampler = SequentialSampler(validation_data); 
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size);



