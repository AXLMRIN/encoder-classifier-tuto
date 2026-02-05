# encoder-classifier-tuto

plan détaillé

- **Context + Intro**
  - Présenter cas existants intéressants pour les sciences sociales
  - Présenter les horizons de ce qu'on peut faire avec des encodeurs (sequence classification, token classification, ...) _Nous on présente surtout la partie sequence classification_
  - Présenter un jeu de donées d'intérêt
- **Load and preprocess your data**
  - forewords like in bertopic tutorial on preprocessing steps to consider
    - level (paragraph, sentence)
    - remove emojis (?) remove html tags, ... => IT DEPENDS ON YOUR TASK
    - What length for what model ??
  - Split your data (test, train, valid)
  - code to laod
- **Choose a model**
  - What is hugging face ?
    - explain concept and what you can do
  - Select pretrained model based on the task / domain
  - Select a model based on your computer power
  - Code to load the model
- **Train the model and interpret the results**
  - code for training
  - code for retrieving the results and compute the metrics
  - Rapidly Explain the metrics and how to interpret them 
- **How to get better results**
  - Read a learning curve
  - what are the key hyperparameters
    1. learning rate +++
    2. weight decay 
    3. Batch size / gradient accumulation : stick to 32 / 64 in total, shouldn't be a problem 
    4. optimizer: explain that most of the time it's useless to look at that
    5. precision: the higher the better explain the tradeoff — maybe leave aside for later?
  - code and execute
- **How to use a GPU**
  - Code change
    - .to(device = "cuda")
    - clean GPU memory
  - solve common issue
- **Extra: use another classifier**
  - reuse internship content? ???
- **Extra: Code training steps yourself**
  - reuse internship content ???
- **Save your work for later**
  - save model
  - save embeddings
- **Conclusion**
