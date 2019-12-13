# neural-semantic-relatedness
Analysis on Semantic Relatedness with Attentive Tree LSTM over BiLSTM


In this project we explored the state-of-the-art BiLSTM model, where in we understood what information it captures and what kind of applications it can be employed in. We also explored the current TreeLSTM models and its applications. We came up with our own model which combines BiLSTM with TreeLSTM model to create a new hierarchical model that gives emphasis on certain words using the attention based mechanism in the TreeLSTM.
Our model was able to improve upon the current TreeLSTM and BiLSTM accuracies by 1.02\% in the task of semantic  relatedness  using  both  attention  and Bi-LSTM.  No  significant  improvement  was  observed  in  the  task  of  contextual  similarity  using Quora dataset with our approach.  Observing the results it can be seen that the Bi-LSTM trained on outputs from attentive tree LSTM outperforms or is par with all the other models.
