 import numpy as np
 import torch
 from transformers import AutoTokenizer, AutoModel
 
 
 def get_word_idx(sent: str, word: str):
     return sent.split(" ").index(word)
 
 
 def get_hidden_states(encoded, token_ids_word, model, layers):
     """Push input IDs through model. Stack and sum `layers` (last four by default).
        Select only those subword token outputs that belong to our word of interest
        and average them."""
     with torch.no_grad():
         output = model(**encoded)
 
     # Get all hidden states
     states = output.hidden_states
     # Stack and sum all requested layers
     output = torch.stack([states[i] for i in layers]).sum(0).squeeze()
     # Only select the tokens that constitute the requested word
     word_tokens_output = output[token_ids_word]
 
     return word_tokens_output.mean(dim=0)
 
 
 def get_word_vector(sent, idx, tokenizer, model, layers):
     """Get a word vector by first tokenizing the input sentence, getting all token idxs
        that make up the word of interest, and then `get_hidden_states`."""
     encoded = tokenizer.encode_plus(sent, return_tensors="pt")
     # get all token idxs that belong to the word of interest
     token_ids_word = np.where(np.array(encoded.word_ids()) == idx)
 
     return get_hidden_states(encoded, token_ids_word, model, layers)
 
 
 def main(layers=None):
     # Use last four layers by default
     layers = [-4, -3, -2, -1] if layers is None else layers
     tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
     model = AutoModel.from_pretrained("bert-base-cased", output_hidden_states=True)
 
     sent = "I like cookies ." 
     idx = get_word_idx(sent, "cookies")

     word_embedding = get_word_vector(sent, idx, tokenizer, model, layers)
     
    return word_embedding 
 
