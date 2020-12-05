import torch
import torch.nn as nn
import torch.nn.functional as F

class LocationBlock(nn.Module):
    
    def __init__(
        self,
        attention_n_filters,
        attention_kernel_size,
        attention_dim
    ):
        super().__init__()
        
        padding = int((attention_kernel_size - 1) / 2)
        self.conv = nn.Conv1d(
            2, attention_n_filters, kernel_size=attention_kernel_size,
            padding=padding, bias=False, stride=1, dilation=1
        )
        self.projection = nn.Linear(attention_n_filters, attention_dim, bias=False)
    
    def forward(self, attention_weights):
        output = self.conv(attention_weights).transpose(1, 2)
        output = self.projection(output)
        return output

class LocationSensitiveAttention(nn.Module):
    
    def __init__(
        self,
        attention_rnn_dim,
        embedding_dim,
        attention_dim,
        attention_location_n_filters,
        attention_location_kernel_size
    ):
        super(LocationSensitiveAttention, self).__init__()
        
        self.query_layer = nn.Linear(attention_rnn_dim, attention_dim, bias=False)
        self.memory = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=False)
        self.location_layer = LocationBlock(
            attention_location_n_filters,
            attention_location_kernel_size,
            attention_dim
        )
        self.score_mask_value = -float("inf")
        
    def get_alignment_energies(
        self,
        query,
        memory,
        attention_weights_cat
    ):
        """
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)
        """
        processed_memory = self.memory(memory)
        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        
        energies = self.v(torch.tanh(
            processed_query.squeeze(2) + processed_attention_weights + processed_memory
        ))

        energies = energies.squeeze(2)
        return energies
    
    def forward(
        self,
        attention_hidden_state,
        memory,
        attention_weights_cat,
        mask
    ):
        """
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        
        alignment = self.get_alignment_energies(
            attention_hidden_state, memory, attention_weights_cat
        )
        #print(alignment[0])

        alignment = alignment.masked_fill(mask, self.score_mask_value)
        #print(alignment[0])
        #print(mask[0])

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context

        return attention_context, attention_weights