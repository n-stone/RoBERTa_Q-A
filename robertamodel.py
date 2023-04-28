import torch.nn as nn
import config
import torch
from transformers import EncoderDecoderModel

model = EncoderDecoderModel.from_encoder_decoder_pretrained(config.ROBERTA_PATH, config.ROBERTA_PATH)

class RoBERTaModel(nn.Module):
    def __init__(self):
        super(RoBERTaModel, self).__init__()
        self.roberta = model

    def generative(self, 
        input_ids,
        do_sample=config.DO_SAMPLE, 
        max_length=config.MAX_LENGTH,
        top_p=config.TOP_P,
        top_k=config.TOP_K,
        temperature=config.TEMPERATURE, 
        no_repeat_ngram_size=config.NO_REPEAT_NGRAM_SIZE,
        num_return_sequence=config.NUM_RETURN_SEQUENCE,
        early_stopping = config.EARLY_STOPPING
        ):
        return self.roberta.generate(
            input_ids,
            do_sample=do_sample,
            max_length=max_length,
            temperature=temperature,
            top_k = top_k,
			top_p = top_p,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=early_stopping

        )
 
    def forward(self, input, labels):
        outputs = self.roberta(input_ids=input, decoder_input_ids=labels, labels=labels)
        return outputs
