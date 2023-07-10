import torch
import torch.nn as nn

class SentenceRephrase(nn.Module):
    def __init__(self, model, tokenizer, device, max_sents = 24):
        super(SentenceRephrase, self).__init__()
        # self.model = T5ForConditionalGeneration.from_pretrained(checkpoint).to(device)
        # self.tokenizer = T5Tokenizer.from_pretrained(checkpoint)

        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.beam_size = max_sents # represent a good trade-off between quality and diversity


    def sentence_cut(sentence, t = 140):
        return sentence[:t] if len(sentence) > t  else sentence

    def forward(self, captions):

        print('Rephrasing sentences...')
        print('Original captions: ', captions)

        # move to device and tokenize the captions
        tokenized = self.tokenizer(captions,
                                padding="max_length",
                                truncation=True,
                                max_length=140,
                                return_tensors='pt')

        # generate the rephrased sentences
        simple_tokenized = self.model.generate(tokenized['input_ids'].to(self.device),
                                attention_mask = tokenized['attention_mask'].to(self.device),
                                max_length=140,
                                num_beams=self.beam_size,
                                num_return_sequences=self.beam_size-len(captions)
                            )

        # decode the generated sentences
        rephrased = self.tokenizer.batch_decode(simple_tokenized, skip_special_tokens=True)

        print('Rephrased captions: ', rephrased)

        # add to the original captions
        captions = captions + rephrased
        # cut sentences longer than 140 tokens
        t = 140
        captions = [sentence[:t] if len(sentence) > t else sentence for sentence in captions]

        return captions