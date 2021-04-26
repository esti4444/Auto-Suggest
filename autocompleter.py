# from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from time import time
import torch
import re

# configuration
MAX_LEN = 3
MIN_LEN = 2
NUM_RETURN_SEQUENCES = 3
TEMPERATURE = 0.7  # language model temperature. 1 is hugginface default


def timing(msg, start_time):
    print("[{:.2f}s] {}".format(time() - start_time, msg))

def clean_text(input_string):
    #  remove spaces/tabs/eol
    clean = re.sub('[ \t\n]+', ' ', input_string)
    clean = re.sub(r"^\s+|\s+$", "", clean)
    # remove non ascii
    clean = re.sub(r'[^\x00-\x7f]', r'', clean)
    clean = "".join(clean.rstrip().lstrip())

    return clean


def delta_text(orig, generated):
    index = generated.find(orig)
    # return None if s2 is not part of s1
    # or if there are no characters behind s2 in s1
    if index != -1 and index + len(orig) < len(generated):
        delta = generated[index + len(orig):]
        return delta
    else:
        return ""


class Autocompleter:

    def __init__(self, model_type):
        self.model_type = model_type
        self.load()

    def load(self):

        start = time()
        # load pretrained model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_type)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_type, output_hidden_states=True)
        timing("---------  Load Model Done {} -----------".format(self.model_type), start)

        # search types configuration
        self.search_config = {
            'BEAM':{
                'temperature':TEMPERATURE,
                'num_beams':5,
                'no_repeat_ngram_size':2,
                'num_return_sequences':NUM_RETURN_SEQUENCES,
                'early_stopping':True,
                'output_scores':True,
                'return_dict_in_generate':True,
                # sort order should be high for higher priority
                'sort_order':2
                },
            'SAMPLE': {
                'temperature': TEMPERATURE,
                'do_sample': True,
                'top_k': 50,
                'top_p': 0.95,
                'num_return_sequences': NUM_RETURN_SEQUENCES,
                'output_scores': True,
                'return_dict_in_generate': True,
                'sort_order': 1
            }
        }

    def get_scores(self, generated_outputs, input_shape):

        gen_sequences = generated_outputs.sequences[:, input_shape:]

        # stack the logits generated at each step to a tensor and transform logits to probs
        probs = torch.stack(generated_outputs.scores, dim=1).softmax(-1)  # -> shape [3, 15, vocab_size]

        # sum the probability of the generated token (add a dummy dim in the end to make gather work)
        gen_probs = torch.gather(probs, 2, gen_sequences[:, :, None]).squeeze(-1)

        # sequences probs in whole model (very small values)
        unique_prob_per_sequence = gen_probs.prod(-1)

        # normalize probs over the three sequences
        normed_gen_probs = gen_probs / gen_probs.sum(0)
        # print("normalized sum:", float(normed_gen_probs[:, 0].sum()))
        check_norm = float(normed_gen_probs[:, 0].sum())
        assert  check_norm > 0.999 and check_norm < 1.001, "probs should be normalized"

        # compare normalized probs to each other
        unique_normed_prob_per_sequence = normed_gen_probs.prod(-1)
        scores = unique_normed_prob_per_sequence/ unique_normed_prob_per_sequence.sum()

        return unique_normed_prob_per_sequence, scores, unique_prob_per_sequence


    def predict(self, text, temp=TEMPERATURE, length=MAX_LEN):
        # Tokenize the input string
        # print("Input:\n" + 100 * '+' + '\n' + text)
        orig_text = text
        text = clean_text(text)
        print("Input Clean:" + text)
        if text == None or text == "":
            return "", "", ["", "", ""]

        input = self.tokenizer.encode(text, return_tensors="pt")
        decoded_input = self.tokenizer.decode(input[0], skip_special_tokens=True)
        max_length = input.shape[1] + length
        min_length = input.shape[1] + MIN_LEN
        highest_score = 0.0
        best_option = ""
        best_delta = ""
        del_len = 0
        options = {}

        for config in self.search_config.keys():
            start = time()
            args = self.search_config[config]
            args['temperature'] = temp
            args['max_length'] = max_length
            args['min_length'] = min_length

            # Note: issue with transformers 4.3.  use 4.2
            # conda install -c huggingface transformers==4.2.0
            outputs = self.model.generate(input, **args)
            print("Output {}:\n".format(config) + 100 * '-')
            scores, scores_norm, raw_prob = self.get_scores(outputs, input.shape[-1])

            for i, output in enumerate(outputs.sequences):
                gen = self.tokenizer.decode(output, skip_special_tokens=True)
                # print("\nfrom decode:", gen)
                generated_txt = clean_text(gen)
                # print("\ncleaned decode:", generated_txt)
                delta = delta_text(decoded_input, generated_txt)
                # delta_len = output.shape[0] - input.shape[1]
                delta_len = len(delta.split())
                s2 = text + " ---- " + generated_txt
                assert(delta_len < MAX_LEN + 1), s2
                print("{}: {} --> {}, {}".format(scores_norm[i], delta, scores[i], raw_prob[i]))
                # set all options
                if (delta != "") and delta not in options:
                        options[delta] = (float(scores_norm[i]), self.search_config[config]["sort_order"])
                #calc best option
                if (delta != "") and (float(scores_norm[i]) > highest_score):
                    highest_score = float(scores_norm[i])
                    best_option = generated_txt
                    best_delta = delta
                    del_len = delta_len
            timing("---------  Predict {} -----------".format(config), start)

        # assert (best_delta != ""), decoded_input + '-->' + generated_txt

        # sort options by prob
        opt = sorted(options, key=lambda k: (options[k][1], options[k][0]), reverse= True)
        num_of_options = len(opt)
        # add empty values if less than 3 options returned
        for i in range(3 - num_of_options):
            opt.append("")
        print("Best Delta:\n" + 50 * '@' + 'XXX  ' + best_delta + " len " + str(del_len))
        print("3 options:\n" + 50 * '@' + 'XXX  ' + ' '.join(map(str, opt[:3])))

        for i in range(min(len(options), 3)):
            if i>0:
                # check that prior option has higher prob AND its within the same search type
                invalid = (options[opt[i-1]][0] < options[opt[i]][0]) and (options[opt[i-1]][1] == options[opt[i]][1])

                k = str(i) +' '+ opt[i-1] +' '+ str(options[opt[i-1]][0]) +' '+ opt[i] +' '+ str(options[opt[i]][0])
                assert (not invalid), k

        # handle spaces: add space on left to the suggested option if needed
        for i in range(len(opt)):
            if orig_text.endswith(' '):
                opt[i] = opt[i].lstrip()
            elif not opt[i].startswith(' '):
                opt[i] = " " + opt[i].lstrip()
        return best_option, best_delta, opt[:3]


# for testing ------------------------------------------------------------------------------------------------------
def iterate_text(session_text):
    text = input("Enter text: \n")
    if text == "Q":
        return "Q"
    elif text == "S":
        session_text = input("Enter text: \n")
    elif text != "R":
        session_text = session_text + ' ' + text
    # auto_text = completer.predict(session_text, float(temp), int(len))
    session_text, delta, options = completer.predict(session_text)
    return session_text

if __name__ == "__main__":
    completer = Autocompleter("distilgpt2")
    session_text = ""

    # manual predictions
    # while session_text != "Q":
    #     session_text = iterate_text(session_text)

    #automatix prediction
    # session_text = "I want to make sure that we have a good relationship with each other."
    session_text = "I want to"
    for i in range(10):
        session_text, delta, options = completer.predict(session_text)




#------ th gumble trick from allenlp

    # https://medium.com/ai2-blog/a-guide-to-language-model-sampling-in-allennlp-3b1239274bc3

    # from allennlp_models.pretrained import load_predictor
    # predictor = load_predictor(
    #     "lm-next-token-lm-gpt2",
    #     overrides={
    #         "model.beam_search_generator": {
    #             "type": "transformer",
    #             "namespace": "gpt2",
    #             "beam_search": {
    #                 "sampler": {
    #                     "type": "gumbel",
    #                 },
    #                 "end_index": 50256,
    #                 "max_steps": 18,
    #                 "beam_size": 4,
    #             },
    #         }
    #     },
    # )
    # text = "I went into town on Saturday morning because..."
    # print(text)
    # for tokens in predictor.predict(text)["top_tokens"]:
    #     string = "".join(tokens).replace("Ġ", " ").replace("<|endoftext|>", "").strip()
    #     print(" ->", string)

    #---- logits-------------------------
    # inputs = self.tokenizer("Hello, my dog is cute", return_tensors="pt")
    # outputs = self.model(**inputs, labels=inputs["input_ids"])
    # loss = outputs.loss
    # logits = outputs.logits


#-------------------   No recalc attention
#-https://huggingface.co/transformers/quickstart.html#using-the-past
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
# import torch
#
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# model = GPT2LMHeadModel.from_pretrained('gpt2')
#
# generated = tokenizer.encode("The Manhattan bridge")
# context = torch.tensor([generated])
# past = None
#
# for i in range(100):
#     print(i)
#     output, past = model(context, past=past)
#     token = torch.argmax(output[..., -1, :])
#
#     generated += [token.tolist()]
#     context = token.unsqueeze(0)
#
# sequence = tokenizer.decode(generated)
#
# print(sequence)


#------- use cache in model foward:
# use_cache=False,


#  --- fine tine
# https://huggingface.co/transformers/custom_datasets.html?highlight=cuda
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#
# model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
# model.to(device)
# model.train()
#
# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
#
# optim = AdamW(model.parameters(), lr=5e-5)
#
# for epoch in range(3):
#     for batch in train_loader:
#         optim.zero_grad()
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         labels = batch['labels'].to(device)
#         outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
#         loss = outputs[0]
#         loss.backward()
#         optim.step()
#
# model.eval()