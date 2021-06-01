"""
Text Autocompleter based on Huggingface Transformers (GPT2)
"""
from transformers import AutoTokenizer, AutoModelForCausalLM
    # , GPT2Tokenizer, GPT2LMHeadModel
from time import time
import torch
import re

# configuration
MAX_LEN = 3
MIN_LEN = 2
NUM_RETURN_SEQUENCES = 3
TEMPERATURE = 0.7  # language model temperature. 1 is hugginface default  # used in sampling decode type only
MAX_PROMPT = 100

def timing(msg, start_time):
    print("[{:.2f}s] {}".format(time() - start_time, msg))

def clean_text(input_string):
    """
    clean a given string from non ascii chars and redundant spaces/tabs/eol
    :param input_string:
    :return: clean string
    """
    #  remove spaces/tabs/eol
    clean = re.sub('[ \t\n]+', ' ', input_string)
    clean = re.sub(r"^\s+|\s+$", "", clean)
    # remove non ascii
    clean = re.sub(r'[^\x00-\x7f]', r'', clean)
    clean = "".join(clean.rstrip().lstrip())

    return clean

def delta_text(orig, generated):
    """
    get the ending text delta between orig and generated
    return the substrng that was automatically generated
    :param orig: the prompt that was given for prediction
    :param generated: the generated text (including the orig prompt)
    :return: text after orig string within generated text
    """
    index = generated.find(orig)
    # return "" if orig string is not part of generated string
    # or if there are no characters after orig in generated string
    if index != -1 and index + len(orig) < len(generated):
        delta = generated[index + len(orig):]
        return delta
    else:
        return ""

def trim_promt(text):
    """
    trim prompt text upto MAX_PROMPT
    :param text:
    :return: trimmed text
    """
    t = text.split()
    if len(t) > MAX_PROMPT:
        trimmed = t[-MAX_PROMPT:]
        text = ' '.join(trimmed)
    return text

class Autocompleter:
    """
    Autocompleter uses pretrained language model to predict next words for a given prompt
    """
    def __init__(self, model_type):
        self.model_type = model_type
        self.load()

    def load(self):

        start = time()
        # load pretrained model
        if self.model_type == "fine_tuned":
            self.tokenizer = AutoTokenizer.from_pretrained("./reference_model")
            self.model = AutoModelForCausalLM.from_pretrained("./reference_model", output_hidden_states=True)
        else:
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
                }
            # ,
            # 'SAMPLE': {
            #     'temperature': TEMPERATURE,
            #     'do_sample': True,
            #     'top_k': 50,
            #     'top_p': 0.95,
            #     'num_return_sequences': NUM_RETURN_SEQUENCES,
            #     'output_scores': True,
            #     'return_dict_in_generate': True,
            #     'sort_order': 1
            # }
        }

    def get_scores(self, generated_outputs, input_shape):
        """
        Calculate probabilities for predicted phrases form the model output
        :param generated_outputs: model's output including generated sequences and their scores
        :param input_shape: length of given prompt to the model. used to retrieve the generated part from the full output sequence (including the prompt)
        :return:
            unique_normed_prob_per_sequence - option probs(options' words probs product): predicted words' probs are normalized across prediction step
            scores - normalized unique_normed_prob_per_sequence (normalization of options prob)
            unique_prob_per_sequence - options' words raw probs product: probs not normalized across options/steps
        """
        #sequences_scores attribute exist only for beam search
        # if hasattr(generated_outputs, 'sequences_scores'):
        #     g = torch.nn.functional.softmax(generated_outputs.sequences_scores)
        #     print("---- BEAM sequences_scores: ", g)

        gen_sequences = generated_outputs.sequences[:, input_shape:]  # -> shape [n_options, n_competing_words] [3,3]

        # stack the logits generated at each step (=n_completing words) to a tensor and transform logits to probs
        # scores= tuple(n_completing words length, i.e number of prediction steps), each contain logits with shape [n_beams, vocab_size]
        # stacking on dim=1 returns [n_beams, steps, vocab]. (each beam gathers its steps)
        probs = torch.stack(generated_outputs.scores, dim=1).softmax(-1)  # -> shape [n_beams, 3 prediction steps (stacked tensores), vocab_size]   (softmax on vocab_size)

        # sum the probability of the generated token (add a dummy dim in the end to make gather work, and unsqueeze to remove the dummy dim)
        # gather (retrieve by index of generated tokens for each option (gen_sequences)) the probs (after softmax) from vocab dim
        gen_probs = torch.gather(probs, 2, gen_sequences[:, :, None]).squeeze(-1)  # -> shape [n options, 3 steps(completing words softmaxed probs)]

        # sequences probs in whole model (very small values)
        # for each option - calc product of its words probability  (product of last dim: product of word probs in an option)
        unique_prob_per_sequence = gen_probs.prod(-1)   # --> shape [n_options]

        # normalize probs over the n_options for each prediction step
        normed_gen_probs = gen_probs / gen_probs.sum(0)  # --> shape [n_options, n_completing_words]
        # print("normalized sum:", float(normed_gen_probs[:, 0].sum()))
        check_norm = float(normed_gen_probs[:, 0].sum())
        assert  check_norm > 0.999 and check_norm < 1.001, "probs should be normalized"

        # compare normalized probs across n_options:
        # calc product of n_words normed probabilities for each n_options
        unique_normed_prob_per_sequence = normed_gen_probs.prod(-1)  # --> shape [n_options]
        # normalization
        scores = unique_normed_prob_per_sequence/ unique_normed_prob_per_sequence.sum()

        return unique_normed_prob_per_sequence, scores, unique_prob_per_sequence


    def predict(self, text, temp=TEMPERATURE, length=MAX_LEN):
        """
        For the gven text promp generate next words suggestions
        :param text: prompt text
        :param temp: temperature for suggestions samplings
        :param length: length of suggested phrase
        :return:
            best_option - input text + best suggested phrase
            best_delta - best suggested phrase
            opt[] - 3 top suggested phrases
        """
        # Tokenize the input string
        orig_text = text
        text = clean_text(text)
        if text == None or text == "":
            return "", "", ["", "", ""]
        text = trim_promt(text)
        input = self.tokenizer.encode(text, return_tensors="pt") #shape [1, n_tokens] like tensor([[ 40, 765, 284]]) "i want to"
        start = time()
        decoded_input = self.tokenizer.decode(input[0], skip_special_tokens=True)
        timing("---------  encoding time -----------", start)
        print("Input Clean [len/tokens]: " + str(len(text.split())) + "/" + str(input.shape[1]) + '**** ' + text)


        max_length = input.shape[1] + length  # input string length + suggestion max length
        min_length = input.shape[1] + MIN_LEN
        highest_score = 0.0
        best_option = ""
        best_delta = ""
        del_len = 0
        options = {}

        for config in self.search_config.keys():
            # get suggestion for each decode confifg type
            start1 = time()
            start = time()
            args = self.search_config[config]
            args['temperature'] = temp
            args['max_length'] = max_length
            args['min_length'] = min_length

            # Note: issue with transformers 4.3.  use 4.2
            # conda install -c huggingface transformers==4.2.0
            # generate suggestions
            outputs = self.model.generate(input, **args)
            timing("---------  generation time {} -----------".format(config), start)
            print("Output {}:\n".format(config) + 100 * '-')
            # calculate suggestions probabilities for comparison
            scores, scores_norm, raw_prob = self.get_scores(outputs, input.shape[-1]) # (outputs, n_tokens)

            start = time()
            for i, output in enumerate(outputs.sequences):
                # calculate and sort best sequences probabilities
                gen = self.tokenizer.decode(output, skip_special_tokens=True)
                generated_txt = clean_text(gen)
                delta = delta_text(decoded_input, generated_txt)
                delta_len = len(delta.split())
                # length validation
                s2 = text + " ---- " + generated_txt
                assert(delta_len < MAX_LEN + 1), s2
                print("{}: {} --> {}, {}".format(scores_norm[i], delta, scores[i], raw_prob[i]))
                # set returned options
                if (delta != "") and delta not in options:
                        options[delta] = (float(scores_norm[i]), self.search_config[config]["sort_order"])
                #calc best option
                if (delta != "") and (float(scores_norm[i]) > highest_score):
                    # note: if several config types are used, best option may not be included or first in the list
                    # since options are sorted first by the config type.
                    highest_score = float(scores_norm[i])
                    best_option = generated_txt
                    best_delta = delta
                    del_len = delta_len
            timing("---------  decoding time {} -----------".format(config), start)
            timing("---------  Total Predict time {} -----------".format(config), start1)

        # assert (best_delta != ""), decoded_input + '-->' + generated_txt

        # sort options: If more than one config types are used, sort first by 'sort_order' desc, and then by probs desc
        # (so higher config type's suggestions comes before the other type even if the second has higher probabilities
        opt = sorted(options, key=lambda k: (options[k][1], options[k][0]), reverse= True)
        num_of_options = len(opt)
        # add empty values if less than 3 options returned
        for i in range(3 - num_of_options):
            opt.append("")
        print("Best Delta:\n" + 50 * '@' + 'XXX  ' + best_delta + " len " + str(del_len))
        print("3 options:\n" + 50 * '@' + 'XXX  ' + ' '.join(map(str, opt[:3])))

        # options order validation
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
        if best_delta not in opt:
            print("WARNING: best delta not in opt: " + best_delta)
        return best_option, best_delta, opt[:3]


# for testing ------------------------------------------------------------------------------------------------------
def iterate_text(session_text):
    text = input("Enter text: \n")
    if text == "Q":
        # quit flag
        return "Q"
    elif text == "S":
        # start new input flag
        session_text = input("Enter text: \n")
    elif text != "R":
        # not repeat flag
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
    session_text = "I am writing to apply to be the farm manager of this local urban farm. I have been involved in growing food for the past ten years in some capacity or another even though it wasn't something that I studied in college. " \
                   "I did my first volunteer farm work while I was in college at an urban farm, and I was hired after that to fill in for someone that left their position. I have since then done work with various farms around the country. " \
                   "I have started my own small space to grow food with some friends on land that I own. I have helped restaurants with preparing their rooftop gardens to grow tomatoes and herbs. " \
                   "I have helped churches build community gardens in underserved communities. It's truly a passion of mine to work with the land and grow healthy food for the community. " \
                   "I have a fair amount of other skills as well that may be useful. I have some web development skills, and I have a long history of customer service experience that could be useful for fundraising or bringing local restaurants on board as customers. I appreciate you taking the time to read this."

    for i in range(2):
        session_text, delta, options = completer.predict(session_text)

    # text1 = """Hello, I am applying for Lead Web Developer at your company . I have been doing web development for just under 10 years now, and it's always something I've been very passionate about. I know the ins and outs of web development and have owned my own web development business for around 5 1/2 years. With that being said, I recently just left a job of 3 years due to some internal issues with my boss. I was the Marketing Director there and had several people under me. I often timed worked with the websites we had, but I would rather fully commit to a job that focuses on web development and am really hoping you will consider me. Thank you for your time and consideration"""
    # text2 = """Dear Hiring Manager,    As a long-time reader of Bella Books, your company has always held significance to me. Your books were some of the first interactions I had with a positive and open inclusion of lesbians in any form of media, and your material is always enjoyable and poignant. When I saw that your company was hiring, I was immediately interested in the possibility of using my skills in the environment and brand that you have created.   As the daughter of a librarian and a family of readers, a love for books and writing was introduced to me in my formative years. It has stayed with me over time, and I have used that passion to tailor my education, volunteer services, and freelance work.    During my time at UCF, I chose my degree in Interdisciplinary Studies, as opposed to a more traditional degree in English. I based this choice on the flexibility it offered, allowing me to enrich my capabilities and specialize in the disciplines of psychology, writing, and women’s studies. Through these subjects, I gained a highly educated level of editorial and writing skills. I have enhanced my skills in developing strong, multidimensional, and relatable characters. Additionally, I have developed extensive knowledge in understanding and creating engaging plots and emotional storytelling devices.   I have been editing fiction novels for nearly six years. This has included a substantial amount of content and copy editing, as well as some minor experience with the meticulous details of proofreading. My rapport with each author is comfortable, trusting, and involves open communication. I am proficient in conveying constructive criticism, allowing me to identify faults in writing and address those areas in a way that is productive and strengthens both the story and the writer. This has led to a continued professional relationship with each client.    I am excited about the possibility of working in an environment that would both challenge me and allow me the opportunity to use my skills on your behalf. I believe my background would allow me to be both successful in this position as well as devoted and eager to succeed.     If you have any concerns or questions about my resume or background, please feel free to contact me. Thank you for your time and consideration. I look forward to hearing from you"""
    # predict_doc(text1)




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


# min_length can be used to force the model to not produce an EOS token (= not finish the sentence) before min_length is reached. This is used quite frequently in summarization, but can be useful in general if the user wants to have longer outputs.