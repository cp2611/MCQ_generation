import streamlit as st
import pandas as pd
import numpy as np
import requests
import os

import neuralcoref
import spacy

from PIL import Image
image = Image.open(r"H:\udemy_c++\Chandraprakash Koshle.png")
width, height = image.size
left = width//8
top = height//6
right = width-width//6
bottom = height-height//6
image = image.rotate(-4.5)
image = image.crop((left,top,right,bottom))
image = image.resize((width//13,height//13),Image.ANTIALIAS)
st.title("Exam Lounge MCQ Generation with NLP")
st.text("""This work is a part of nlp deep learning teamwork at Exam Lounge Summer Internship 2021""")
c1,c2, = st.beta_columns((3,1))
c1.header("Chandraprakash Koshle")
c1.text("Data Science Intern")
c1.text("Department of Mechanical Engineering")
c1.text("Indian Institute of Technology Kharagpur")
c1.text("17 May 2021 - 18 July 2021")

c1.markdown('email :' '<a href="mailto:chandraprakash.iitkgp@gmail.com">chandraprakash.iitkgp@gmail.com</a>', unsafe_allow_html=True)
c2.image(image)
st.markdown("<h1 style='text-align: center; color: navy;'>Input text</h1>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Add text file !")
if uploaded_file is not None:
    text_input = uploaded_file.getvalue()
    text_input = text_input.decode("utf-8",errors='ignore')
    st.write(text_input)

text_input2 = st.text_area("Type a text to anonymize")
if text_input2:
    st.write(text_input2)


nlp = spacy.load('en')
neuralcoref.add_to_pipe(nlp)

from tqdm import tqdm
import re
from collections import Counter
from word2number import w2n
import random
from dateutil import parser as date_parser
from collections import namedtuple
import os
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
import torch
import math
from transformers import BertModel, BertConfig, BertPreTrainedModel, BertTokenizer

class BertWSD(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

        self.ranking_linear = torch.nn.Linear(config.hidden_size, 1)

        self.init_weights()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_dir =r"H:\Internship_6_Exam_Lounge\bert_base-augmented-batch_size=128-lr=2e-5-max_gloss=6"


model = BertWSD.from_pretrained(model_dir)
tokenizer = BertTokenizer.from_pretrained(model_dir)
if '[TGT]' not in tokenizer.additional_special_tokens:
    tokenizer.add_special_tokens({'additional_special_tokens': ['[TGT]']})
    assert '[TGT]' in tokenizer.additional_special_tokens
    model.resize_token_embeddings(len(tokenizer))
    
model.to(DEVICE)

def get_distractors_wordnet(syn,word):
    distractors=[]
    word= word.lower()
    orig_word = word
    if len(word.split())>0:
        word = word.replace(" ","_")
    hypernym = syn.hypernyms()
    if len(hypernym) == 0: 
        return distractors
    for item in hypernym[0].hyponyms():
        name = item.lemmas()[0].name()
        #print ("name ",name, " word",orig_word)
        if name == orig_word:
            continue
        name = name.replace("_"," ")
        name = " ".join(w.capitalize() for w in name.split())
        if name is not None and name not in distractors:
            distractors.append(name)
    return distractors
GlossSelectionRecord = namedtuple("GlossSelectionRecord", ["guid", "sentence", "sense_keys", "glosses", "targets"])
BertInput = namedtuple("BertInput", ["input_ids", "input_mask", "segment_ids", "label_id"])



def _create_features_from_records(records, max_seq_length, tokenizer, cls_token_at_end=False, pad_on_left=False,
                                  cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                  sequence_a_segment_id=0, sequence_b_segment_id=1,
                                  cls_token_segment_id=1, pad_token_segment_id=0,
                                  mask_padding_with_zero=True, disable_progress_bar=False):
    features = []
    for record in tqdm(records, disable=disable_progress_bar):
        tokens_a = tokenizer.tokenize(record.sentence)

        sequences = [(gloss, 1 if i in record.targets else 0) for i, gloss in enumerate(record.glosses)]

        pairs = []
        for seq, label in sequences:
            tokens_b = tokenizer.tokenize(seq)

            tokens = tokens_a + [sep_token]
            segment_ids = [sequence_a_segment_id] * len(tokens)

            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

            if cls_token_at_end:
                tokens = tokens + [cls_token]
                segment_ids = segment_ids + [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                segment_ids = [cls_token_segment_id] + segment_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            pairs.append(
                BertInput(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_id=label)
            )

        features.append(pairs)

    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

import re
import torch
#from tabulate import tabulate
from torch.nn.functional import softmax
from tqdm import tqdm
from transformers import BertTokenizer
import time


MAX_SEQ_LENGTH = 128

def get_sense(sent):
  re_result = re.search(r"\[TGT\](.*)\[TGT\]", sent)
  ambiguous_word = re_result.group(1).strip()

  results = dict()

  wn_pos = wn.NOUN
  for i, synset in enumerate(set(wn.synsets(ambiguous_word, pos=wn_pos))):
      results[synset] =  synset.definition()

  if len(results) ==0:
    return (None,None,ambiguous_word)

  sense_keys=[]
  definitions=[]
  for sense_key, definition in results.items():
      sense_keys.append(sense_key)
      definitions.append(definition)


  record = GlossSelectionRecord("test", sent, sense_keys, definitions, [-1])

  features = _create_features_from_records([record], MAX_SEQ_LENGTH, tokenizer,
                                            cls_token=tokenizer.cls_token,
                                            sep_token=tokenizer.sep_token,
                                            cls_token_segment_id=1,
                                            pad_token_segment_id=0,
                                            disable_progress_bar=True)[0]

  with torch.no_grad():
      logits = torch.zeros(len(definitions), dtype=torch.double).to(DEVICE)
      for i, bert_input in list(enumerate(features)):
          logits[i] = model.ranking_linear(
              model.bert(
                  input_ids=torch.tensor(bert_input.input_ids, dtype=torch.long).unsqueeze(0).to(DEVICE),
                  attention_mask=torch.tensor(bert_input.input_mask, dtype=torch.long).unsqueeze(0).to(DEVICE),
                  token_type_ids=torch.tensor(bert_input.segment_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
              )[1]
          )
      scores = softmax(logits, dim=0)
      preds = (sorted(zip(sense_keys, definitions, scores), key=lambda x: x[-1], reverse=True))
  sense = preds[0][0]
  meaning = preds[0][1]
  return (sense,meaning,ambiguous_word)

from transformers import T5ForConditionalGeneration,T5Tokenizer

question_model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_squad_v1')
question_tokenizer = T5Tokenizer.from_pretrained('t5-base')

def get_question(sentence,answer):
  text = "context: {} answer: {} </s>".format(sentence,answer)
  max_len = 256
  encoding = question_tokenizer.encode_plus(text,max_length=max_len, pad_to_max_length=True, return_tensors="pt")

  input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

  outs = question_model.generate(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  early_stopping=True,
                                  num_beams=5,
                                  num_return_sequences=1,
                                  no_repeat_ngram_size=2,
                                  max_length=200)


  dec = [question_tokenizer.decode(ids) for ids in outs]


  Question = dec[0].replace("question:","")
  Question= Question.strip()
  return Question
def getMCQs(sent):
  sentence_for_bert = sent.replace("**"," [TGT] ")
  sentence_for_bert = " ".join(sentence_for_bert.split())
  # try:
  sense,meaning,answer = get_sense(sentence_for_bert)
  if sense is not None:
    distractors = get_distractors_wordnet(sense,answer)
  else: 
    distractors = ["Word not found in Wordnet. So unable to extract distractors."]
  sentence_for_T5 = sent.replace("**"," ")
  sentence_for_T5 = " ".join(sentence_for_T5.split()) 
  ques = get_question(sentence_for_T5,answer)
  return ques,answer,distractors,meaning


def format_special_characters(text):
    pat = '[ ]'+r'[^a-zA-z0-9<>]'+'[ ]*'
    return re.findall(pat,text)

def inputs_and_dictionary(doc,dictionary=None):
  inputs = []
  if dictionary == None:
    dictionary = {}
    dictionary["Person"]=[]
    dictionary["Date"]=[]
    dictionary["Place"]=[]
    dictionary["Org"]=[]
  items = [x.text for x in doc.ents]
  for sent in doc.sents:
    entities = sent.ents
    temp = []
    sent_input= []
    for token in sent:
      if (token.text.lower() in ["he", "she", "it", "him", "they","them","this","i","we"]) and len(token._.coref_clusters):
        temp.append("<>".join(list(token._.coref_clusters[0][0].text)))
      elif  ( token.text.lower() in ["his", "her", "its","their","my","our"]) and (len(token._.coref_clusters))>0 and (token._.coref_clusters[0][0].text.lower() not in ["he", "she", "it", "him", "they","them","this"]):
        temp.append("<>".join(list(token._.coref_clusters[0][0].text)) + "'s")
      else :
        if (token.text.lower()=="am"):
          temp.append("is")
        else:
          temp.append(token.text)
      

    temp = ' '.join(temp)
    to_replace = format_special_characters(temp)
    for char in to_replace:
      temp = temp.replace(char,char.strip())
    if len(entities):
      for x in entities:
        temp1 = temp
        temp1 = temp1.replace(x.text,'**'+x.text+'**')
        temp1 = temp1.replace('<>','')
        if ("**" not in temp1):
          temp2 = temp
          temp2 = temp2.replace(x.text.replace(", ",","),'**'+x.text+'**')
          temp1 = temp2.replace('<>','')
        if ("**" not in temp1):
          print(temp1)
          print(x)
          print(x.text)
          print(x.text in temp1)     
        if (x.label_ == 'PERSON'):
          id = 'Person'
        elif (x.label_ == 'DATE'):
          id = 'Date'
        elif (x.label_ == 'GPE'):
          id = 'Place'
        elif (x.label_ == 'ORG'):
          id = 'Org'
        else:
          id = None
        if id :
          dictionary[id].append(x.text)
        if ("**" in temp1):
          input= [temp1,id if id else x.label_,x.text,Counter(items)[x.text]]
          sent_input.append(input)
      random.shuffle(sent_input)
      flag={}
      for (a,b,c,d) in sent_input:
        if (b=="CARDINAL"):
          try:
            cemo = w2n.word_to_num(c)
          except:
            cemo = 0
        if (b != "CARDINAL") or (b == "CARDINAL" and cemo >3):
          if b not in flag.keys():
            flag[b]=1
          if (b=='Date'):
            if any(chr.isdigit() for chr in x.text):
              inputs.append([a,b,c])
          else:
            if d < 2 :
              if flag[b]==1:
                inputs.append([a,b,c])
                flag[b]=0
  return dictionary, inputs

import random
def generate_mcqs(doc,inputs):
  j=1
  tracker = 1
  st.write(doc.text)
  st.write('\n')
  for i in inputs:
    sentence = i[0]
    exceed_cardinal=0
    if (i[1]!= "Date") or ((i[1]== "Date") and (any(chr.isdigit() for chr in i[2]))):
      question,answer,distractors,meaning = getMCQs(sentence)
      new_distractors =[]
      if i[1] in dictionary.keys():
        new_distractors = dictionary[i[1]].copy()
        random.shuffle(new_distractors)
        new_distractors.remove(i[2])
        if (i[1] == "Date"):
          try:
            date_distractors = generate_date(parse_date_(i[2]))
            k = [j for j in i[2].replace(","," ").split(" ") if len(j)!=0]

            if (i[2].isdigit()):
              date_distractors=[i[2] for i in date_distractors]
            elif (len(k)==2):
              if (len(k[0])==2):
                date_distractors = [' '.join(i[:2]) for i in date_distractors]
              else:
                date_distractors = [' '.join(i[2:]) for i in date_distractors]
            elif (i[2].split(" ")[1].lower() in ['year','years'] ):
              date_distractors = [random.choice([i for i in list(range(int(i[2].split(" ")[0])-5,int(i[2].split(" ")[0])+5)) if i!=int(i[2].split(" ")[0])]) for j in range(2)]
              date_distractors = [" ".join([str(k),i[2].split(" ")[1]]) for k in date_distractors]
            else:
              date_distractors=[' '.join(i) for i in date_distractors]
          except:
            if (i[2][-1]=='s' and i[2][-2]=='0'):
              date_distractors = [i[2][:-3]+str(int(i[2][-3])+1)+i[2][-2:],i[2][:-3]+str(int(i[2][-3])-1)+i[2][-2:]]
            else:
              date_distractors=[]
          date_distractors.extend(new_distractors)
          new_distractors = date_distractors
      elif (i[1]=="CARDINAL"):
        try:
          number = w2n.word_to_num(i[2])
          if number<5:
            cardinal_distractors = [1,2,3,4]
            cardinal_distractors = [c for c in np.unique(cardinal_distractors) if c!=number]
          
          elif number <10:
            cardinal_distractors = [5,6,7,8,9,10,11,12,13,14,15]
            cardinal_distractors = [c for c in np.unique(cardinal_distractors) if c!=number]     
          elif number<600:
            cardinal_distractors = [number+i*(1 if random.random() <0.5 else -1) for i in range(8)]
            cardinal_distractors = [c for c in np.unique(cardinal_distractors) if c>0.8*number and c!=number]
          else:
            exceed_cardinal = 1
          random.shuffle(cardinal_distractors)
          cardinal_distractors=cardinal_distractors[:4]
        except:
          cardinal_distractors=[]
        cardinal_distractors.extend(new_distractors)
        new_distractors = cardinal_distractors

      if (len(new_distractors)>=4 or len(distractors)>=4) and (exceed_cardinal==0):
        st.write('Que   %d :%s\n' % (j, question))
        st.write('Options :')
        for k in range(4):  
          if len(new_distractors)>=4:
            st.write('(%d) %s ' % (k+1,new_distractors[k] if k<len(new_distractors) else '----'))
          elif 'Wordnet' not in distractors[0]:
            st.write('(%d) %s ' % (k+1,distractors[k] if k<len(distractors) else '----'))
        st.write('\n    Ans :%s\n' % answer)
        print(tracker)
        tracker=tracker+1
        
        j=j+1
st.write("Hi")
#if text_input:
st.write("text uploaded")
doc = nlp(text_input)
dictionary,inputs = inputs_and_dictionary(doc)
generate_mcqs(doc,inputs)
