from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from nltk import tokenize
# from scene_classification.scene_embedding import SceneClassificator
# from object_classification.object_embedding import ObjectDetector
# from location_verification.location_embedding import GeoEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import os
from utils import open_json, save_file
import glob
import numpy as np
from scipy import spatial
import json
from flair.data import Sentence
from flair.embeddings import FlairEmbeddings, BertEmbeddings
from pathlib import Path

class get_features:

   def __init__(self):
       model_path_sc = "C:/Users/TahmasebzadehG/PycharmProjects/InfoRetrievalresources/scene_classification/resnet50_places365.pth.tar"
       hierarchy = "C:/Users/TahmasebzadehG/PycharmProjects/InfoRetrievalresources/scene_classification/scene_hierarchy_places365.csv"
       labels = "C:/Users/TahmasebzadehG/PycharmProjects/InfoRetrievalresources/scene_classification/categories_places365.txt"
       self.scence_obj = SceneClassificator(model_path=model_path_sc, labels_file=labels, hierarchy_file=hierarchy)

       model_path_obj = "C:/Users/TahmasebzadehG/PycharmProjects/InfoRetrievalresources/object_classification/resnet50_weights_tf_dim_ordering_tf_kernels.h5"
       self.object_obj = ObjectDetector(model_path=model_path_obj)

       model_path_loc = "C:/Users/TahmasebzadehG/PycharmProjects/InfoRetrievalresources/geolocation_estimation/base_M/"
       self.location_obj = GeoEstimator(model_path_loc, use_cpu=True)


       self.flair_forward_embedding = FlairEmbeddings('multi-forward')
       self.flair_backward_embedding = FlairEmbeddings('multi-backward')
       self.bert_embedding = BertEmbeddings('bert-base-multilingual-cased')


   def save_only_bert_flair_new_version(self,name_whole_features, articles,lang):

       whole_features = np.load("features/" + name_whole_features,allow_pickle=True)
       events = articles.keys()
       indplus = 0

       for event in events:
           docs = articles[event]

           for doc in docs:
               text = doc['body'][:1500]
               id = doc['uri']
               for f in whole_features:
                   if f['article_id'] == id:
                       features = f

               sentences = tokenize.sent_tokenize(text)
               len_sent = len(sentences)
               vec_embs = []

               for ind in range(len_sent):

                   if len(sentences[ind]) > 0:
                       if ind == len_sent - 1:
                           text = sentences[ind]
                       elif ind == len_sent - 2:
                           text = sentences[ind] + " " + sentences[ind + 1]
                       else:
                           text = sentences[ind] + " " + sentences[ind + 1] + " " + sentences[ind + 2]

                       sentence = Sentence(text)

                       self.bert_embedding.embed(sentence)
                       avg_vector = np.zeros(768)
                       counter = 0
                       # now check out the embedded tokens.
                       for token in sentence:
                           vector = token.embedding.cpu().detach().numpy()
                           avg_vector = np.add(avg_vector, vector[0:768])
                           counter += 1
                       avg_vector /= counter
                       vec_embs.append(avg_vector)

               features['features']['t_bert'] = vec_embs
               indplus += 1
               print("******************   " + str(indplus))
           # np.save("features/" + lang + "features_bert_flair.npy", whole_features)

       np.save("features/" + lang + "features_bert.npy", whole_features)
       print("features bert saved successfully!!!")


   def tf_idf(self, docs):
       tfidf = TfidfVectorizer()
       x = tfidf.fit_transform(docs)
       # tfidf = pd.DataFrame(x.toarray(), columns=tfidf.get_feature_names())
       return x.toarray()


   def preprocess_named_link(self, entity):

       e = entity[entity.find("/", 20) + 1:].replace("_", " ").replace("%2e", "")
       w = e.find("wiki/")
       if w>0:
          e = e[e.find("wiki/")+5:]


       return e.lower()


   def postprocess_entities_main_vector(self, vector):
       punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~''';
       for ind, v in zip(range(len(vector)), vector):
           if len(v)<1:
               vector.remove(v)
           for s in v:
               if s in punctuations:
                   v = v.replace(s,"")
               vector[ind] = v

       return vector


   def entities_main_vector(self, articles0, nel_tool, ner_tool):
       vector = []
       events = articles0.keys()
       for event in events:
           articles = articles0[event]
           for art in articles:
               for ent in art['nel'][nel_tool][ner_tool]:

                   if ent not in vector:
                       vector.append(ent)

       return vector


   def map_entities_to_main_vector(self, entities, vector):

       lev = len(vector)
       one_hot = np.zeros(lev)

       for ind in range(lev):
               e1 = vector[ind]
               for ent in entities:

                   if ent == vector[ind]:
                       one_hot[ind] = 1
                       break
                   else:
                       one_hot[ind] = 0


       return one_hot


   def save_only_bert(self, name_whole_features, articles,lang):


       whole_features = np.load("features/" + name_whole_features,allow_pickle=True)
       tokenizer = self.tokenizer_bert
       events = articles.keys()
       indplus = 0
       e3 = []

       for event in events:
           docs = articles[event]
           indplus_ = 0
           for doc in docs:

               text = doc['body'][:70]
               id = doc['uri']
               for f in whole_features:
                   if f['article_id'] == id:
                       features = f

               sentences = tokenize.sent_tokenize(text)
               len_sent = len(sentences)

               vec_embs = []
               temp_embeding = torch.zeros(768)

               for ind in range(len_sent):
                   if len(sentences[ind]) > 0:
                       if ind == len_sent - 1:
                           text = sentences[ind]
                       elif ind == len_sent - 2:
                           text = sentences[ind] + " " + sentences[ind + 1]
                       else:
                           text = sentences[ind] + " " + sentences[ind + 1] + " " + sentences[ind + 2]

                       tokenized_text = tokenizer.tokenize("[CLS] " + text + " [SEP]")
                       indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
                       segments_ids = [1] * len(tokenized_text)
                       tokens_tensor = torch.tensor([indexed_tokens])
                       segments_tensors = torch.tensor([segments_ids])
                       self.bert_model.eval()

                       indplus_ += 1
                       # print(indplus_)

                       try:
                           with torch.no_grad():
                               encoded_layers, _ = self.bert_model(tokens_tensor, segments_tensors)
                           e2 = encoded_layers[-1][:][:]
                           e3 = torch.mean(e2, 1)
                           vec_embs.append(e3)
                       except Exception as e:
                           print(e)
                           vec_embs.append(e3)
                   else:
                       vec_embs.append(temp_embeding)
               features['features']['t_bert'] = vec_embs
               indplus += 1
               print("******************   " + str(indplus))
           np.save("features/"+lang+"features_bert_______.npy", whole_features)

       for wh in whole_features:
           whf = wh['features']['t_bert']

           for ind in range(len(whf)):
               if len(whf[ind]) < 1:
                   whf[ind] = tb_1
               tb_1 = whf[ind]

       np.save("features/"+lang+"features_bert.npy", whole_features)
       print("features bert saved successfully!!!")


   def save_only_entity_overlap(self, name_whole_features, articles,lang):

       all_entities_vector = self.entities_main_vector(articles, 'wikifier', 'spacy')
       np.save("features/main_vector_"+lang, all_entities_vector)

       whole_features = np.load("features/"+name_whole_features,allow_pickle=True)
       events = articles.keys()

       indplus = 0
       for event in events:
           docs = articles[event]

           for doc in docs:
               id = doc['uri']
               features = ""
               for f in whole_features:
                   if f['article_id']==id:
                       features = f
                       print("not    ")

               t_entity_overlap = self.map_entities_to_main_vector(doc['nel']['wikifier']['spacy'], all_entities_vector)
               if features!="":
                  features['features']['t_entity_overlap'] = t_entity_overlap
               indplus += 1
               print("******************   "+str(indplus))

       np.save("features/"+lang+"features_entity_____.npy", whole_features)

       print("features entity overlap saved successfully!!!")


   def compute_sim_entity_method0(self, entities, query_entities):
       intersection = 0
       union_len = len(entities) + len(query_entities)

       for e11 in entities:
           for e22 in query_entities:

                       e1 = e11['uri']
                       e2 = e22['uri']
                       e1 = e1[e1.find("/", 20) + 1:].replace("_", " ").replace("%2e","")
                       e2 = e2[e2.find("/", 20) + 1:].replace("_", " ").replace("%2e","")
                       # for s in e1:
                       #     if s in punctuations:
                       #         e1 = e1.replace(s, "")
                       # for s in e2:
                       #     if s in punctuations:
                       #         e2 = e2.replace(s, "")

                       if len(e1)> len(e2):
                           bigger = e1
                           smaller = e2
                       else:
                           bigger = e2
                           smaller = e1

                       n_similar_chars = 0

                       for ind in range(len(smaller)):
                               s = smaller[ind].lower()
                               b = bigger[ind].lower()
                               if s == b:
                                   n_similar_chars += 1
                       iou_char = n_similar_chars/len(bigger)

                       if iou_char >= 0.8:
                           intersection += 1
       union_len = union_len - intersection
       sim = intersection/union_len
       return sim


   def save_only_location_feature(self, pre_path, name_whole_features,lang,image_format):
       whole_features = np.load("features/"+name_whole_features,allow_pickle=True)

       for wf in whole_features:
           img = pre_path+wf['article_id']+"."+image_format

           # try:
           location_features = self.location_obj.get_img_embedding(img)
           # except Exception as e:
           #     np.zeros([])

           if len(location_features)<1:
               print(wf['article_id'])
           wf['features']['v_location'] = location_features
           print(location_features.shape)

       np.save("features/"+lang+"features_location.npy", whole_features)

       print("features location saved successfully!!!")


   def save_features_obj_scene(self, articles, image_path, image_names, img_format,lang):
       if img_format =="png":
           till = -4
       else:
           till = -5
       whole_features = []
       events = articles.keys()

       for event in events:
           print("event "+event+" ...")
           docs = articles[event]
           for i in range(len(docs)):  # img_name == article_id
               img = ""
               for image_n, image in zip(image_names, glob.glob(image_path + "*."+img_format)):
                   o=image_n[0:till]
                   if image_n[0:till] == docs[i]['uri']:
                       img = image
                       break
               if img!="":
       # IMAGE
                   print("article " + str(i) + " ...")
                   v_scene = self.scence_obj.get_img_embedding(img)
                   v_object = self.object_obj.get_img_embedding(img)
                   v_location = []
                   # v_location = self.location_obj.get_img_embedding(img)
       # TEXT
                   article_id = docs[i]['uri']
                   t_bert = []
                   t_entity_overlap = []
                   features_article = {'v_scene':v_scene, 'v_object':v_object, 'v_location':v_location, 't_bert': t_bert, 't_entity_overlap':t_entity_overlap}
                   dic = {'article_id': article_id, 'event': event, 'features': features_article}
                   whole_features.append(dic)
       np.save("features/"+lang+"features_obj_scene.npy", whole_features)
       print("features obj and scene saved successfully!!!")


   def convert_to_class(self, event, vector):

       sims = vector['similarity']
       scores = np.zeros(len(sims))
       ret = np.zeros(len(sims))

       for s in range(len(sims)):
           sim = sims[s]
           if sim['event'] == event:
               ret[s] = 1
           scores[s] = sim['average']
       return ret, scores


   def get_feats(self, list_features, name_saved_features, articles_with_nel, images_path, image_names, lang,image_format):

        for feature in list_features:
            if feature=='scene_obj':
                if not os.path.exists("features/"+lang+"features_obj_scene.npy"):
                    self.save_features_obj_scene(articles_with_nel, images_path, image_names, image_format,lang)
                    name_saved_features = lang+"features_obj_scene.npy"
            if feature=='bert':
                if not os.path.exists("features/"+lang+"features_bert.npy"):
                    self.save_only_bert_flair_new_version( name_saved_features, articles_with_nel,lang)
                    name_saved_features = lang+"features_bert.npy"
            if feature=='entity':
                if not os.path.exists("features/"+lang+"features_entity_new.npy"):
                     self.save_only_entity_overlap(name_saved_features, articles_with_nel,lang)
                     name_saved_features = lang+"features_entity.npy"
            if feature=='location':
                if not os.path.exists("features/"+lang+"features_location.npy"):
                    self.save_only_location_feature(images_path, name_saved_features,lang,image_format)
                    name_saved_features = lang+"features_location.npy"
        return name_saved_features


def main():

   lang = "eng"
   #  eng: whole_features_with_entity.npy
   docs_path = "data/nel_"+lang+".json"
   articles_with_nel = open_json(docs_path)
   images_path = "data/collected_images/"
   image_names = [img[img.find("/", 74) + 1:] for img in glob.glob(images_path + "*.png")]
   obj_get_f = get_features()
   ls_features = [['scene_obj', 'bert', 'entity', 'location'][2]]
   name_saved_features = obj_get_f.get_feats(ls_features, "eng_features.npy", articles_with_nel, images_path, image_names, lang, "jpeg")
   # name_saved_features = "deu_features.npy"
   # whole_featurs = np.load("features/" + name_saved_features,allow_pickle=True)
   print("features saved!!!")






if __name__=='__main__':
   main()

