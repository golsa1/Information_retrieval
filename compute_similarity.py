import numpy as np
from scipy import spatial
from utils import open_json, save_file
from sklearn.metrics import average_precision_score



def entity_overlap0( entities, query_entities):
       intersection = 0
       union_len = len(entities) + len(query_entities)
       punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~''';

       for e11 in entities:
           for e22 in query_entities:

                       e1 = e11['uri']
                       e2 = e22['uri']
                       e1 = e1[e1.find("/", 20) + 1:].replace("_", " ").replace("%2e","")
                       e2 = e2[e2.find("/", 20) + 1:].replace("_", " ").replace("%2e","")

                       if e1==e2:
                           intersection +=1
                           break
       union_len = union_len - intersection
       if union_len==0:
           sim = 0
       else:
           sim = intersection/union_len
       return sim

def save_s_entity_overlap(articles, lang):

    n = 350
    s_entity_overlap = np.zeros([n, n])

    for event1 in articles.keys():
        l1 = len(articles[event1])
        for i in range(l1):
            article1 = articles[event1][i]
            entities = article1['nel']['wikifier']['spacy']
            for event2 in articles.keys():
                for j in range(len(articles[event2])):
                    print(i,j)
                    query = articles[event2][j]
                    query_entities = query['nel']['wikifier']['spacy']
                    s_entity_overlap[i, j] = entity_overlap0(entities, query_entities)
    np.save("similarity/" + lang + "/s_entity_overlap_new.npy", s_entity_overlap)
    print("done")


def convert_to_class( event, vector, selected_feature):
    sims = vector['similarity']
    # for s in sims:
    #     if s['article_id'] == vector['article_id']:
    #         del s
    scores = np.zeros(len(sims))
    ret = np.zeros(len(sims))

    for s in range(len(sims)):
            sim = sims[s]
        # if sim['article_id'] != vector['article_id']:
            if sim['event'] == event:
                ret[s] = 1
            if np.isnan(sim[selected_feature]):
                print("dffkjdfkdfhd")
            else:
                try:
                 scores[s] = sim[selected_feature]
                except Exception as e:
                    print("")

    return ret, scores


def compute_sims_combinations(whole_featurs, lang, name_postfix=''):
    s_scene = np.load("similarity/" + lang + "/s_scene.npy")
    s_object = np.load("similarity/" + lang + "/s_object.npy")
    s_location = np.load("similarity/" + lang + "/s_locaction.npy")
    s_bert = np.load("similarity/" + lang + "/s_bert.npy")
    s_entity_overlap = np.load("similarity/" + lang + "/s_entity_overlap.npy")

    dic_query_comparisons_with_the_rest = {}

    print("changing similarity saved format... ")
    for i in range(len(whole_featurs)):
        query = whole_featurs[i]
        avg_total = []

        for j in range(0, len(whole_featurs), 1):
            print(i, j)
            art = whole_featurs[j]
            sim_obj = s_object[i, j]
            sim_scene = s_scene[i, j]
            sim_loc = s_location[i, j]
            sim_bert = s_bert[i, j]
            sim_entity = s_entity_overlap[i, j]
            sim_avg_text = (sim_bert + sim_entity) / 2
            sim_avg_visual = (sim_obj + sim_loc + sim_scene) / 3
            sim_avg_total = (sim_bert + sim_entity + sim_obj + sim_loc + sim_scene) / 5

            # sim_max_text = np.maximum(sim_bert, sim_entity)
            # sim_max_obj_scene = np.maximum(sim_obj, sim_scene)
            # sim_max_visual = np.maximum(sim_max_obj_scene, sim_loc)
            # sim_max_total = np.maximum(sim_max_visual, sim_max_text)
            if np.isnan(sim_entity):
                print("dffkjdfkdfhd")
            avg_total.append({'article_id': art['article_id'], 'event': art['event'],
                              'sim_bert': sim_bert, 'sim_entity': sim_entity, 'sim_obj': sim_obj, 'sim_loc': sim_loc,
                              'sim_scene': sim_scene,
                              'sim_avg_text': sim_avg_text, 'sim_avg_visual': sim_avg_visual,
                              'sim_avg_total': sim_avg_total,
                              # 'sim_max_text': sim_max_text, 'sim_max_visual': sim_max_visual, 'sim_max_total': sim_max_total
                              })

        dic_query_comparisons_with_the_rest[query['event'] + "/" + query['article_id']] = {
            'article_id': query['article_id'], 'similarity': avg_total}
    file_name = lang + "_total_similarity_file"+name_postfix+".json"
    save_file("similarity/" + file_name, dic_query_comparisons_with_the_rest)
    return file_name


def save_sims_articles(whole_featurs, lang, is_bert=False, is_entity=False):
    n = len(whole_featurs)
    s_scene = np.zeros([n, n])
    s_object = np.zeros([n, n])
    s_location = np.zeros([n, n])
    s_bert = np.zeros([n, n])
    s_entity_overlap = np.zeros([n, n])
    black = ['486810751', '1436821318', '435220228']

    for i in range(len(whole_featurs)):
        for j in range(len(whole_featurs)):

            print(i, j)
            article_features = whole_featurs[i]['features']
            query_article_features = whole_featurs[j]['features']
            try:
                # s_scene[i, j] = 1 - spatial.distance.cosine(query_article_features['v_scene'], article_features['v_scene'])
                # s_object[i, j] = 1 - spatial.distance.cosine(query_article_features['v_object'], article_features['v_object'])
                # if whole_featurs[i]['article_id'] not in black:
                #     if whole_featurs[j]['article_id'] not in black:
                #          s_location[i, j] = 1 - spatial.distance.cosine(query_article_features['v_location'], article_features['v_location'])
                if is_bert:
                    a = np.array([np.asarray(aa) for aa in query_article_features['t_bert']])
                    b = np.array([np.asarray(bb) for bb in article_features['t_bert']])
                    s = spatial.distance.cdist(a, b, 'cosine')
                    s_bert[i, j] = 1 - np.mean(s)

                if is_entity:
                    s_entity_overlap[i, j] = 1 - spatial.distance.cosine(article_features['t_entity_overlap'], query_article_features['t_entity_overlap'])
                    if np.isnan(s_entity_overlap[i, j]):
                        s_entity_overlap[i, j] = 0

            except Exception as e:
                print(e)
                print("*********************")
                s_bert[i, j] = 0
                # s_entity_overlap[i, j] = 0
    # np.save("similarity/"+lang+"/s_scene.npy", s_scene)
    # np.save("similarity/"+lang+"/s_object.npy", s_object)
    # np.save("similarity/"+lang+"/s_locaction.npy", s_location)
    if is_bert:
        np.save("similarity/" + lang + "/s_bert.npy", s_bert)
    if is_entity:
        np.save("similarity/" + lang + "/s_entity_overlap.npy", s_entity_overlap)


def main():
   lang = "deu"
   docs_path = "data/nel_"+lang+".json"
   events = open_json(docs_path).keys()
   # save_sim_entity_overlap(articles, lang)


# ************************* save similarities

   # whole_featurs = np.load("features/" + lang + "_features.npy", allow_pickle=True)  # last saved file
   # save_sims_articles(whole_featurs, lang, is_bert=True, is_entity=False)
   # total_sims_file_name = compute_sims_combinations(whole_featurs, lang, name_postfix='')

# *************** compute average prec of retrievals            e.g. 'Barack'{ '2323223', 'similarity': {'Barack','8734645', 0.8 } }
   total_sims_file_name = lang+"_total_similarity_file.json"
   dic_sims_articles = open_json("similarity/"+ total_sims_file_name)
   keys = dic_sims_articles.keys()
   avg_event = {}
   ls_features = ['sim_bert', 'sim_entity', 'sim_obj', 'sim_loc', 'sim_scene', 'sim_avg_text', 'sim_avg_visual', 'sim_avg_total'
                   # 'sim_max_text', 'sim_max_visual', 'sim_max_total'
                  ]
   for selected_feature in ls_features:
       print("*********   "+selected_feature)
       for event in events:
           print(event)
           queries_by_event = []
           for k0 in keys:
               k = k0.split("/")[0]
               if event == k:
                   queries_by_event.append(dic_sims_articles[k0])   # collects all queries of this event
           avg = []
           for query, i in zip(queries_by_event, range(len(queries_by_event))):
               y_true0, scores0 = convert_to_class(event, query, selected_feature)
               y_true = np.zeros([len(y_true0)-1])
               y_true[:i] = y_true0[:i]
               y_true[i:] = y_true0[i+1:]
               scores = np.zeros([len(y_true0)-1])
               scores[:i] = scores0[:i]
               scores[i:] = scores0[i + 1:]
               avg.append(average_precision_score(y_true, scores))
           avg_event[event] = np.mean(avg)
       save_file("similarity/avgs/"+lang+"/"+selected_feature, avg_event)

   print('')


if __name__=='__main__':
   main()

