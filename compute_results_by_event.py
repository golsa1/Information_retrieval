from utils import *
import numpy as np
from os import listdir
from os.path import isfile, join


lang = "deu"
main_avgs = open_json("similarity/avgs/"+lang+"_main_avgs_file")
politics = ['2016_United_States_presidential_election','Impeachment_of_Donald_Trump','War_in_Donbass','European_Union–Turkey_relations','Brexit','Cyprus–Turkey_maritime_zones_dispute']
environment = ['Global_warming','Water_scarcity','2019–20_Australian_bushfire_season','Indonesian_tsunami','Water_scarcity_in_Africa','2018_California_wildfires','Palm_oil_production_in_Indonesia']
sports=['2016_Summer_Olympics','2022_FIFA_World_Cup','2018_FIFA_World_Cup','2020_Summer_Olympics']
health=['Coronavirus','Ebola_virus_disease','Zika_fever','Avian_influenza','Swine_influenza']
economy=['Financial_crisis_of_2007–08','Greek_government-debt_crisis','Volkswagen_emissions_scandal']
events = main_avgs.keys()
politics_avgs = []
environment_avgs=[]
sports_avgs=[]
health_avgs=[]
economy_avgs=[]

for av in events:
    for p in politics:
        if p==av:
         main_avgs_p = main_avgs[p].split(',')
         eiur = [np.float(ma) for ma in main_avgs_p]
         politics_avgs.append(np.split(main_avgs[p], "'"))
    for p in environment:
        if p == av:
            environment_avgs.append(main_avgs[p])
    for p in sports:
        if p==av:
            sports_avgs.append(main_avgs[p])
    for p in health:
        if p==av:
            health_avgs.append(main_avgs[p])
    for p in economy:
        if p==av:
            economy_avgs.append(main_avgs[p])

politics_avg = np.mean(politics_avgs)
environment_avg = np.mean(environment_avgs)
sports_avg = np.mean(sports_avgs)
health_avg = np.mean(health_avgs)
economy_avg = np.mean(economy_avgs)
print('')