import numpy as np
from src import parameters as params
from scipy.stats import truncnorm

""" Global variables, specified in parameter file """
N = params.nrOfSimulatedPeople
C = params.nrOfCharacteristics
M = params.nrOfMicroskillsNormal
Q = params.nrOfMicroskillsIQ
T = params.nrOfTestOccasions
F = params.nrOfFactors

np.random.seed(41)
# TODO: move numbers to parameters file + structure? + concentration/cog_cap index as param?


def create_personality_matrix(twin_type):
    """
    Matrix that codes the value of person i on characteristic c (cognitive capacity and concentration) (N x C)
    """

    if twin_type == 'none' or twin_type == 'diz':
        cog_cap = truncnorm.rvs(a=0, b=np.inf, loc=params.PERS_MEAN_COP_CAP, scale=params.PERS_SD_COG_CAP, size=(N, 1))
        conc = truncnorm.rvs(a=0, b=np.inf, loc=params.PERS_MEAN_CONC, scale=params.PERS_SD_CONC, size=(N, 1))

    if twin_type == 'mono':
        #initiate cog cap: size is N/2, and then duplicate, so that every twin is exactly the same
        cog_cap = truncnorm.rvs(a=0, b=np.inf, loc=params.PERS_MEAN_COP_CAP, scale=params.PERS_SD_COG_CAP, size=(int(N/2), 1))
        #repeats every value and inserts it after the value (so 1,2,3 becomes 1,1,2,2,3,3)
        cog_cap = np.repeat(cog_cap, 2, axis=0)
        
        conc = truncnorm.rvs(a=0, b=np.inf, loc=params.PERS_MEAN_CONC, scale=params.PERS_SD_CONC, size=(int(N/2), 1))
        conc = np.repeat(conc, 2, axis=0)

    if twin_type == 'diz': # for every twin, take half of cogcap of twin, plus half of own (random) cogcap, and divide by two. Same for conc
        cog_cap[1::2] = (cog_cap[::2] + cog_cap[1::2])/2
        conc[1::2] = (conc[::2] + conc[1::2])/2
        
    personality = np.hstack((cog_cap, conc))
    
    return personality


def create_knowledge_matrix(personality_matrix):
    """
    Matrix that codes how each factor f loads onto each microskill m (M x F) F0=cog_cap
    """

    #get the maximum cog cap of the group
    max_cog_cap_personality = np.max(personality_matrix[:, 0])
    
    #how many microskills are in first years (to set 0 cog_cap_required)
    first_years = int(params.YEARS_ZERO_COG_CAP * (M / params.TOTAL_YEARS_OF_SIMULATION))

    #create for every microskill (from year 2), an ascending required cog cap, from 0 to hardest.
    mean = np.linspace(0, max_cog_cap_personality * params.PERC_OF_MAX_COG_CAP, M - first_years) 
    # number of microskills except first years
    time = np.linspace(0, M-first_years, M-first_years)
    #add sinusoid
    sinusoid = np.sin(time / (M / params.TOTAL_YEARS_OF_SIMULATION) * (2 * np.pi))
    conv_mean = mean + 0.5 * np.multiply(mean, sinusoid) 

    cog_cap_first_years = np.zeros((first_years, 1))
    cog_cap_last_years = [truncnorm.rvs(a=0, b=np.inf, loc=u, scale=params.KNOW_SD_COG_CAP, size=1) for u in conv_mean] #add noise 
    cog_cap = np.vstack((cog_cap_first_years, cog_cap_last_years))
    
    other = truncnorm.rvs(a=0, b=np.inf, loc=params.KNOW_MEAN, scale=params.KNOW_SD, size=(M, F - 1)) # build factors
    
    knowledge = np.hstack((cog_cap, other))
  
    return knowledge


def create_test_matrix(knowledge_matrix):
    """
    Matrix that codes how each factor f loads onto each microskill m (Q x F)
    """

    cog_cap = knowledge_matrix[:, 0] # all cog caps of skills
    TOTAL_YEARS = params.TOTAL_YEARS_OF_SIMULATION
    test_types = {'child':  np.array([10, 11, 12, 13, 14], dtype=int), 'adult': np.array([18, 19, 20, 21, 22], dtype=int)}
    part_matrix = {}
    chosen_skills=[]

    for type, age in test_types.items():
        
        # Select microskills on last 5 peaks and last 5 valleys of cognitive capacity
        max_sine = (T / TOTAL_YEARS) * age + ((T / TOTAL_YEARS) / 4)
        min_sine = (T / TOTAL_YEARS) * age + ((T / TOTAL_YEARS) / 4) * 3
        peak_valley_skills = np.concatenate((max_sine.astype(int), min_sine.astype(int)))
        chosen_skills.append(peak_valley_skills) # save for later (debugging.py)
        
        # Permutate factor values except cog_cap
        factors_without_cog_cap = knowledge_matrix[peak_valley_skills, 1:] #all 9 factors on chosen microskills
        factors_permuted = cog_cap[peak_valley_skills, np.newaxis] # cog cap of chosen microskills
        for column in factors_without_cog_cap.T: #for every skill:
            column_permuted = np.random.permutation(column) #randomly shuffle the factors through skills
            factors_permuted = np.hstack((factors_permuted, np.expand_dims(column_permuted, axis=1))) # put it back in non transposed matrix

        # Copy items with randomly distributed noise: make from 10-matrix a 100-matrix
        rest_skills_without_noise = np.tile(factors_permuted, [9, 1])  # Tile: repeat ten skills another x (=9) times. So [1,2,3] becomes [1,2,3,1,2,3]
        noise = np.random.normal(0, .1, rest_skills_without_noise.shape) 
        rest_skills_with_noise = rest_skills_without_noise + noise

        part_matrix[type] = np.vstack((factors_permuted, rest_skills_with_noise))
    

    test_matrix = np.vstack((part_matrix['child'], part_matrix['adult']))

    return test_matrix, chosen_skills

def create_schooling_array():
    """
    List that codes which microskill m is offered at which time step t. This matrix is created for every person i (M)
    """

    skill_sample_age = params.SKILLS_TO_SAMPLE_FROM_PER_AGE #40
    periods = params.PERIODS 
    all_perc_rand = params.PERC_RAND
    TOTAL_YEARS = params.TOTAL_YEARS_OF_SIMULATION #25

    time_steps_per_year = int(T / TOTAL_YEARS) #40
    schooling_array = []

    for i in range(TOTAL_YEARS): #per year, chose percentage randommess and replacement y/n

        if i < periods['first_period']:
            perc_rand = all_perc_rand['first_period']
            replace = True 
        elif i >= periods['first_period'] and i < periods['second_period']:
            perc_rand = all_perc_rand['second_period']
            replace = True 

        elif i >= periods['second_period'] and i < periods['third_period']:
            perc_rand = all_perc_rand['third_period']
            replace = False 

        elif i >= periods['third_period']:
            perc_rand = all_perc_rand['fourth_period']
            replace = True 
        

        #total skills to sample from random. Example: 1 year=40 time steps. 75% random means 0.75*40=30 skills to sample         
        sample_size_all_skills = np.round((time_steps_per_year * perc_rand), 0).astype(int)
        #inverse. So total skills per year minus sample_size_all_skills. 
        sample_size_skills_age = np.round((time_steps_per_year * (1 - perc_rand)), 0).astype(int)

        #sample the random skills in the whole M.
        random = np.random.choice(np.arange(M),
                                  size=sample_size_all_skills,
                                  replace=True)

        # Sample from skills associated with age
        fitting_for_period = np.random.choice(
            np.arange(i * skill_sample_age, (i * skill_sample_age + skill_sample_age), dtype=int),
            size=sample_size_skills_age,
            replace=replace) 

        #shuffle
        selected_skills = np.append(random, fitting_for_period)
        np.random.shuffle(selected_skills)
        schooling_array.extend(selected_skills)
        

    return schooling_array
