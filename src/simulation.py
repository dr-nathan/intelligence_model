import numpy as np
from src import parameters as params, matrices
from scipy.stats import truncnorm

"""Global variables, specified in parameter file."""
N = params.nrOfSimulatedPeople
C = params.nrOfCharacteristics
M = params.nrOfMicroskillsNormal
Q = params.nrOfMicroskillsIQ
T = params.nrOfTestOccasions
F = params.nrOfFactors
TOTAL_YEARS = params.TOTAL_YEARS_OF_SIMULATION

np.random.seed(41)

class Simulation:

    def __init__(self):
        self.personality_matrix = matrices.create_personality_matrix(params.PERS_TWIN)
        self.knowledge_matrix = matrices.create_knowledge_matrix(self.personality_matrix) #M x F
        self.test_matrix, self.chosen_skills = matrices.create_test_matrix(self.knowledge_matrix) # 2*Q x F
        self.allskills_matrix=np.vstack((self.knowledge_matrix, self.test_matrix)) # matrix with all skills and factors, test included
        self.microskill_similarity_matrix = self.allskills_matrix.dot(self.allskills_matrix.T) #get similarity of every skill and test skill
        self.learning_matrix = np.zeros((T + (Q * len(params.TEST_AGES)), N), dtype=bool) #1400x100. Represents amount of microskills learned (T/F). But does not index microskills! For example, [934,1] is not microskill number 934, but the microskill presented at timestep number 934. But for the tests, it does index correctly. Ex: test skill 23 will be in postion 1023
        self.achievement_matrix = np.zeros((N, M), dtype=bool) # 100x1000. Indexes microskill, and wether it is learned or not (T/F)
        self.schooling_matrix = np.zeros((N, M), dtype=int) #100x1000. shows wich skill is presented at which timestep. 
        self.concentration_matrix = np.zeros((T, N))
        self.cog_cap_matrix = np.zeros((T, N))
        self.available_cogcap_matrix=np.zeros((T, N))
        self.acquired_knowledge_matrix=np.zeros((N,(T + (Q * len(params.TEST_AGES))))) # this matrix remembers how much acquired knowledge was present by the learnign fo every skill. useful for debugging purposes. Q*2 because you have 100 esay and 100 difficult tests.

    def run(self):
        """ Create schooling array for every person (take into account twins). Save it in schooling matrix, and update achievement and learning matrix for every person """

        percentage = params.PERC_SIMILAR_TWINS

        for person in range(params.nrOfPersInTest):
            schooling_array = matrices.create_schooling_array()  # Create schooling array for person n
            self.schooling_matrix[person, :] = schooling_array  # Save schooling array for every person for use in tests.py
            # when twins, get uneven persons 
            if params.PERS_TWIN != 'none' and (person % 2) != 0:
                #choose 80% of total schooling moments randomly,
                x = np.random.choice(len(schooling_array), size=(int(len(schooling_array) * percentage)), replace=False)
                #replace the skills at these indexes with the ones at the same index for the person-1 (the twin)
                self.schooling_matrix[person, x] = self.schooling_matrix[person-1, x]
                
            self.update(person, schooling_array)  # Update achievement matrix for person n: calculate for every time step if it has been learned, and do the IQ tests
            

    def update(self, person: int, schooling_array: list):
        """ Update achievement and learning matrix for person n for every timestep t """

        cog_cap = self.get_cog_cap(person)  # Get cognitive capacity for person at every timestep (array)
        self.cog_cap_matrix[:, person] = cog_cap  # Save cognitive capacity so this can be easily accessed in tests.py
        test_timesteps = np.multiply((T / TOTAL_YEARS), params.TEST_AGES) # test times are first learning moment of test age

        for timestep in range(T):
            microskill = schooling_array[timestep]  # Get microskill that is offered at this timestep

            # If the microskill is learned set this in achievement and learning matrix to true
            if self.is_learned(person, microskill, timestep, cog_cap):
                self.achievement_matrix[person, microskill] = True
                self.learning_matrix[timestep, person] = True # Note: learning_matrix is of size 1400x100, but tests are at the end so this works out
            # If timestep is test age, take test
            if timestep in test_timesteps.astype(int):
                self.take_test(person, timestep, cog_cap, test_timesteps)
             

        return

    def take_test(self, person: int, overall_timestep: int, cog_cap: np.ndarray, test_timesteps:np.ndarray):
        """ Take IQ test. Fills in the results at the end of the learning matrix """
        #function gets the relevant 100 slots in the learning matrix, calls is_learned_testversion 100 times and updates learning matrix
        test_index_in_learning_matrix = int(np.where(test_timesteps == overall_timestep)[0][0] * Q) #gives number of test occasion. So first test will yield 0, second 100, third 200 etc

        for microskill in range(Q):
            test_timestep = T + test_index_in_learning_matrix + microskill # go through all 100 test microskills of that particular test. (tests are located at the end of the learning matrix)
            if self.is_learned_testversion(person, microskill, overall_timestep, cog_cap):
                self.learning_matrix[test_timestep, person] = True
                

    def is_learned_testversion(self, person: int, microskill: int, timepoint: int, cog_cap: np.ndarray):
        """ Check whether person n was able to learn the microskill m """

        if timepoint > (T / TOTAL_YEARS) * params.DIFFICULT_TEST_AGE:
            microskill = microskill + Q  # first 100 in test_matrix are simple test, so add 100 for difficult test
        req_cog_cap = self.test_matrix[microskill, 0] #look up test in test matrix.
        concentration = self.personality_matrix[person, 1] # get max conc for person. Concentration is max during a test, so we dont call get_concentration

        acquired_know = self.get_acquired_knowledge(person, microskill+M) # +1000 because test index in similarity_matrix are index 1000-1200
        self.acquired_knowledge_matrix[person, microskill+M]=acquired_know  # save it in matrix
        cog_cap = cog_cap[timepoint]

        total_req_cog_cap = req_cog_cap - (acquired_know * params.ACQ_KNOWL_WEIGHT)
        avail_cog_cap = cog_cap * concentration

        return total_req_cog_cap < avail_cog_cap
        

    def is_learned(self, person: int, microskill: int, timepoint: int, cog_cap: np.ndarray):
        """ Check whether person n was able to learn the microskill m """


        req_cog_cap = self.knowledge_matrix[microskill, 0]
        concentration = self.get_concentration(person) # generate semi random concentration at this timestep
        self.concentration_matrix[timepoint, person] = concentration #save it to check it out later

        acquired_know = self.get_acquired_knowledge(person, microskill) 
        self.acquired_knowledge_matrix[person, microskill]=acquired_know # it is ordered through time, so use timpoint as x index, not skill
        cog_cap = cog_cap[timepoint]

        total_req_cog_cap = req_cog_cap - (acquired_know * params.ACQ_KNOWL_WEIGHT) 
        avail_cog_cap = cog_cap * concentration
        self.available_cogcap_matrix[timepoint, person] = avail_cog_cap # for plotting later

        return total_req_cog_cap < avail_cog_cap
    #TODO: idea: plot req cog cap and available cog cap through life of someone and check it out
        

    def get_cog_cap(self, person: int):
        """ Get the cognitive capacity of person n at time t (cog_cap changes as a function of age) """

        x_max_cog_cap = (T / TOTAL_YEARS) * params.PEAK_YEAR_COG_CAP  #time steps until peak
        y_max_cog_cap = self.personality_matrix[person, 0] #max cog cap
        x = np.arange(T) #total time steps
        start_perc_cog_cap = params.START_PERC_COG_CAP

        #create inverse parabole with above parameters. For a visualization of cog cap through life, head to debugging.py
        a = (-y_max_cog_cap) / np.power(x_max_cog_cap, 2) 
        y = ((a * np.power((x - x_max_cog_cap), 2)) + y_max_cog_cap) * (1 - start_perc_cog_cap) + (start_perc_cog_cap * y_max_cog_cap) 
        
        return y

    def get_concentration(self, person: int):
        """ Get concentration of person n, random noise is added """

        max_concentration = self.personality_matrix[person, 1]
        rand_noise = truncnorm.rvs(a=0, b=np.inf, loc=params.MEAN_CONC_NOISE, scale=params.SD_CONC_NOISE)
        concentration = float(max_concentration - rand_noise)

        concentration = 0 if concentration < 0 else concentration
        
            

        return concentration

    def get_acquired_knowledge(self, person: int, microskill: int):
        """ Get sum of already acquired microskills similar to microskil m """
        acquired_microskills = np.argwhere(self.achievement_matrix[person, :] > 0) # get all skills that are set to true in the learning matrix(learned)

        if len(acquired_microskills) == 0:
            return int(0)
        
        return sum(self.microskill_similarity_matrix[microskill, acquired_microskills]) # for every learned skill, get the similiraty to the skill at hand, and sum it.
        #should maybe be normalized! Right now we get the sum, and choose to weigh them by 0.001. But more insight would be nice #TODO: plot to see range of values
        #register all acq know: 100* 1400 values. too much. Plot acq. knowledge for a handful of people? Ok lets try. 