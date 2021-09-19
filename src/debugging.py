from src import parameters as params
import matplotlib.pyplot as plt
import numpy as np

Q = params.nrOfMicroskillsIQ

class Debugging:

    
    def __init__(self, simulation):
        self.simulation = simulation

    def run(self):
        
        #Plot cognitive capacity to get insight in similarity between different twin types, and also to get an idea of cog cap prgression through life
        for person in range(8): # only take first 8 people
            cog_cap = self.simulation.get_cog_cap(person)
            plt.plot(np.arange(params.nrOfMicroskillsNormal), cog_cap,
                            label='Person (p=' + str(person) + ') Cognitive Capacity')
        plt.title(params.PERS_TWIN)
        plt.ylabel('Cognitive Capacity')
        plt.xlabel('Time')
        plt.legend()
        plt.show()
        
        #show generated distribution of cog cap. To see spread and shape of cog cap of people
        max_smartness=np.max(self.simulation.personality_matrix[:, 0]).round(2) # get cog cap of the smartest person
        plt.hist(self.simulation.personality_matrix[:, 0], bins=20)
        plt.title('cog cap distribution. Max:{}'.format(max_smartness))
        plt.xlabel('cog cap')
        plt.ylabel('frequency')
        plt.show()

        
        #plot an example of concentration progression through life of a person   
        plt.plot(self.simulation.concentration_matrix[:, 16])
        plt.title('concentration throughout life of person 16')
        plt.show()
        #same but for available cog cap (cog cap * concentration)        
        plt.plot(self.simulation.available_cogcap_matrix[:, 16])
        plt.title('Available cog cap throughout life of person 16')
        plt.show()
        
        
        #plot chosen cog cap for test matrix, child and adult: visualize which req cog cap has been chosen to create test skills
        req_cog_cap = self.simulation.knowledge_matrix[:, 0]
        plt.plot(req_cog_cap)
        peak_valley_skills=self.simulation.chosen_skills
        colors=['go', 'ro']
        for i in range(2):
            plt.plot(peak_valley_skills[i], req_cog_cap[peak_valley_skills[i]], colors[i]) # child
        plt.title('chosen skills. green=child, red=adult')
        plt.show()
         #TODO: peaks and valleys could be picked a bit more precisely! 
         
        #insgiht into value of acquired knowledge. Only for a single person
        plt.plot(self.simulation.acquired_knowledge_matrix[16,:])
        plt.title('weight of acquired knoweldge for the learning of every skill')
        plt.ylabel('weight of already acquired knowledge')
        plt.xlabel('timepoint at which skill is being learned')
        running_avg=[] # also plot trend, to see if acq. know goes up on average.
        for i in range(20, (len(self.simulation.acquired_knowledge_matrix[16,:])-20)): # average acq know weight across 40 timepoints
            running_avg.append(np.mean(self.simulation.acquired_knowledge_matrix[16,i-20:i+20]))
        plt.plot(range(20, (len(self.simulation.acquired_knowledge_matrix[16,:])-20)), running_avg)
        plt.show()                  