import numpy as np
import matplotlib.pyplot as plt
import math

s = [-5,-4,-3,-2,-1,0,1,2,3,4,5]
k = [-5,-4,-3,-2,-1,0,1,2,3,4,5,6]
kfrequency = np.zeros(len(k))
hfrequency = np.zeros(len(k))

iteratingstrategiesfrequency = np.zeros(len(k))

class Simulation:
	def __init__(self, population, rounds, benefit, cost, generations, mutation, observers, useH):
		self.population = population
		self.mutation = mutation
		self.rounds = rounds
		self.observers = observers
		self.b = benefit
		self.c = cost
		if(useH == True):
			self.h = np.random.randint(k[0], k[len(k)-1]+1, population)
			self.hkmatrix = np.zeros((len(k), len(k)))
		else:
			self.h = None
			self.hkmatrix = None
		self.generations = generations
		self.strategiesOverGenerations = np.zeros(generations)
		self.payoffsOverGenerations = np.zeros(generations)
		self.strategiesfrequency = np.zeros(len(k))
		self.imageScores = np.zeros(population)
		self.imageScoresObservers = np.zeros((population, population))
		self.payoffs = np.zeros(population)
		self.payoffsprobabilities = np.zeros(population)
		self.cooperators = np.random.randint(k[0], k[len(k)//2-1]+1, population//2)
		self.defectors = np.random.randint(k[len(k)//2-1]+1, k[len(k)-1]+1, population//2)
		self.strategies = np.zeros(population)
		for i in range(0, len(self.cooperators)):
			self.strategies[i] = self.cooperators[i]
			self.strategies[len(self.defectors)+i] = self.defectors[i]
		
	def chooseAgents(self):
		donor = self.chooseRandomAgent()
		recipient = self.chooseRandomAgent()
		while(donor == recipient):
			recipient = self.chooseRandomAgent()
		return donor, recipient
		
	def chooseRandomAgent(self):
		return np.random.randint(0, self.population)
	
	def isCooperative(self, donor, recipient):
		if(isinstance(self.h, (np.ndarray, np.generic))):
			# return ... and/or ...
			return self.imageScores[recipient] >= self.strategies[donor] and self.imageScores[donor] < self.h[donor]
		else:
			return self.imageScores[recipient] >= self.strategies[donor]
		
	def isCooperativewObservers(self, donor, recipient):
		if(isinstance(self.h, (np.ndarray, np.generic))):
			# return ... and/or ...
			return self.imageScoresObservers[donor][recipient] >= self.strategies[donor] and self.imageScoresObservers[donor][donor] < self.h[donor]
		else:
			return self.imageScoresObservers[donor][recipient] >= self.strategies[donor]
			
	def cooperate(self, donor, recipient):
		self.imageScores[donor] = min(self.imageScores[donor] + 1, s[-1])
		self.payoffs[donor] -= self.c
		self.payoffs[recipient] += self.b
		
	def cooperatewObservers(self, donor, recipient):
		for i in range(0, self.observers):
			observer = self.chooseRandomAgent()
			self.imageScoresObservers[observer][donor] = min(self.imageScoresObservers[observer][donor] + 1, s[-1])
		self.imageScoresObservers[recipient][donor] = min(self.imageScoresObservers[recipient][donor] + 1, s[-1])
		self.payoffs[donor] -= self.c
		self.payoffs[recipient] += self.b
	
	def defect(self, donor, recipient):
		self.imageScores[donor] = max(self.imageScores[donor] - 1, s[0])
		
	def defectwObservers(self, donor, recipient):
		for i in range(0, self.observers):
			observer = self.chooseRandomAgent()
			self.imageScoresObservers[observer][donor] = max(self.imageScoresObservers[observer][donor] - 1, s[0])
		self.imageScoresObservers[recipient][donor] = max(self.imageScoresObservers[recipient][donor] - 1, s[0])
		
	def increasePayoffs(self):
		for i in range(0, len(self.payoffs)):
			self.payoffs[i] += self.c
		
	def generatePairsAndAct(self):
		for i in range(0,self.rounds,1):
			donor, recipient = self.chooseAgents()
			if(self.isCooperative(donor, recipient)):
				self.cooperate(donor, recipient)
			else:
				self.defect(donor, recipient)
			self.increasePayoffs()
			
	def generatePairsAndActwObservers(self):
		for i in range(0,self.rounds,1):
			donor, recipient = self.chooseAgents()
			if(self.isCooperativewObservers(donor, recipient)):
				self.cooperatewObservers(donor, recipient)
			else:
				self.defectwObservers(donor, recipient)
			self.increasePayoffs()
			
	def normalizePayoffs(self):
		for i in range(0, len(self.payoffs)):
			self.payoffsprobabilities[i] = self.payoffs[i] / (self.rounds*self.c)
			
	def selectAgents(self):
		sortedPayoffArgs = self.payoffs.argsort()
		sortedPayoff = sorted(self.payoffs, key=int, reverse=True)
		'''# Used for imitating individual with probability that increases with fitness difference
		# p = fB − fA / ΔfMAX
		fMax = sortedPayoff[0]
		fMin = sortedPayoff[-1]
		normFactor = 1/(fMax - fMin)
		if(normFactor < 0):
			normFactor = 0
		
		# Type 1 - Imitate a random individual with a probability that increases with the fitness difference
		for i in range(0, self.population):
			agentA, agentB = self.chooseAgents()
			probSelection = normFactor * (self.payoffs[agentB] - self.payoffs[agentA])
			if(np.random.rand() > probSelection):
				self.strategies[agentA] = self.strategies[agentB]'''
				
		''' Type 2 - Proportional to payoff '''
		totalPayoffs = sum(self.payoffs)
		kfrequency = np.zeros(len(k))
		
		# Gathers frequency of each strategy based on payoff: Higher payoffs will increase strategy frequency
		for i in range(0, len(self.payoffs)):
			if(np.random.rand() >= (1-self.mutation)):
				# Mutate and disobey father
				randomStrategy = np.random.randint(k[0], k[len(k)-1]+1)
				kfrequency[int(randomStrategy)+5] += (self.payoffs[i]/totalPayoffs)*self.population
				continue
			kfrequency[int(self.strategies[i])+5] += (self.payoffs[i]/totalPayoffs)*self.population
		
		# For each strategy, create a childs to match frequencies
		count = 0
		for i in range(0, len(kfrequency)):
			for j in range(0, int(kfrequency[i])):
				if(isinstance(self.h, (np.ndarray, np.generic))):
					self.h[count] = self.h[i]
				self.strategies[count] = k[i]
				count += 1
		
		# In case there are childs left to match the population size, gather the ones with higher payoffs and imitate
		if(count < self.population):
			for i in range(count, self.population):
				if(np.random.rand() >= (1-self.mutation)):
					# Mutate and disobey father
					randomStrategy = np.random.randint(k[0], k[len(k)-1]+1)
					self.strategies[i] = randomStrategy
					continue
				self.strategies[i] = self.strategies[sortedPayoffArgs[len(sortedPayoffArgs) -1 - (i-count)]]
				if(isinstance(self.h, (np.ndarray, np.generic))):
					self.h[i] = self.h[sortedPayoffArgs[len(sortedPayoffArgs)-1-(i-count)]]
		
	def resetAgents(self):
		self.payoffs = np.zeros(self.population)
		self.imageScores = np.zeros(self.population)
		
	def resetAgentswObservers(self):
		self.payoffs = np.zeros(self.population)
		self.imageScoresObservers = np.zeros((self.population, self.population))
		
	def printInformation(self):
		print("Payoffs")
		print(self.payoffs)
		print("Sum of payoffs: "+str(sum(self.payoffs)))
		print("Payoffs Probabilities")
		print(self.payoffsprobabilities)
		print("Sum of payoff probabilities: "+str(sum(self.payoffsprobabilities)))
		print("Image Scores")
		print(self.imageScores)
		print("Image Scores with observers")
		print(self.imageScoresObservers)
		print("Strategies")
		print(self.strategies)
		print("Strategies Frequencies")
		print(self.strategiesfrequency)
		print("H Values")
		print(self.h)
		
	def updateFrequencies(self, generation):
		self.payoffsOverGenerations[generation] = np.mean(self.payoffs)
		self.strategiesOverGenerations[generation] = np.mean(self.strategies)
		for i in range(0, len(self.strategies)):
			if(isinstance(self.h, (np.ndarray, np.generic))):
				self.hkmatrix[int(self.strategies[i])+5][int(self.h[i])+5] += 1
			self.strategiesfrequency[int(self.strategies[i])+5] += 1/self.population
		
	def run(self):
		print("Running simulation without observers")
		for generation in range(0,self.generations,1):
			self.generatePairsAndAct()
			self.normalizePayoffs()
			self.printInformation()
			self.selectAgents()
			if(generation==0 or generation==10 or generation==20 or generation == 150):
				self.plotStrategies(generation)
			self.updateFrequencies(generation)
			self.resetAgents()
		self.plotPayoffsOverGenerations()
		self.plotStrategiesOverGenerations()
		for i in range(0, len(self.strategiesfrequency)):
			self.strategiesfrequency[i] = self.strategiesfrequency[i]/self.generations
		self.plotStrategiesFrequency()
		
	def runwObservers(self):
		print("Running simulation with observers")
		for generation in range(0,self.generations,1):
			self.generatePairsAndActwObservers()
			self.normalizePayoffs()
			self.printInformation()
			self.selectAgents()
			if(generation==0 or generation==10 or generation==20 or generation == 150):
				self.plotStrategies(generation)
			self.updateFrequencies(generation)
			self.resetAgentswObservers()
		self.plotPayoffsOverGenerations()
		self.plotStrategiesOverGenerations()
		for i in range(0, len(self.strategiesfrequency)):
			self.strategiesfrequency[i] = self.strategiesfrequency[i]/self.generations
		self.plotStrategiesFrequency()
		return self.getStrategiesFrequency()
			
	def plotStrategies(self, generation):
		plt.hist(self.strategies, bins=k)
		plt.ylabel("Frequency")
		plt.xlabel("Strategy")
		plt.title("t="+str(generation)+";n="+str(self.population))
		plt.show()
		
	def plotPayoffsOverGenerations(self):
		plt.scatter(range(0, self.generations), self.payoffsOverGenerations)
		plt.ylabel("Payoff")
		plt.xlabel("Time (in generations)")
		plt.title("n="+str(self.population)+";generations="+str(self.generations))
		plt.show()
		
	def plotStrategiesOverGenerations(self):
		plt.scatter(range(0, self.generations), self.strategiesOverGenerations)
		plt.ylabel("Strategy")
		plt.xlabel("Time (in generations)")
		plt.title("n="+str(self.population)+";generations="+str(self.generations))
		plt.show()
		
	def plotStrategiesFrequency(self):
		plt.scatter(range(-5, 7), self.strategiesfrequency)
		plt.ylabel("Frequency")
		plt.xlabel("Strategy")
		plt.title("n="+str(self.population))
		plt.show()
	
	def getHKMatrix(self):
		return self.hkmatrix

	def getStrategiesFrequency(self):
		return self.strategiesfrequency
		

def parameterizeStrategiesFrequenciesValues(val):
	for i in range(0, len(iteratingstrategiesfrequency)):
		iteratingstrategiesfrequency[i] = iteratingstrategiesfrequency[i]/val
			
def fillIteratingStrategiesFrequency(strategiesfrequency):
	for i in range(0, len(strategiesfrequency)):
		iteratingstrategiesfrequency[i] += strategiesfrequency[i]
	
def plotStrategiesFrequencyIterating(val, population):
	parameterizeStrategiesFrequenciesValues(val)
	plt.scatter(range(-5, 7), iteratingstrategiesfrequency)
	plt.ylabel("Frequency")
	plt.xlabel("Strategy")
	plt.title("n="+str(population))
	plt.show()
	
def plotHKMatrix(matrix):
	plt.matshow(matrix, cmap='hot')
	plt.ylabel("k")
	plt.xlabel("h")
	plt.show()

		
def main():
	''' Parameters to change '''
	benefit=1
	cost=0.1
	h=False #False if we dont want to use h; True to use h value.
	mutation=0.001
	generations = 200
	population = 100
	observers = 10
	rounds = 125
	sampling=10**2
	
	''' Simulation with perfect information '''
	if(h):
		rounds = 500
		totalHKmatrix = np.zeros((len(k),len(k)))
		for i in range(0, sampling):
			sim = Simulation(population, rounds, benefit, cost, generations, mutation, observers, h)
			sim.run()
			matrix = sim.getHKMatrix()
			totalHKmatrix += matrix
		plotHKMatrix(totalHKmatrix)
	else:
		sim = Simulation(population, rounds, benefit, cost, generations, mutation, observers, h)
		sim.run()
		
	''' Simulation with observers '''
	if(h):
		rounds = 500
		population = 20
		totalHKmatrix = np.zeros((len(k),len(k)))
		for i in range(0, sampling):
			sim = Simulation(population, rounds, benefit, cost, generations, mutation, observers, h)
			sim.run()
			matrix = sim.getHKMatrix()
			totalHKmatrix += matrix
		plotHKMatrix(totalHKmatrix)
	else:
		sim2 = Simulation(population, rounds, benefit, cost, generations, mutation, observers, h)
		sim2.runwObservers()

	# Uncomment for Diferences in population size in simulation with observers 
	population = 20
	
	strategiesfrequency = np.zeros(len(k))
	rounds = 10*population
	for i in range(0, sampling):
		sim2 = Simulation(population, rounds, benefit, cost, generations, mutation, observers, h)
		strategyfrequency = sim2.runwObservers()
		fillIteratingStrategiesFrequency(strategyfrequency)
	plotStrategiesFrequencyIterating(sampling, population)
	
	
	return 0
	
main()