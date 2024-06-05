from collections import namedtuple
import copy
from typing import Callable

import numpy as np
import statistics
from numpy.random import random as randu
from numpy.random import randint as randi
from numpy.random import choice as randc
from numpy.random import shuffle
import time, inspect
from copy import deepcopy
from joblib.parallel import Parallel, delayed

from genepro.node import Node
from genepro.variation import *

from genepro.selection import tournament_selection
import torch
import torch.optim as optim
import torch.nn as nn

class Evolution:
  """
  Class concerning the overall evolution process.

  Parameters
  ----------
  fitness_function : function
    the function used to evaluate the quality of evolving trees, should take a Node and return a float; higher fitness is better

  internal_nodes : list
    list of Node objects to be used as internal nodes for the trees (e.g., [Plus(), Minus(), ...])

  leaf_nodes : list
    list of Node objects to be used as leaf nodes for the trees (e.g., [Feature(0), Feature(1), Constant(), ...])

  pop_size : int, optional
    the population size (default is 256)

  init_max_depth : int, optional
    the maximal depth trees can have at initialization (default is 4)

  max_tree_size : int, optional
    the maximal number of nodes trees can have during the entire evolution (default is 64)

  crossovers : list, optional
    list of dictionaries that contain: "fun": crossover functions to be called, "rate": rate of applying crossover, "kwargs" (optional): kwargs for the chosen crossover function (default is [{"fun":subtree_crossover, "rate": 0.75}])

  mutations : list, optional
    similar to `crossovers`, but for mutation (default is [{"fun":subtree_mutation, "rate": 0.75}])

  coeff_opts : list, optional
    similar to `crossovers`, but for coefficient optimization (default is [{"fun":coeff_mutation, "rate": 1.0}])
  
  selection : dict, optional
    dictionary that contains: "fun": function to be used to select promising parents, "kwargs": kwargs for the chosen selection function (default is {"fun":tournament_selection,"kwargs":{"tournament_size":4}})

  max_evals : int, optional
    termination criterion based on a maximum number of fitness function evaluations being reached (default is None)

  max_gens : int, optional
    termination criterion based on a maximum number of generations being reached (default is 100)

  max_time: int, optional
    termination criterion based on a maximum runtime being reached (default is None)

  n_jobs : int, optional
    number of jobs to use for parallelism (default is 4)

  verbose : bool, optional
    whether to log information during the evolution (default is False)

  Attributes
  ----------
  All of the parameters, plus the following:

  population : list
    list of Node objects that are the root of the trees being evolved

  num_gens : int
    number of generations

  num_evals : int
    number of evaluations

  start_time : time
    start time

  elapsed_time : time
    elapsed time

  best_of_gens : list
    list containing the best-found tree in each generation; note that the entry at index 0 is the best at initialization
  """
  def __init__(self,
    # required settings
    fitness_function : Callable[[Node], float],
    internal_nodes : list,
    leaf_nodes : list,
    # optional evolution settings
    n_trees : int,
    pop_size : int=256,
    init_max_depth : int=4,
    max_tree_size : int=64,
    crossovers : list=[{"fun":subtree_crossover, "rate": 0.5}],
    mutations : list= [{"fun":subtree_mutation, "rate": 0.5}],
    coeff_opts : list = [{"fun":coeff_mutation, "rate": 0.5}],
    selection : dict={"fun":tournament_selection,"kwargs":{"tournament_size":8}},
    elitism : float = 0.1,
    # termination criteria
    max_evals : int=None,
    max_gens : int=100,
    max_time : int=None,
    # other
    n_jobs : int=4,
    verbose : bool=False,
    ):

    # set parameters as attributes
    _, _, _, values = inspect.getargvalues(inspect.currentframe())
    values.pop('self')
    for arg, val in values.items():
      setattr(self, arg, val)

    # fill-in empty kwargs if absent in crossovers, mutations, coeff_opts
    for variation_list in [crossovers, mutations, coeff_opts]:
      for i in range(len(variation_list)):
        if "kwargs" not in variation_list[i]:
          variation_list[i]["kwargs"] = dict()
    # same for selection
    if "kwargs" not in selection:
      selection["kwargs"] = dict()

    # initialize some state variables
    self.population = list()
    self.num_gens = 0
    self.num_evals = 0
    self.start_time, self.elapsed_time = 0, 0
    self.best_of_gens = list()
    self.bestFitnesses = list()
    self.bestFitnesses.append(-1000)
    self.bestFitnesses.append(-700)
    self.memory = None


  def _must_terminate(self) -> bool:
    """
    Determines whether a termination criterion has been reached

    Returns
    -------
    bool
      True if a termination criterion is met, else False
    """
    self.elapsed_time = time.time() - self.start_time
    if self.max_time and self.elapsed_time >= self.max_time:
      return True
    elif self.max_evals and self.num_evals >= self.max_evals:
      return True
    elif self.max_gens and self.num_gens >= self.max_gens:
      return True
    return False

  def _initialize_population(self):
    """
    Generates a random initial population and evaluates it
    """
    # initialize the population
    self.population = Parallel(n_jobs=self.n_jobs)(
        delayed(generate_random_multitree)(self.n_trees, 
          self.internal_nodes, self.leaf_nodes, max_depth=self.init_max_depth )
        for _ in range(self.pop_size))

    for count, individual in enumerate(self.population):
      individual.get_readable_repr()
    
    # evaluate the trees and store their fitness
    
    
    self.initSeeds()
    fitnesses = Parallel(n_jobs=self.n_jobs)(delayed(self.fitness_function)(t) for t in self.population)
    fitnesses = list(map(list, zip(*fitnesses)))
    memories = fitnesses[1]
    memory = memories[0]
    for m in range(1,len(memories)):
      memory += memories[m]

    self.memory = memory

    fitnesses = fitnesses[0]

    for i in range(self.pop_size):
      self.population[i].fitness = fitnesses[i]
    # store eval cost
    self.num_evals += self.pop_size
    # store best at initialization
    best = self.population[np.argmax([t.fitness for t in self.population])]
    
    #
    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
    batch_size = 128
    GAMMA = 0.99

    constants = best.get_subtrees_consts()

    if len(constants)>0:
      optimizer = optim.AdamW(constants, lr=1e-3, amsgrad=True)

    for _ in range(500):

      if len(constants)>0 and len(self.memory)>batch_size:
        target_tree = copy.deepcopy(best)

        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), dtype=torch.bool)

        non_final_next_states = torch.cat([s for s in batch.next_state
                                                  if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = best.get_output_pt(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(batch_size, dtype=torch.float)
        with torch.no_grad():
          next_state_values[non_final_mask] = target_tree.get_output_pt(non_final_next_states).max(1)[0].float()

        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
      
        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(constants, 100)
        optimizer.step()
    #
    
    
    self.best_of_gens.append(deepcopy(best))

  def _perform_generation(self):
    """
    Performs one generation, which consists of parent selection, offspring generation, and fitness evaluation
    """
    # select promising parents
    sel_fun = self.selection["fun"]
    parents = sel_fun(self.population, self.pop_size, **self.selection["kwargs"])
    
    elite_count = int(0.1 * self.pop_size)
    elites = sorted(self.population, key=lambda x: x.fitness, reverse=True)[:elite_count]
    
    # generate offspring
    offspring_population = Parallel(n_jobs=self.n_jobs)(delayed(generate_offspring)
      (t, self.crossovers, self.mutations, self.coeff_opts, 
      parents, self.internal_nodes, self.leaf_nodes,
      constraints={"max_tree_size": self.max_tree_size}) 
      for t in parents)


  
    # evaluate each offspring and store its fitness
    if(self.population[np.argmax([t.fitness for t in self.population])].fitness>200 and statistics.mean([ind.fitness for ind in self.population]) > 150):
      self.initSeeds(True)
    else:
      self.initSeeds()

    #for elites only
    
    fitnesses = Parallel(n_jobs=self.n_jobs)(delayed(self.fitness_function)(t) for t in elites)
    fitnesses = list(map(list, zip(*fitnesses)))

    memories = fitnesses[1]
    memory = memories[0]

    fitnesses = fitnesses[0]

    for i in range(elite_count):
      elites[i].fitness = fitnesses[i]
    
    #for elites end
    
    
    fitnesses = Parallel(n_jobs=self.n_jobs)(delayed(self.fitness_function)(t) for t in offspring_population)
    fitnesses = list(map(list, zip(*fitnesses)))
    

    
    memories = fitnesses[1]
    memory = memories[0]
    for m in range(1,len(memories)):
      memory += memories[m]

    self.memory = memory + self.memory

    fitnesses = fitnesses[0]

    for i in range(self.pop_size):
      offspring_population[i].fitness = fitnesses[i]
      
      

    

    # store cost
    self.num_evals += self.pop_size
    # update the population for the next iteration
    self.population = offspring_population
    # update info
    self.num_gens += 1
    best = self.population[np.argmax([t.fitness for t in self.population])]
    
    self.bestFitnesses.append(max([t.fitness for t in self.population]))
    
    
    lowest = self.population[np.argmin([t.fitness for t in self.population])]
    
    
    self.population.remove(lowest)
    self.population.append(deepcopy(best))
    #print(f"Removing {lowest.fitness} adding {best.fitness}")
    
    self.best_of_gens.append(deepcopy(best))
    for i in range(1,elite_count):
      lowest = self.population[np.argmin([t.fitness for t in self.population])]
      self.population.remove(lowest)
      self.population.append(deepcopy(elites[i]))
      #print(f"Removing {lowest.fitness} adding {elites[i].fitness}")

  def evolve(self):
    """
    Runs the evolution until a termination criterion is met;
    first, a random population is initialized, second the generational loop is started:
    every generation, promising parents are selected, offspring are generated from those parents, 
    and the offspring population is used to form the population for the next generation
    """
    # set the start time
    self.start_time = time.time()

    self._initialize_population()

    # generational loop
    while not self._must_terminate():
      # perform one generation
      self._perform_generation()
      # log info
      if self.verbose:
        print("gen: {},\tbest of gen fitness: {:.3f},\tbest of gen size: {}".format(
            self.num_gens, self.best_of_gens[-1].fitness, len(self.best_of_gens[-1])
            ))
        print(f"Average fitness: {statistics.mean(f := [ind.fitness for ind in self.population]):.2f}, "
      f"Standard deviation: {statistics.stdev(f):.2f}, "
      f"Minimum fitness: {min(f):.2f}, "
      f"Maximum fitness: {max(f):.2f}, "
      f"Median fitness: {statistics.median(f):.2f}, "
      f"Variance: {statistics.variance(f):.2f}")
