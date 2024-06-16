# genepromulti

## Changes to the genepro package

In order to be able to implement an e-greedy strategy for Deep Q-learning coefficient optimization the genepro had to be slightly adjusted. Below are the main changes to the package.

evo.py

> def perform_generation(self):
> 
> \+ chosen  =  parents[np.argmax([t.fitness for t in  parents])]
> \+ if randu() < 0.1
>> \+ chosen = np.random.choice(parents)
>>




variation.py
> generate_offsprint(...):
>  
>  \+ optimize = False
>  \+ if parent == chosen:
>  > \+ optimize = True

Although the optimize parameter is added in the signature of both gaussian_coeff_mutation and DQN_coeff_opt, it is only used in the latter, such that the e-greedy implementation only affects the DQN_coeff_opt method.
