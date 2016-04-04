import random
import copy
from deap import tools
from neat_operators import neatGP
from speciation import ind_specie, species, specie_parents_child
from fitness_sharing import SpeciesPunishment
from ParentSelection import p_selection
from my_operators import avg_nodes


def varOr(population, toolbox, cxpb, mutpb):
    assert (cxpb + mutpb) <= 1.0, ("The sum of the crossover and mutation "
        "probabilities must be smaller or equal to 1.0.")

    new_pop = [toolbox.clone(ind) for ind in population]
    offspring = []
    for i in range(1, len(new_pop), 2):
        new_pop[i-1].off_cx_set(0), new_pop[i].off_cx_set(0)
        if random.random() < cxpb and len(ind)>1:
            new_pop[i-1].off_cx_set(1)
            new_pop[i].off_cx_set(1)
            offspring1, offspring2 = toolbox.mate(new_pop[i-1], new_pop[i])
            del offspring1.fitness.values
            del offspring2.fitness.values
            offspring1.bestspecie_set(0), offspring2.bestspecie_set(0)
            offspring1.LS_applied_set(0), offspring2.LS_applied_set(0)
            offspring1.LS_fitness_set(None), offspring2.LS_fitness_set(None)
            offspring1.off_cx_set(1), offspring2.off_cx_set(1)
            offspring.append(offspring1)
            offspring.append(offspring2)
    for i in range(len(new_pop)):
        if new_pop[i].off_cx_get() != 1:
            if random.random() < (cxpb+mutpb):  # Apply mutation
                offspring1, = toolbox.mutate(new_pop[i])
                del offspring1.fitness.values
                offspring1.bestspecie_set(0)
                offspring1.LS_applied_set(0)
                offspring1.LS_fitness_set(None)
                offspring1.off_mut_set(1)
                offspring.append(offspring1)

    if len(offspring) < len(population):
        for i in range(len(new_pop)):
            if new_pop[i].off_mut_get() != 1 and new_pop[i].off_cx_get() != 1:
                offspring1 = copy.deepcopy(new_pop[i])
                offspring.append(offspring1)

    return offspring


def varAnd(population, toolbox, cxpb, mutpb):
    """Part of an evolutionary algorithm applying only the variation part
    (crossover **and** mutation). The modified individuals have their
    fitness invalidated. The individuals are cloned so returned population is
    independent of the input population.

    :param population: A list of individuals to vary.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :returns: A list of varied individuals that are independent of their
              parents.

    The variation goes as follow. First, the parental population
    :math:`P_\mathrm{p}` is duplicated using the :meth:`toolbox.clone` method
    and the result is put into the offspring population :math:`P_\mathrm{o}`.
    A first loop over :math:`P_\mathrm{o}` is executed to mate pairs of consecutive
    individuals. According to the crossover probability *cxpb*, the
    individuals :math:`\mathbf{x}_i` and :math:`\mathbf{x}_{i+1}` are mated
    using the :meth:`toolbox.mate` method. The resulting children
    :math:`\mathbf{y}_i` and :math:`\mathbf{y}_{i+1}` replace their respective
    parents in :math:`P_\mathrm{o}`. A second loop over the resulting
    :math:`P_\mathrm{o}` is executed to mutate every individual with a
    probability *mutpb*. When an individual is mutated it replaces its not
    mutated version in :math:`P_\mathrm{o}`. The resulting
    :math:`P_\mathrm{o}` is returned.

    This variation is named *And* beceause of its propention to apply both
    crossover and mutation on the individuals. Note that both operators are
    not applied systematicaly, the resulting individuals can be generated from
    crossover only, mutation only, crossover and mutation, and reproduction
    according to the given probabilities. Both probabilities should be in
    :math:`[0, 1]`.
    """
    offspring = [toolbox.clone(ind) for ind in population]

    # Apply crossover and mutation on the offspring
    for i in range(1, len(offspring), 2):
        if random.random() < cxpb:
            offspring[i-1], offspring[i] = toolbox.mate(offspring[i-1], offspring[i])
            del offspring[i-1].fitness.values, offspring[i].fitness.values
            offspring[i-1].bestspecie_set(0), offspring[i].bestspecie_set(0)

    for i in range(len(offspring)):
        if random.random() < mutpb:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values
            offspring[i].bestspecie_set(0)


    return offspring


def neat_GP( population, toolbox, cxpb, mutpb, ngen, neat_alg, neat_cx, neat_h, neat_pelit, n_corr, num_p, params, problem, stats=None,
             halloffame=None, verbose=__debug__):
    """This algorithm reproduce the simplest evolutionary algorithm as
    presented in chapter 7 of [Back2000]_.

    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param ngen: The number of generation.
    :param neat_alg: wheter or not to use species stuff.
    :param neat_cx: wheter or not to use neatGP cx
    :param neat_h: indicate the distance allowed between each specie
    :param neat_pelit: probability of being elitist, it's used in the neat cx and mutation
    :param LS_flag: wheter or not to use LocalSearchGP
    :param LS_select: indicate the kind of selection to use the LSGP on the population.
    :param cont_evalf: contador maximo del numero de evaluaciones
    :param n_corr: run number just to wirte the txt file
    :param p: problem number just to wirte the txt file
    :param params:indicate the params for the fitness sharing, the diffetent
                    options are:
                    -DontPenalize(str): 'best_specie' or 'best_of_each_specie'
                    -Penalization_method(int):
                        1.without penalization
                        2.penalization fitness sharing
                        3.new penalization
                    -ShareFitness(str): 'yes' or 'no'
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population.

    It uses :math:`\lambda = \kappa = \mu` and goes as follow.
    It first initializes the population (:math:`P(0)`) by evaluating
    every individual presenting an invalid fitness. Then, it enters the
    evolution loop that begins by the selection of the :math:`P(g+1)`
    population. Then the crossover operator is applied on a proportion of
    :math:`P(g+1)` according to the *cxpb* probability, the resulting and the
    untouched individuals are placed in :math:`P'(g+1)`. Thereafter, a
    proportion of :math:`P'(g+1)`, determined by *mutpb*, is
    mutated and placed in :math:`P''(g+1)`, the untouched individuals are
    transferred :math:`P''(g+1)`. Finally, those new individuals are evaluated
    and the evolution loop continues until *ngen* generations are completed.
    Briefly, the operators are applied in the following order ::

        evaluate(population)
        for i in range(ngen):
            offspring = select(population)
            offspring = mate(offspring)
            offspring = mutate(offspring)
            evaluate(offspring)
            population = offspring

    This function expects :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox.

    .. [Back2000] Back, Fogel and Michalewicz, "Evolutionary Computation 1 :
       Basic Algorithms and Operators", 2000.
    """

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    if neat_alg:  # assign specie to each individual on the population
        species(population,neat_h)
        ind_specie(population)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # save data for the best individual
    best = open('./Results/%s/bestind_%d_%d.txt'%(problem, num_p, n_corr), 'a')
    best_st = open('./Results/%s/bestind_string_%d_%d.txt'%(problem, num_p, n_corr), 'a')

    #  take the best on the population
    best_ind = best_pop(population)
    fitnesst_best = toolbox.map(toolbox.evaluate_test, [best_ind])
    best_ind.fitness_test.values=fitnesst_best[0]
    best.write('\n%s;%s;%s;%s;%s' % (0, best_ind.fitness_test.values[0], best_ind.fitness.values[0], len(best_ind), avg_nodes(population)))
    best_st.write('\n%s;%s' % (0, best_ind))

    if neat_alg:  # applying fitness sharing
        SpeciesPunishment(population,params,neat_h)

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print logbook.stream

    # Begin the generational process
    for gen in range(1, ngen+1):

        if neat_alg:  # select set of parents
            parents = p_selection(population, gen)
        else:
            parents = toolbox.select(population, len(population))

        if neat_cx:  # applying neat-crossover
            n = len(parents)
            mut = 1
            cx = 1
            offspring = neatGP(toolbox, parents, cxpb, mutpb, n, mut, cx, neat_pelit)
        else:
            offspring = varOr(parents, toolbox, cxpb, mutpb)

        if neat_alg:  # Assign species
            specie_parents_child(parents, offspring, neat_h)
            offspring[:] = parents+offspring
            ind_specie(offspring)

        for ind in offspring:
            del ind.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        if neat_alg:
            SpeciesPunishment(offspring, params, neat_h)

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(population), **record)
        if verbose:
            print logbook.stream

        best_ind = best_pop(population)
        fitnesses_test = toolbox.map(toolbox.evaluate_test, [best_ind])
        best_ind.fitness_test.values = fitnesses_test[0]
        best.write('\n%s;%s;%s;%s;%s'%(gen, best_ind.fitness_test.values[0], best_ind.fitness.values[0], len(best_ind), avg_nodes(population)))
        best_st.write('\n%s;%s' % (gen, best_ind))

    return population, logbook


def best_pop(population):
    orderbyfit = list()
    orderbyfit = sorted(population, key=lambda ind:ind.fitness.values)
    return orderbyfit[0]