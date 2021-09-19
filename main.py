from src import simulation, tests, debugging
import time

start_time = time.time()

""" Run Simulation """
simulation = simulation.Simulation()
simulation.run()

print("--- Simulation took %s seconds ---" % round(time.time() - start_time, 2))

""" Run Tests """
tests = tests.Test(simulation)
tests.run()

""" Compare multiple simulations """
tests.save_iq_test()  # the tests that are saved here are used to compute the heritability.
tests.compute_heritability() # Run the simulation both for diz and mono, to get updated h2 score


"""some optional extra insight in data. For debugging purposes mainly"""
debug = debugging.Debugging(simulation)
debug.run()