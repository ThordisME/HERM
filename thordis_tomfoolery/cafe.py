import simpy
import itertools
import numpy as np
import matplotlib.pyplot as plt

rates = {
    90: 10/60,  # 06:30 - 08:00 --> 90 minutes. 
    150: 50/60, # 08:00 - 09:00 --> 90+60 = 150 minutes
    330: 20/60, # And so on.
    390: 40/60, 
    540: 15/60  # - 15:30, end of day.
}
closing_time = 540

class Stats:
    def __init__(self):
        # CUSTOMER TIME-TRACKING
        self.register_queue_time = []           # Time each customer spent waiting to place their order.
        self.pickup_queue_time = []             # Time each custimer spent picking up their order.
        self.total_time_in_system = []          # Total time each customer spent from arrival time to pickup time.
        
        self.time_until_prep_begins = []        # The time from the customer places the order until a barista begins working on it.

        # OTHER CUSTOMER-RELATED MEASURES
        self.avg_length_register_queue = []     # This is a list because we might also want to graph this. Average length of queue per every hour, or something.
        self.avg_length_pickup_queue = []       # I don't think this has ever not been 0. I'm tracking it for the sake of consistency, though.
        self.avg_count_customers_waiting = []   # Could also be cool to graph.

        # TODO: RESOURCE UTILIZATION

# Once again, we have yet to pin down the service time distribution,
# but to start implementing I'll just assume that it takes at least 5 seconds
# and that the mean is 45 seconds with a standard deviation of 10. Or something.
def order_time(rng):
    '''How long it takes a customer to order, in seconds.'''
    return max(5, rng.normal(45, 10)) 



# ... it's gotta take at least 5 seconds, that's if you're just taking out a croissant and putting it on a dish or something.
# I'll just take a wild guess of 3 minutes (180 seconds) as the mean and 20 seconds as the standard deviation.
# Also I added the time it takes to pay. 
def making_order_time(rng):
    '''How long it takes to get the order ready.
    TODO: We might want to change this process later, add coffe machines and such as resources, and so on.
    For now though, it's just a distribution like so.'''

    #       Time taken to order             Time taken to pay
    return (max(10, rng.normal(180, 20)) + max(3, rng.normal(7, 3)))



# It's gotta take at least a second.
# I'm just going to be optimistic and guess that the mean is 5 seconds, and the std is 1.
def pickup_time(rng):
    '''How long it takes the customer to pick up the order.'''
    return max(1, rng.normal(5, 1))


def customer(env, stats_keeper, ID, register, barista, pickup, rng):
    '''Customer's "lifecycle".
    They arrive, wait in line, order + pay, wait for their order, and then pick up their order.
    
    Params:
        - env:          simpy.Environment.
        - ID:           int, customer's ID.
        - register:     int, ID of the register they queue at.
        - rng:          RNG.

    Returns:
        - everlasting happiness

    '''

    arrival = env.now
    print(f'{ID} arrives at {arrival:.2f}')

    # TODO: I could refactor these "acts" into different function. For readibility. 

    # # # # # # # # # # # # # # 
    #   ACT I: The Register   #
    # # # # # # # # # # # # # # 
    with (register.request() as req):
        yield req # Waiting in line.

        start_ordering = env.now

        print(f"{ID} begins ordering at {start_ordering:.2f}")

        # We wait while the customer places their order.
        ordering = order_time(rng)
        yield env.timeout(ordering)

        stop_ordering = env.now
        print(f"{ID} finishes ordering at {stop_ordering:.2f}")

    # # # # # # # # # # # # # # 
    #   ACT II: The Waiting   #
    # # # # # # # # # # # # # # 
    with (barista.request() as req):
        yield req

        prep_time = making_order_time(rng)
        prep_begin = env.now
        print(f"{ID}'s order preperation begins at {prep_begin:.2f}")

        yield env.timeout(prep_time)
        time_ready = env.now
        print(f"{ID}'s is ready for pickup at {time_ready:.2f}")


    # # # # # # # # # # # # # # 
    #   ACT III: The Pickup   #
    # # # # # # # # # # # # # #
    with pickup.request() as req:
        yield req

        start_pickup = env.now
        print(f"{ID} goes to pick up their order at {start_pickup:.2f}")

        pickupp_time = pickup_time(rng)
        yield env.timeout(pickupp_time)

        end_time = env.now
        print(f"{ID} leaves at {end_time:.2f}")

    # # # # # # # # # # # # # # #
    #   ALSO: Gathering Stats   #
    # # # # # # # # # # # # # # #

    time_in_register_queue = (start_ordering - arrival)
    time_in_pickup_queue = (time_ready - start_pickup)
    total_time = end_time - arrival
    stats_keeper.register_queue_time.append(time_in_register_queue)  # TODO: Do we maybe want :.2f?
    stats_keeper.pickup_queue_time.append(time_in_pickup_queue)
    stats_keeper.total_time_in_system.append(total_time)

    time_from_order_to_prep_start = time_ready - prep_begin
    stats_keeper.time_until_prep_begins.append(time_from_order_to_prep_start)



def arrivals_generator(env, stats_keeper, register, barista, pickup, rates, closing_time, seed=None):
    '''
    Generates customer arrival times using a
        PIECEWISE-CONSTANT NON-HOMOGENEOUS POISSON PROCESS.

    Params:
        - env:      simpy.Environment.
        - rates:    a dictionary, the keys being time passed (breakpoints) in minutes 
                    and the values being arrival rates.
        - seed:     int, seed for RNG.

    Returns:
        - Arrivals. Floats.
    '''

    rng = np.random.default_rng(seed)
    breakpoints = sorted(rates.keys())

    def get_rate(time):
        '''Returns arrival rate for current time, however we defined it.'''
        for point in breakpoints:
            if (time < point):
                return rates[point]
        return rates[breakpoints[-1]] # If past all, just use the last one.


    customer_ID = 0
    inter_arrival = 0
    while True:

        if ((env.now + inter_arrival) >= closing_time):
            print(f"\nCLOSING SHOP. \nNo more arrivals after closing at {closing_time:.2f}.\n")
            return

        rate = get_rate(env.now)                    # Rate at this time.

        inter_arrival = rng.exponential(1 / rate)   # Inter arrival from exponential dist.
        yield env.timeout(inter_arrival)            # Waiting until the next arrival.

        env.process(customer(
            env, 
            stats_keeper,
            customer_ID, 
            register, 
            barista,
            pickup,
            rng
        ))
        customer_ID += 1

def simulate(num_registers, num_baristas, num_pickups, rates, closing_time, ttl):
    '''
    Simulates the process with the specified:
        - num_registers:    int, number of registers / staff at registers.
        - num_baristas:     int, number of baristas (those preparing orders).
        - num_pickups:      int, number of places where drinks can be picked up.
        - rates:            dict, the customers-per-hour rate for each period of time.
        - closing_time:     int, this is when new customers should stop coming in. This is just when we stop taking orders.
        - ttl:              number, "time-to-live", how long the simulation will be allowed to run (in case there are some customers still awaiting their orders).

    Returns:
        - statistics:       Stats class object. Just stats for this particular run.
    '''
    
    env = simpy.Environment()

    statistics = Stats()

    register = simpy.Resource(env, capacity=num_registers)      # Note: SimPy resources are FIFO by default.
    barista = simpy.Resource(env, capacity=num_baristas)        # We have [CAPACITY] amount of baristas, each can focus on one order at a time.
    pickup = simpy.Resource(env, capacity=num_pickups)          # One can pick up their order at a time.

    env.process(arrivals_generator(
        env=env, 
        stats_keeper=statistics,
        register=register, 
        barista=barista,
        pickup=pickup,
        rates=rates, 
        closing_time=closing_time,
        seed=360
    ))
    env.run(until=ttl)  # TODO: NOTE: way beyond closing time. I just wanted to see how much would be done in this time. People stop coming in after closing.

    return statistics


def print_stats(statistics):
    print("Total times in register queue:\n", statistics.register_queue_time)
    print("Time from when the order is placed until a barista begins working on it:\n", statistics.time_until_prep_begins)
    print("Total times in pickup queue:\n", statistics.pickup_queue_time)
    print("Total times in system:\n", statistics.total_time_in_system)
    # We'll want to add more.

def plot_stats(statistics):
    plt.plot(statistics.total_time_in_system)
    plt.ylabel('Waiting time for customers 0 to i')
    plt.show()
    
    plt.plot(statistics.register_queue_time)
    plt.ylabel('Waiting time in register queue for customers 0 to i')
    plt.show()
    
    plt.plot(statistics.pickup_queue_time)
    plt.ylabel('Waiting time in pickup queue for customers 0 to i')
    plt.show()
    
    plt.plot(statistics.time_until_prep_begins)
    plt.ylabel('Waiting time until barista begins working on order for customers 0 to i')
    plt.show()



default = simulate(1, 2, 1, rates, closing_time, 1000)
print_stats(default)
plot_stats(default)

exp_1 = simulate(2, 2, 1, rates, closing_time, 1000)
print_stats(exp_1)
plot_stats(exp_1)

exp_2 = simulate(100, 100, 100, rates, closing_time, 1000)
print_stats(exp_2)
plot_stats(exp_2)