import copy
import simpy
import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt

HOURS_SIMULATED = 9     # The café is open for 9 hours.
RUSH_HOUR = (2,3)       # The Rush hour reaches its peak when the clock strikes {HOUR 2 OF SHIFT}. 
                        # This is where we want to experiment with adding more staff.
CLOSES = 540            # The simulation ends after those 9 hours and not a minute later.
OPEN_FOR = 525          # They'll stop serving new customers 8.75 hours in, that's 15 minutes before officially closing the shop.
                        # This should give them time to finish any orders they were working on and close up the café.

def arrival_rate(time):
    '''
    Time-varying arrival rate in customers per minute. Lambda(t).
    (Describes how frequently customers tend to arrive at each moment in time).

    We assume that a normal day at the café has:
    - A constanit baseline arrival rate of {baseline}
    - A Gaussian-shaped "rush hour" bump centered at {rush} minutes into the day.

    arrival_rate(t) = baseline + Gaussian bump
    '''

    baseline    = 6/60      # 6 customers per hour as the baseline.
    peak        = 30/60     # 30 customers per hour at peak hours.
    rush        = 120       # That's 2 hours into the day. 
    rush_length = 60        # Length of rush hour (1 hour).

    return (baseline +   (peak *   (np.exp(-((time-rush)**2) / (2*(rush_length**2))))   )   )

ttt = np.linspace(0, 540, 1000)   # 9 hours, 1000 points (smooth).

rate = (arrival_rate(ttt) * 60) # *60 since the arrival_rate returns the arrival rate in minutes. 

plt.figure(figsize=(10,5))
plt.plot(ttt, rate)
plt.xlabel("Minute of workday")
plt.ylabel("Customers/minute")
plt.title("Customer Arrival Rate")
plt.grid(True)
plt.show()

ttt = np.linspace(0, 540, 1000)
interarrival = 1 / arrival_rate(ttt)
plt.figure(figsize=(5,2.5)) # I made it smaller.
plt.plot(ttt, interarrival)
plt.title("Instantaneous inter-arrival time")
plt.xlabel("Minutes of workday")
plt.ylabel("Mean inter-arrival time (minutes)")
plt.grid(True)
plt.show()

# TIME-RELATED GLOBALS - - - - - - - - - - - - - - - - - - - - - - - - - -  
SECONDS = 1/60
MINUTES = 1
HOURS = 60

# LIKELIHOOD-RELATED GLOBALS - - - - - - - - - - - - - - - - - - - - - - -  
CHANCE_THAT_COFFEE_MACHINE_NEEDED = 0.9
CHANCE_THAT_OVEN_NEEDED = 0.15

# SHIFT-CONSTRAINT-RELATED GLOBALS - - - - - - - - - - - - - - - - - - - - 
NUM_TOTAL_STAFF_THAT_FIT_BEHIND_COUNTER = 5
NUM_REGISTERS_THAT_FIT = 2

# DEFAULT NUMBER GLOBALS - - - - - - - - - - - - - - - - - - - - - - - - - 
DEFAULT_NUM_REGISTERS       = 1
DEFAULT_NUM_BARISTAS        = 2 
DEFAULT_NUM_COFFEE_MACHINES = 1 
DEFAULT_NUM_OVENS           = 1 


class Shifts:
    '''NOTE: Many of the functions below are not currently in use, but could be used for future experiments if so desired.'''
    
    def __init__(self, default_num_registers=1, default_num_baristas=1):
        '''
        shifts.add_register_to_shift(shift=1, count=1)      --> adds 1 register to shift 1.
        shifts.add_register_to_shift(shift=1, count=2)      --> adds 2 registers to shift 1.
        shifts.remove_register_from_shift(shift=1, count=1) --> removes 1 register from shift 1.
        ... and so on.

        Each time you modify a shift, the shift automatically gets validated according to the constraints we set 
        (no more than 5 people can fit behind the counter and no more than 2 registers can fit in the space).
        If the shift is not valid, the function will return False.
        If it's valid, then it'll return True and you're free to move on.

        Inputs:
            default_num_registers:  Int.    You can initialize a schedule with different numbers of each type of staff member. The default for registers is 1, the bare minimum.
            default_num_baristas:   Int.    You can initialize a schedule with different numbers of each type of staff member. The default for baristas is 1, the bare minimum.
        '''

        # If the inputs weren't valid, we return an error.
        validate_inputs = [default_num_registers, default_num_baristas]
        if not self._validate_shift(validate_inputs):
            raise ValueError(f"Invalid initial values: registers={default_num_registers} and baristas={default_num_baristas}.")

        # Else, we put this --> {HOURS_SIMULATED} <-- many identical shifts into a list.
        self.shifts = [[default_num_registers, default_num_baristas] for _ in range(HOURS_SIMULATED)]
        
    def _validate_shift(self, shift):

        # Are there at least one register and one barista?
        if ((shift[0] < 1) or (shift[1] < 1)):
            return False
        
        # Are there more than 5 people behind the counter?
        if ((shift[0] + shift[1]) > NUM_TOTAL_STAFF_THAT_FIT_BEHIND_COUNTER):
            return False
        
        # Are there more registers here than available? 
        if (shift[0] > NUM_REGISTERS_THAT_FIT):
            return False
        
        return True
    
    def _add_employee_to_shift(self, shift_nr, count, employee_type):
        shift = self.shifts[shift_nr - 1]
        shift[employee_type] += count

        if (self._validate_shift(shift)):
            return True
        else:
            shift[employee_type] -= count
            return False
        
    def _remove_employee_from_shift(self, shift_nr, count, employee_type):
        shift = self.shifts[shift_nr - 1]
        shift[employee_type] -= count

        if (self._validate_shift(shift)):
            return True
        else:
            shift[employee_type] += count
            return False
        
    def _add_employee_to_every_shift(self, count, employee_type):
        for shift in self.shifts:
            shift[employee_type] += count

        # If this was a valid action:
        if (all(self._validate_shift(shift) for shift in self.shifts)):
            return True

        # If this was not a valid action:
        for shift in self.shifts:
            shift[employee_type] -= count

        return False
    
    def _remove_employee_from_every_shift(self, count, employee_type):
        for shift in self.shifts:
            shift[employee_type] -= count

        # If this was a valid action:
        if (all(self._validate_shift(shift) for shift in self.shifts)):
            return True

        # If this was not a valid action:
        for shift in self.shifts:
            shift[employee_type] += count

        return False

    def _add_employee_to_shift_range(self, shift_range, count, employee_type):
        '''Should take in a tuple like (2,3) and add {count} of {employee_type} employees to shifts 2 and 3.'''
        for i in range(shift_range[0], shift_range[1]+1):
            e = self._add_employee_to_shift(i, count, employee_type)
            if not e:
                return False # Something went wrong.
        return True
    
    def _remove_employee_from_shift_range(self, shift_range, count, employee_type):
        '''Should take in a tuple like (2,3) and remove {count} of {employee_type} employees from shifts 2 and 3.'''
        for i in range(shift_range[0], shift_range[1]+1):
            e = self._remove_employee_from_shift(i, count, employee_type)
            if not e:
                return False # Something went wrong.
        return True

    def add_register_to_shift(self, shift_nr, count):
        return self._add_employee_to_shift(shift_nr, count, 0)

    def add_barista_to_shift(self, shift_nr, count):
        return self._add_employee_to_shift(shift_nr, count, 1)

    def remove_register_from_shift(self, shift_nr, count):
        return self._remove_employee_from_shift(shift_nr, count, 0)

    def remove_barista_from_shift(self, shift_nr, count):
        return self._remove_employee_from_shift(shift_nr, count, 1)
    
    def add_register_to_every_shift(self, count):
        return self._add_employee_to_every_shift(count, 0)

    def add_barista_to_every_shift(self, count):
        return self._add_employee_to_every_shift(count, 1)
    
    def remove_register_from_every_shift(self, count):
        return self._remove_employee_from_every_shift(count, 0)

    def remove_barista_from_every_shift(self, count):
        return self._remove_employee_from_every_shift(count, 1)

    def add_register_to_shift_range(self, shift_range, count=1):
        return self._add_employee_to_shift_range(shift_range, count, 0)

    def add_barista_to_shift_range(self, shift_range, count=1):
        return self._add_employee_to_shift_range(shift_range, count, 1)
    
    def remove_register_from_shift_range(self, shift_range, count=1):
        return self._remove_employee_from_shift_range(shift_range, count, 0)

    def remove_barista_from_shift_range(self, shift_range, count=1):
        return self._remove_employee_from_shift_range(shift_range, count, 1)

def adjust_staff(role, count, current, pool, hour):
    '''
    shift_manager's helper.
    Adjusts the staff count for the given role at shift change.

    - role:         Str.        "register" or "barista".
    - count:        Int.        Scheduled staff count at this hour.
    - current:      Int.        Current staff count in the shift record
    - pool:         Store.      Pool of staff.
    - hour:         Int.        Current simulation hour.
    '''

    # If we need more staff:
    if (count > current):
        for _ in range(count - current):
            yield pool.put(f"{role}_{hour}")

    # If we need less staff:
    elif (count < current):
        # Note that we're only removing staff that's free. We won't remove someone when they're in the middle of making an order. We'll let them finish.
        remove_count = (current - count)
        for _ in range(remove_count):
            if len(pool.items) > 0:
                yield pool.get()

    # And if we already have the correct amount of staff, nothing changes.
    else:
        pass


def shift_manager(env, shifts, register_pool, barista_pool, current_shift):
    '''
    Every hour, updating the number of registers and baristas according to schedule.
    
    NOTE: shift.shifts[hour] = [# registers, # baristas]
    '''

    registers_scheduled, baristas_scheduled = shifts.shifts[0]
    current_shift["registers"] = registers_scheduled
    current_shift["baristas"] = baristas_scheduled

    hour = 1
    while (hour < HOURS_SIMULATED):

        yield env.timeout(60) # The shift passes (1 hour, always).

        registers_scheduled, baristas_scheduled = shifts.shifts[hour]

        # - - - REGISTERS - - - - -

        yield from adjust_staff(
            role="register",
            count=registers_scheduled,
            current=current_shift["registers"],
            pool=register_pool,
            hour=hour
        )
        current_shift["registers"] = registers_scheduled

        # - - - BARISTAS - - - - -

        yield from adjust_staff(
            role="barista",
            count=baristas_scheduled,
            current=current_shift["baristas"],
            pool=barista_pool,
            hour=hour
        )
        current_shift["baristas"] = baristas_scheduled

        # - - - 
        hour += 1 # ... and the shift passes.

class Stats:
    '''A class to keep track of the numbers for each simulation.'''

    def __init__(self):

        # - - - GENERAL INFO - - - - - - - - - - - - - - - - - - -

        self.run_seed = 0

        # - - - CUSTOMER-RELATED NUMBERS - - - - - - - - - - - - -

        # TIME-TRACKING
        self.register_queue_time = []           # Time each customer spent waiting to place their order.
        self.total_time_in_system = []          # Total time each customer spent from arrival time to pickup time.
        self.time_until_prep_begins = []        # The time from the customer places the order until a barista begins working on it.
        self.arrival_times = []                 # For plotting.

        self.per_hour = {
            "arrivals":                 ([0] * HOURS_SIMULATED),
            "completed_orders":         ([0] * HOURS_SIMULATED),
            "avg_wait_time":            [[] for _ in range(HOURS_SIMULATED)],
        }

        self.num_customers = 0                  # Not super important but we might as well.

        # - - - RESOURCE UTILIZATION - - - - - - - - - - - - - - -

        # STAFF BUSYNESS
        self.register_busy_time = 0.0           # Total minutes all registers spent actively serving customers.
        self.barista_busy_time = 0.0            # Total minutes all baristas spent preparing orders.
        self.register_time_available = 0.0      # Total register minutes available from shift schedule. NOTE: Idle time = register_time_available - register_busy_time.
        self.barista_time_available = 0.0       # Total barista minutes available from shift schedule. Similar NOTE: Idle time = barista_time_available - barista_busy_time.

    def compute_staff_time_available(self, shifts, minutes_per_hour=HOURS): # NOTE: HOURS is a global currently set to 60. The amount of minutes in an hour.
        '''Computes the total available staff time from shifts for each staff type, given the shifts.'''

        self.register_time_available = 0.0  # Must always reset to 0.
        self.barista_time_available = 0.0

        for hour, (registers, baristas) in enumerate(shifts.shifts):
            self.register_time_available += (registers * minutes_per_hour)
            self.barista_time_available  += (baristas * minutes_per_hour)

    def register_utilization(self):
        '''Fraction of total scheduled register time that was spent busy. (Register utilization %).'''
        if (self.register_time_available == 0):
            return 0.0
        
        return (self.register_busy_time / self.register_time_available)
    
    def barista_utilization(self):
        '''Fraction of total scheduled barista time that was spent busy. (Barista utilization %).'''
        if (self.barista_time_available == 0):
            return 0.0
        
        return (self.barista_busy_time / self.barista_time_available)
    
# - - - - - - - - - -

class MetricSummary:
    '''
    We may additionally want a summary for different metrics in different experiments,
    for further analysis.
    '''
    def __init__(self, mean, std, ci_low, ci_high, num_replications):
        self.mean = mean
        self.std = std
        self.ci_low = ci_low
        self.ci_high = ci_high
        self.num_replications = num_replications

# Defining the function for each:
METRICS = {
    "mean_total_time":      lambda stats: np.mean(stats.total_time_in_system),
    "mean_register_queue":  lambda stats: np.mean(stats.register_queue_time),
    "register_utilization": lambda stats: stats.register_utilization(),
    "barista_utilization":  lambda stats: stats.barista_utilization(),
    "num_customers":        lambda stats: stats.num_customers,
}

def order_time(rng):
    '''How long it takes a customer to order, in minutes.'''
    sample = 10 + rng.gamma(5, 11)      # Shape=5, scale=11, then +10 sec base.
    return (min(sample, 120) * SECONDS) # Hard max cutoff: 2 minutes = 120 seconds.

 
def making_order_time(rng):
    '''
    How long it takes to get the order ready.
    Triangular distribution: min=15s, mode=3min, max=10min.
    '''
    return (rng.triangular(15, 3*60, 10*60) * SECONDS)


# Uniform distribution, minimum 0.5 minutes and maximum 2 minutes.
def coffee_machine_time(rng):
    '''How long it takes the barista to use the coffee machine for a given order.'''
    return (rng.uniform(low=30, high=120) * SECONDS)


# Uniform distribution, minimum 1 minute maximum 5 minutes (just to heat things up).
def oven_time(rng):
    '''How long it takes the barista to use the oven for a given order.'''
    return (rng.uniform(low=60, high=300) * SECONDS)

def use_coffee_machine(env, coffee_machine, rng):
    '''If a barista needs to use a coffee machine, this function will run.'''

    with (coffee_machine.request() as req):
        yield req
        time = coffee_machine_time(rng)
        yield env.timeout(time)

def use_oven(env, oven, rng):
    '''If a barista needs to use an oven, this function will run.'''

    with (oven.request() as req):
        yield req
        time = oven_time(rng)
        yield env.timeout(time)


def customer(env, stats_keeper, register_pool, barista_pool, coffee_machine, oven, rng, current_shift):
    '''
    Customer's "lifecycle".
    A customer arrives to wait in line, orders + pays, waits while a barista prepares their order, and then pick up their order when it's ready.
    
    Note that the baristas may also use resources such a coffee machine or oven, and the availability of such devices affects the customer's waiting time.
    
    Params:
        - env:              simpy.Environment.  The environment of this current run.
        - stats_keeper:     Stats.              Used for collecting stats on the simulation run.
        - ID:               Int.                The customer's ID.
        - register_pool:    simpy.Store.        Stores registers, which are of the type simpy.Resource.
        - barista_pool:     simpy.Store.        Stores baristas, which are of the type simpy.Resource.
        - coffee_machine:   simpy.Resource.     The baristas likely need to use it to prepare your order.
        - oven:             simpy.Resource.     The baristas may need to use it to prepare your order.
        - rng:              Generator.          A random number generator (RNG).
        - current_shift:    Int.                The number of the hour.
    '''

    arrival = env.now
    stats_keeper.num_customers += 1
    hour = int(env.now // 60)
    stats_keeper.per_hour["arrivals"][hour] += 1

    # # # # # # # # # # # # # # 
    #   ACT I: The Register   #
    # # # # # # # # # # # # # # 
    register = yield register_pool.get()        # Acuiring staff member.
    
    start_ordering = env.now
    ordering = order_time(rng)
    stats_keeper.register_busy_time += ordering # The register is busy for "ordering" minutes here, adding to the total "register_busy_time".
    yield env.timeout(ordering)                 # We wait while the customer places their order and pays.
    
    stop_ordering = env.now
    yield register_pool.put(register)           # Releasing staff member.

    # # # # # # # # # # # # # # 
    #   ACT II: The Waiting   #
    # # # # # # # # # # # # # # 
    # --> Barista --> Coffee machine? --> Oven? --> General prep --> Done.

    barista = yield barista_pool.get()
    prep_begin = env.now

    needs_coffee =  (rng.random() < CHANCE_THAT_COFFEE_MACHINE_NEEDED)
    needs_oven =    (rng.random() < CHANCE_THAT_OVEN_NEEDED)
    
    if (needs_coffee):
        yield env.process(use_coffee_machine(env, coffee_machine, rng))
    if (needs_oven):
        yield env.process(use_oven(env, oven, rng))

    prep_time = making_order_time(rng)
    yield env.timeout(prep_time)

    # - - - - - - - - - - Order finished! - - - - - - - - - -

    time_ready = env.now
    stats_keeper.barista_busy_time += (time_ready - prep_begin) # Tracking how long baristas were busy on this order.

    # The barista then leaves if their shift is over.
    if (len(barista_pool.items) < current_shift["baristas"]):
        yield barista_pool.put(barista)
    else:
        pass

    # # # # # # # # # # # # # # 
    #   ACT III: The Pickup   #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Originally we also simulated the pickup queue, with num_pickup_stations being an input.   #
    # However, picking up an order is a quick ordeal so no queues were forming.                 #
    # Therefore we took out, to keep things simple and avoid wasting time and computing power.  #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # # # # # # # # # # # # # # #
    #   ALSO: Gathering Stats   #
    # # # # # # # # # # # # # # #

    end_time = env.now
    hour = int(end_time // 60)

    stats_keeper.per_hour["completed_orders"][hour] += 1
    stats_keeper.per_hour["avg_wait_time"][hour].append(end_time - arrival)

    time_in_register_queue          = max(0, (start_ordering - arrival))
    time_from_order_to_prep_start   = max(0, (prep_begin - stop_ordering))
    total_time                      = max(0, (end_time - arrival))

    stats_keeper.register_queue_time.append(time_in_register_queue)
    stats_keeper.time_until_prep_begins.append(time_from_order_to_prep_start)
    stats_keeper.total_time_in_system.append(total_time)



def arrivals_generator(env, stats_keeper, register_pool, barista_pool, coffee_machine, oven, current_shift, rng):
    '''
    Generates arrivals of customers according to the chosen arrival_rate.
    Runs the 'customer' process for each of the arrivals.

    Params:
        - env:              simpy.Environment.  The environment of this current run.
        - stats_keeper:     Stats.              Used for collecting stats on the simulation run.
        - register_pool:    simpy.Store.        Stores registers, which are of the type simpy.Resource.
        - barista_pool:     simpy.Store.        Stores baristas, which are of the type simpy.Resource.
        - coffee_machine:   simpy.Resource.     The baristas likely need to use it to prepare your order.
        - oven:             simpy.Resource.     The baristas may need to use it to prepare your order.
        - arrival_rate:     A function.         Returns the "customers per minute" arrival rate based on the the current time.
        - current_shift:    Int.                The number of the hour.
        - rng:              Int.                Random number generator.
    '''

    inter_arrival = 0
    while True:

        if ((env.now + inter_arrival) >= OPEN_FOR):     # "OPEN_FOR" is the point at which we stop taking orders, assume around 15 minutes before closing. There will be o more "arrivals" after this point.
            return

        rate = arrival_rate(env.now)
        inter_arrival = rng.exponential(1 / rate)       # Inter-arrival time depeds on the current rate, arrival_rate(t). Drawing an exponential gap between arrivals.
                                                        # NOTE: We're updating the rate every time we generate a new arrival.
        yield env.timeout(inter_arrival)                # Waiting until the next arrival. Then we start the customer lifecycle, and repeat.

        arrival_time = env.now                          # For plotting.
        stats_keeper.arrival_times.append(arrival_time)

        env.process(customer(
            env,
            stats_keeper,
            register_pool,
            barista_pool,
            coffee_machine,
            oven,
            rng,
            current_shift
        ))

def simulate(shifts, num_coffee_machines, num_ovens, ttl, seed=None):
    '''
    Simulates the process with the specified:
        - shifts:               Shifts.     Contained the scheduled shifts.
        - num_coffee_machines:  Int.        Number of coffee machines which baristas have at their disposal.
        - num_ovens:            Int.        Number of ovens which baristas have at their disposal.
        - arrival_rate:         Function.   Returns the arrival rate in customers per minute.
        - ttl:                  Float.      "time-to-live", how long the simulation will be allowed to run (in case there are some customers still awaiting their orders).
        - seed:                 Int.        Seed for a RNG, which will be fed into the arrivals_generator process.

    Returns:
        - statistics:           Stats.  The numbers for this particular simulation run.
    '''
    
    env = simpy.Environment()

    statistics = Stats()
    statistics.compute_staff_time_available(shifts, minutes_per_hour=HOURS) # Pre-computing how many staff-minutes we have on the schedule.

    # - - - STAFF POOLS - - - - - 
    register_pool = simpy.Store(env)
    barista_pool  = simpy.Store(env)

    # Initializing shift 1 staff:
    first_shift_num_registers, first_shift_num_baristas = shifts.shifts[0]
    for i in range(first_shift_num_registers):
        register_pool.put(f"register_0")
    for i in range(first_shift_num_baristas):
        barista_pool.put(f"barista_0")

    current_shift = {"registers": first_shift_num_registers, "baristas": first_shift_num_baristas} # This is to track the current shift's capacity.

    # - - - RESOURCES WHOSE CAPACITIES DON'T CHANGE THROUGHOUT THE DAY - - - - -
    coffee_machine  = simpy.Resource(env, capacity=num_coffee_machines)
    oven            = simpy.Resource(env, capacity=num_ovens)

    # - - - STARTING PROCESSES - - - - -
    env.process(shift_manager(env, shifts, register_pool, barista_pool, current_shift)) # Updates the number of registers and baristas according to schedule each hour.

    env.process(arrivals_generator(
        env=env, 
        stats_keeper=statistics,
        register_pool=register_pool,
        barista_pool=barista_pool,
        coffee_machine=coffee_machine,
        oven=oven,
        current_shift=current_shift,
        rng=np.random.default_rng(seed)
    )) # "arrivals_generator" in turn calls the "customer" process, simulating the customer's "life cycle".
    env.run(until=ttl)

    return statistics


def run_simulations(e, exp_seed=None):
    '''
    Runs the simulation several times.

    Input:  
    - e:            Experiment.     The ID and other parameters of an experiment.
    - exp_seed:     Int.            Seed for a random number generator, which will in turn be used to generate many more seeds for each simulation run.
    
    Output: 
    - A list of Stats, one for each run.
    '''
    exp_rng = np.random.default_rng(exp_seed)
    all_stats = []

    for i in range(e.num_of_simulations):

        per_run_seed = exp_rng.integers(0, (2**63)-1) # Making a per-run seed from the experiment's RNG.

        stats = simulate(
            shifts=e.shifts, 
            num_coffee_machines=e.num_coffee_machines, 
            num_ovens=e.num_ovens, 
            ttl=CLOSES,
            seed=per_run_seed
        )
        stats.run_seed = int(per_run_seed)
        all_stats.append(stats)

    return all_stats

DEFAULT_NUM_SIMULATIONS = 1000

WAIT_WEIGHT         = 0.85
UTILIZATION_WEIGHT  = 0.15
CONFIDENCE          = 0.95
TIE_BREAKER         = 1e-9 

BARE_MINIMUM = Shifts()                                 # All (1,1) by default. The minimum employees needed for functioning.
AVERAGESTAFFED = Shifts(1,2)                            # The ORIGINAL schedule, where there's 1 register and 2 baristas on the clock at all hours.

HIGHSTAFFED = copy.deepcopy(AVERAGESTAFFED)
HIGHSTAFFED.add_barista_to_shift_range(RUSH_HOUR)       # (1,3) during rush hour, schedule stays at (1,2) during all other hours.

FULLSTAFFED_1 = copy.deepcopy(HIGHSTAFFED)
FULLSTAFFED_1.add_register_to_shift_range(RUSH_HOUR)    # (2,3) during rush hour, schedule stays at (1,2) during all other hours.

FULLSTAFFED_2 = copy.deepcopy(HIGHSTAFFED)
FULLSTAFFED_2.add_barista_to_shift_range(RUSH_HOUR)     # (1,4) during rush hour, schedule stays at (1,2) during all other hours.

class Experiment:
    '''
    A class representing a single experiment.
    The ID is the only required input. I could've just made it 0 by default, 
    but I wanted to make super sure that adding it isn't forgotten.
    '''

    def __init__(self, ID, num_of_simulations=DEFAULT_NUM_SIMULATIONS):
        self.ID                     = ID                            # Nr. of experiment. 
        self.num_of_simulations     = num_of_simulations            # Whatever number we've decided will suffice.
        self.shifts                 = BARE_MINIMUM                  # The minimum amount of staff needed for the system to function is 1 register and 1 barista.
        self.num_coffee_machines    = DEFAULT_NUM_COFFEE_MACHINES   # 1 by default. May be changed.
        self.num_ovens              = DEFAULT_NUM_OVENS             # 1 by default. May be changed.
        self.results                = None                          # This will be added once a test is run.
        self.metrics                = None                          # Will also be added.

def clone_experiment(exp):
    '''We'll use clones to determine the ideal amount of equipment.'''
    new = Experiment(exp.ID, num_of_simulations=exp.num_of_simulations)
    new.shifts = copy.deepcopy(exp.shifts)
    new.num_coffee_machines = exp.num_coffee_machines
    new.num_ovens = exp.num_ovens
    return new

expr_nr_seed        = ["0", "1", "2", "3", "4"] # TODO: Added "0". Are there any chnages?
machines_nr_seed    = ["one_one", "one_two", "one_three", "two_one", "two_two"]
exp_seeds           = ([0] * (len(expr_nr_seed) * len(machines_nr_seed)))

# SeedSequence expects int or sequence of ints for entropy, so we will provide just that:
seed_i = 0
for i in expr_nr_seed:
    for j in machines_nr_seed:
        seed_combo = f"{i}_{j}"
        seed_value = abs(hash(seed_combo)) % (2**63) 
        seed_i += 1

def helper_numeric_per_hour(results, per_hour, key, out):
    '''
    Builds a numeric array of shape (number of runs x HOURS_SIMULATED).
    Computes the mean across runs for that metric.
    '''
    arr = np.array([np.array(result.per_hour[key], dtype=float) for result in results])
    out[key] = np.nanmean(arr, axis=0)


def helper_list_of_samples_per_hour(results, runs, key, out):
    '''
    Computing a per-hour means for each run (hour's samples --> mean), 
    then averaging the per-hour means across runs.
    Used for keys whose per-hour data are lists of samples [[...], [...], ... etc.].
    '''
    per_run_hour_means = np.full((runs, HOURS_SIMULATED), np.nan, dtype=float)

    for run, result in enumerate(results):
        for hour in range(HOURS_SIMULATED):
            samples = result.per_hour[key][hour]
            if samples:
                per_run_hour_means[run, hour] = np.mean(samples)

    out[key] = np.nanmean(per_run_hour_means, axis=0)


def get_all_averages_per_hour(results):
    '''
    Summarizing per-hour entries across multiple simulation runs.

    2 cases as of right now:
        1) Numeric per-hour
            Like [1, 2, 3, ...]
            We directly acerage across runs.
        2) List of samples per-hour
            Like [[1, 2, 3, ...], ...]
            We first compute the per-hour mean per run, 
            then average that across runs.

    Returns a dict mapping each per-hour key to a vector of length HOURS_SIMULATED.
    '''

    per_hour = results[0].per_hour
    keys = list(per_hour.keys())
    runs = len(results)

    out = {}

    for key in keys:
        first = per_hour[key]

        is_correct_length       = (isinstance(first, (list, tuple)) and len(first) == HOURS_SIMULATED)
        is_flat_numeric_list    = all(not isinstance(x, (list,tuple)) for x in first)
        contains_nested_lists   = any(isinstance(x, (list,tuple)) for x in first)

        # - - - NUMERIC-PER-HOUR - - - - -
        if (is_correct_length and is_flat_numeric_list ):
            helper_numeric_per_hour(results, per_hour, key, out)

        # - - - LIST-OF-SAMPLES-PER-HOUR - - - - - 
        elif (is_correct_length and contains_nested_lists):
            helper_list_of_samples_per_hour(results, runs, key, out)

        # - - - WE DON'T KNOW WHAT WE'RE LOOKING AT - - - - -
        else:
            try: # If this doesn't work out, we just skip it.
                arr = np.array([np.array(r.per_hour[key]) for r in results], dtype=float)
                out[key] = np.nanmean(arr, axis=0)
            except Exception:
                print(f"Skipping '{key}' (unsupported per_hour shape).")
                out[key] = None

    return out

def plot_all_averages_per_hour(results, excluding=()):
    '''We already have a function that prints these averages, I figured a graph would be more easily digestible to readers.'''

    averages = get_all_averages_per_hour(results)

    keys_to_plot = [
        key for key in averages.keys()
        if averages[key] is not None and not any(x in key.lower() for x in excluding)
    ]
    num_plots = len(keys_to_plot)
    if num_plots == 0:
        return
    
    # - - - - -

    fig, axes = plt.subplots(num_plots, 1, figsize=(8, (3*num_plots)))
    if (num_plots == 1): # I think it'll never be 1, but just in case...
        axes = [axes]

    for i, j in zip(axes, keys_to_plot):
        vec = averages[j]
        hours = np.arange(len(vec))
        i.plot(hours, vec, marker='*')
        i.set_title(f"{j}: Per-hour avg. over {DEFAULT_NUM_SIMULATIONS} simulation runs.")
        #i.set_ylabel("Average")
        i.set_xlabel("Hour of Day")
        i.grid(True, linestyle='--', alpha=0.6)

    axes[-1].set_xlabel("Hour of Day")
    plt.tight_layout()
    plt.show()


def trim_outliers(vals, lower=1, upper=99):
    '''Helper for plotting distributions.'''
    vals = np.array(vals)
    low, high = np.percentile(vals, [lower, upper])
    return (vals[(vals >= low) & (vals <= high)])
    
def plot_distributions_across_runs(results, excluding=(), bins=15):
    '''Foreach attribute in the given Stats object, this'll plot a histogram of how that attribute varies across simulation runs.'''

    if (len(results) == 0): # No results to plot.
        return
    
    # - - - DETERMINING ATTRIBUTE NAMES FROM STATS OBJECT - - - - - 
    first = results[0]
    all_attrs = []
    valid_attrs = []

    for attr, value in vars(first).items():
        if (any(x in attr.lower() for x in excluding)):
            continue # Skip if excluded.
        if isinstance(value, dict):
            continue # Skip if dict.
        all_attrs.append(attr)

    # - - - PICKING OUT THE ATTRIBUTES THAT AREN'T VALID (I feel like I could be doing this more elegantly) - - - - -
    for attr in all_attrs:
        vals = []
        for i in results:
            value = getattr(i, attr)

            # Special case for utilizations:
            if attr == "barista_busy_time":
                value = i.barista_utilization()
            elif attr == "register_busy_time":
                value = i.register_utilization()

            if (isinstance(value, list)):
                if (len(value) > 0):
                    vals.append(np.mean(value))
            else:
                vals.append(value)

        if (len(vals) > 0):
            valid_attrs.append(attr)

    num_plots = len(valid_attrs)
    if (num_plots == 0):
        return
    
    # - - - FIG - - - - -

    fig, axes = plt.subplots(num_plots, 1, figsize=(8, (3*num_plots)))
    if (num_plots == 1): # I think it'll never be 1, but just in case...
        axes = [axes]

    for i, attr in zip(axes, valid_attrs):
        vals = []

        # - - - GATHERING FROM ACROSS RUNS - - - - -
        for stats in results:
            value = getattr(stats, attr)

            if attr == "barista_busy_time":
                value = stats.barista_utilization()
            elif attr == "register_busy_time":
                value = stats.register_utilization()

            if (isinstance(value, list)):
                if (len(value) > 0):
                    vals.append(np.mean(value))
            else:
                vals.append(value)

        if (len(vals) == 0): # Skipping empty arrays.
            continue
        
        # - - - PLOTTING HISTOGRAM FOR ATTRIBUTE - - - - -
        vals = trim_outliers(vals, lower=1, upper=99)
        i.hist(vals, bins=bins, color="steelblue", edgecolor="black")
        
        # Special case for utilization:
        if attr == "barista_busy_time":
            label = "barista_utilization"
        elif attr == "register_busy_time":
            label = "register_utilization"
        else:
            label = attr
        
        i.set_title(f"Distribution of '{attr}' across runs.")
        i.set_xlabel(label)
        i.set_ylabel("# Runs")
        i.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()



def plot_customer_waiting_times_multi(results):
    '''The average total waiting time for customer index 0 to N across multiple simulation runs.'''

    min_length          = min(len(result.total_time_in_system) for result in results)                   # Minimum length of total_time_in_system defined, so all runs can be aligned safely...
    n_runs_x_min_length = np.array([result.total_time_in_system[:min_length] for result in results])    # Matrix: (num_runs × min_length)
    mean_vals           = np.mean(n_runs_x_min_length, axis=0)                                                  # Mean and standard deviation.
    std_vals            = np.std(n_runs_x_min_length, axis=0)

    plt.figure(figsize=(10,5))
    plt.plot(mean_vals, label="Mean waiting time (minutes)")
    plt.fill_between(range(min_length), (mean_vals - std_vals), (mean_vals + std_vals), alpha=0.2, label="±1 std dev")
    plt.ylabel("Waiting time (minutes)")
    plt.xlabel("# Customer")
    plt.legend()
    plt.title("Mean total customer waiting time across runs (minutes).")
    plt.show()

def plot_register_queue_waiting_times_multi(results):
    '''Same as plot_customer_waiting_times_multi but for waiting time in queue.'''
    min_length          = min(len(result.register_queue_time) for result in results)
    n_runs_x_min_length = np.array([result.register_queue_time[:min_length] for result in results])

    plt.figure(figsize=(10,5))
    plt.plot(np.mean(n_runs_x_min_length, axis=0))
    plt.ylabel("Register queue time (minutes)")
    plt.xlabel("# Customer")
    plt.title("Mean register queue waiting time across runs (minutes).")
    plt.show()


def plot_all_multi(results):
    plot_all_averages_per_hour(results)
    plot_distributions_across_runs(results, excluding=('arrival_times', 'num_customers', 'run_seed', 'register_time_available', 'barista_time_available',), bins=15) # Excluded measures not appropriate for this type of graph.
    plot_customer_waiting_times_multi(results)
    plot_register_queue_waiting_times_multi(results)

def summarize_metric(values, confidence=0.95):
    '''
    Helper for compute_experiment_metrics.
    Summarizes a given metric. Its mean, std, lower ci, higher ci, and number of replications.
    '''
    value_array         = np.array(values)
    num_replications    = len(value_array)

    if (num_replications == 0):
        return MetricSummary(0, 0, 0, 0, 0) # Nothing to see here.

    mean = float(value_array.mean())
    std  = float(value_array.std(ddof=1)) if (num_replications > 1) else 0.0

    alpha               = (1 - confidence)
    t_critical_value    = t.ppf((1 - alpha/2), df=(num_replications-1)) if (num_replications > 1) else 0.0
    margin              = ((t_critical_value * std) / np.sqrt(num_replications)) if (num_replications > 1) else 0.0

    return MetricSummary(mean, std, mean - margin, mean + margin, num_replications)

def compute_experiment_metrics(exp, alpha=0.5, beta=0.5, confidence=0.95):
    '''
    Computes the mean, std and ci for all metrics of a given Experiment.
    Stores inside the Experiment's self.metrics.
    '''

    replications = exp.results["replications"]
    metrics = {}

    for name, extractor in METRICS.items():
        values = [extractor(rep) for rep in replications]
        metrics[name] = summarize_metric(values, confidence=confidence)

    reg_mean = metrics["register_utilization"].mean
    bar_mean = metrics["barista_utilization"].mean
    metrics["staff_utilization"] = MetricSummary(mean=((reg_mean + bar_mean) / 2), std=0, ci_low=0, ci_high=0, num_replications=len(replications))  # Custom job

    exp.metrics = metrics


def normalize(x, min_num, max_num, epsilon):
    '''Min-max normalization.'''
    if (max_num - min_num) < epsilon:
        return 0.5  # Essentially the same.
    return ((x - min_num) / (max_num - min_num))


def ci_overlap(ci_1_low, ci_1_high, ci_2_low, ci_2_high):
    '''
    Checks if intervals [1a, 1b] and [2a, 2b] overlap.
    Inputs:
        ci_{i}_low:     Float.  ci_low for exp i.
        ci_{i}_high:    Float.  ci_high for exp i.

    Returns:
        True of confidence intervals overlap, else False.
    '''
    return (max(ci_1_low, ci_2_low) <= min(ci_1_high, ci_2_high))


def compare_ci_overlap(exp_1, exp_2, metric_names=None):
    '''
    Compare CI overlap between two experiments for the given metric names.

    Inputs:
        exp_1:          Experiment.     #1
        exp_2:          Experiment.     #2
        metric_names:   List.           Metric names to compare.
                                        If None, compare all available in exp.metrics.

    Returns:
        A dictionary where each metric maps to exp_1's confidence interval for it, 
        exp_2's confidence interval for it, and whether there's overlap (bool).
    '''
    results = {}

    if metric_names is None:
        metric_names = list(exp_1.metrics.keys())

    # - - - - - - - - - -

    for metric in metric_names:
        metric_1    = exp_1.metrics[metric]
        metric_2    = exp_2.metrics[metric]
        overlap     = ci_overlap(metric_1.ci_low, metric_1.ci_high, metric_2.ci_low, metric_2.ci_high)

        results[metric] = {
            "A_CI":     (metric_1.ci_low, metric_1.ci_high),
            "B_CI":     (metric_2.ci_low, metric_2.ci_high),
            "overlap":  overlap
        }

    return results

def ci_overlap_matrix(experiments, metric_name):
    '''
    Matrix showing CI overlap for a specific metric across ALL experiments

    Inputs:
        experiments:    List of Experiments.    All experiments being compared.
        metric_name:    String.                 Name of metric.

    Returns: 
        List of lists of bools.
    '''
    num_exp = len(experiments)
    matrix = [[None]*num_exp for _ in range(num_exp)]

    for i in range(num_exp):
        for j in range(num_exp):
            metric_1        = experiments[i].metrics[metric_name]
            metric_2        = experiments[j].metrics[metric_name]
            matrix[i][j]    = ci_overlap(metric_1.ci_low, metric_1.ci_high, metric_2.ci_low, metric_2.ci_high)

    return matrix


# - - - - - - - - - 


def experiment_means(exp):
    '''
    Takes the metrics obtained by an experiment, and returns the means across simulation runs.

    Inputs:
        exp:    Experiment. The experiment we'd like to extract these values out of.

    Outputs:
        A dict containing the ID of the experiment, the mean_wait, and mean staff utilization for each staff type seperately AND both combined.
    '''
    
    metrics = exp.metrics
    
    mean_wait                   = metrics["mean_total_time"].mean       # Lower is better.
    mean_register_utilization   = metrics["register_utilization"].mean  # Higher is better for the rest.
    mean_barista_utilization    = metrics["barista_utilization"].mean
    mean_staff_utilization      = metrics["staff_utilization"].mean

    # Instead of computing a score for each experiment 
    # (because it doesn't really make sense to take it out of context and score it on its own),
    # We'll just return these values here.

    return {
        "ID":                           exp.ID,
        "mean_wait":                    mean_wait,
        "mean_register_utilization":    mean_register_utilization,
        "mean_barista_utilization":     mean_barista_utilization,
        "mean_staff_utilization":       mean_staff_utilization
    }

def comparison_printer(ranked):
    for exp in ranked:
        print("-" * 50)
        print(f'Experiment {exp["ID"]}')
        print("------------")
        print(f'composite: {exp["composite"]:.3f}\t<--')
        print(f'mean_wait: {exp["mean_wait"]:.3f}')
        print(f'mean_staff_utilization: {exp["mean_staff_utilization"]:.3f}')
        print(f'\t- mean_register_utilization: {exp["mean_register_utilization"]:.3f}')
        print(f'\t- mean_barista_utilization: {exp["mean_barista_utilization"]:.3f}')
        print(f'Coffee machines: {exp["num_coffee_machines"]}\tOvens: {exp["num_ovens"]}')
    print("-" * 50)

def compare_all(experiments, alpha=0.5, beta=0.5, epsilon=1e-9, print_comparison=False):
    '''
    Compares all experiments, ranks them.

    Inputs:
        experiments:        List of Experiment objects. The experiments we ran.
        alpha:              Weight of customer waiting times. 
        beta:               Weight of staff utilization.
        epsilon:            Tie-breaker.
        print_comparison:   Whether we want the comparison printed or not.

    NOTE: By default, we just say we care equally about customer waiting times and staff utilization. 
    In reality though, we may care more about one over the other.

    Outputs:
        Experiments, ranked.
        Perhaps a printed table, if requested. (Not exactly an "output", just something that happens).
    '''

    # - - - FETHICNG MEAN STATS FOR EACH EXPERIMENT - - - - -
    exp_and_means           = [(exp, experiment_means(exp)) for exp in experiments]
    mean_waits              = np.array([m["mean_wait"] for (_, m) in exp_and_means])
    mean_staff_utilizations = np.array([m["mean_staff_utilization"] for (_, m) in exp_and_means])

    # - - - DEFINING RANGE  - - - - -
    # The minimum and maximum of both mean wait times and staff utilizations (for normalizing).
    min_wait        = mean_waits.min()              # The best mean wait time across experiments.
    max_wait        = mean_waits.max()              # The worst.
    min_utilization = mean_staff_utilizations.min() # Same concept.
    max_utilization = mean_staff_utilizations.max()

    # - - - NORMALIZING (in order to make wait times and utilization scores comparable) AND COMPUTING WEIGHTED COMPOSITE SCORE - - - - - 
    ranked = []

    for exp_obj, means in exp_and_means:

        normalized_wait         = (1 - normalize(means["mean_wait"], min_wait, max_wait, epsilon))
        normalized_utilization  = normalize(means["mean_staff_utilization"], min_utilization, max_utilization, epsilon)
        composite               = ((normalized_wait * alpha) + (normalized_utilization * beta))

        ranked.append({
            "ID": means["ID"],
            "mean_wait": means["mean_wait"],
            "mean_register_utilization": means["mean_register_utilization"],
            "mean_barista_utilization": means["mean_barista_utilization"],
            "mean_staff_utilization": means["mean_staff_utilization"],
            "composite": composite,
            "num_coffee_machines": exp_obj.num_coffee_machines,
            "num_ovens": exp_obj.num_ovens
        })

    # - - - SORTING RANK - - - - -
    ranked = sorted(ranked, key=lambda x: x["composite"], reverse=True)

    # - - - PRINTING - - - - -
    if (print_comparison):
        comparison_printer(ranked)

    # - - - DONE - - - - -
    return ranked



def optimal_setup_printer(results_table, best):
    print("\n| - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - |")
    print("| Number of coffee machines and ovens, ranked by optimality for staffing. |")
    print("| - - - - - - - - | - - - | - - - - - | - - - - - - - - - - - - | - - - - |")
    print("| coffee machines | ovens | mean wait | mean staff utilization  |  score  |")
    for r in results_table:
        print(
            f"| {r['coffee']:>15d} | "
            f"{r['ovens']:>4d} | "
            f"{r['mean_wait']:>8.3f} | "
            f"{r['mean_staff_utilization']:>22.3f} | "
            f"{r['composite']:>10.3f} |"
        )
    print("| - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - |")
    print("| The optimal amount of machines for this staffing configuration is:      |")
    print(f"| - '{best['coffee']}' coffee machines and '{best['ovens']}' ovens.\t\t\t\t\t  |")
    print("| - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - |")


def find_optimal_setup_for_staffing(experiment, machinery_list, alpha=0.5, beta=0.5, confidence=0.95, epsilon=1e-9, seed=None, print_comparison=False):
    '''
    For each staffing configuration (bare minimum, medium, high and full) there will be a different optimal number of ovens and coffee machines. 
    If you only have 1 barista on duty, it won't matter if there's one or two coffee machines. If there's 4, then the limited amount of coffee machines may become a limiting factor.

    This function runs the same staffing configuration with all machine configurations, comparing the results and determining the optimal amount of machines for the staffing.
    Such that we're not being wasteful (too many machines) and not being inefficient (too few).

    Inputs:
        - The experiment and its configurations.
        - A list of machine configurations (how many of each). 

    Returns:
        - a table of results and the best (coffee, ovens) combination.
    '''
    
    results_table = []
    all_experiments = []

    for coffee_machines, ovens in machinery_list:
        tmp = clone_experiment(experiment)
        tmp.num_coffee_machines = coffee_machines
        tmp.num_ovens = ovens

        tmp.results = {"replications": run_simulations(tmp, seed)}
        compute_experiment_metrics(
            exp=tmp,
            alpha=alpha,
            beta=beta,
            confidence=confidence
        )

        means = experiment_means(tmp)

        results_table.append({
            "coffee": coffee_machines,
            "ovens": ovens,
            "mean_wait": means["mean_wait"],
            "mean_staff_utilization": means["mean_staff_utilization"],
            #"composite": composite
        })
        if (print_comparison):
            all_experiments.append(tmp)

    # If we wish to compare as we go:
    if (print_comparison):
        print("-" * 50)
        print("TRYING DIFFERING EQUIPMENT SETUPS")
        print("(ranked from best to worst)")
        compare_all(
            experiments=all_experiments,
            alpha=alpha,
            beta=beta,
            epsilon=epsilon,
            print_comparison=True
        )

    # - - - FINDING COMPOSITE - - - - -
    mean_waits              = [row["mean_wait"] for row in results_table]
    mean_staff_utilizations = [row["mean_staff_utilization"] for row in results_table]
    min_wait                = min(mean_waits)
    max_wait                = max(mean_waits)
    min_utilization         = min(mean_staff_utilizations)
    max_utilization         = max(mean_staff_utilizations)

    for row in results_table:
        normalized_wait = normalize(
            x=row["mean_wait"],
            min_num=min_wait,
            max_num=max_wait,
            epsilon=epsilon
        )
        normalized_utilization = normalize(
            x=row["mean_staff_utilization"],
            min_num=min_utilization,
            max_num=max_utilization,
            epsilon=epsilon
        )
        row["composite"] = (((1 - normalized_wait) * alpha) + (normalized_utilization * beta))

    # - - - SORTING BEST FIRST - - - - -
    results_table.sort(key=lambda d: d["composite"], reverse=True)
    best = results_table[0]
    optimal_setup_printer(results_table, best)
    return best # We could return the table aswell, but I see no good reason to.

def print_the_winner(most_successful_experiment):
    '''
    Prints the relevant stats of the most successful experiment.
    Input:
        most_successful_experiment: Experiment. The parameters of the most successful experiment.
    '''
    
    print(f"\nThe most successful experiment was experiment nr. {most_successful_experiment.ID}.")

    print("\nIt had the following schedule:")
    hours_line = " ".join(f"|   {i+1}   " for i in range(9))
    print("HOUR: ", hours_line, "|")
    shifts_line = " ".join(f"| {i}" for i in most_successful_experiment.shifts.shifts)
    print("SHIFT:", shifts_line, "|")

    print("\nAnd for that staffing, the appropriate amount of machines was found to be:")
    print("|", "-"*20, "|")
    print("| COFFEE MACHINES: ", most_successful_experiment.num_coffee_machines, " |")
    print("| OVENS:           ", most_successful_experiment.num_ovens, " |")
    print("|", "-"*20, "|")

all_experiments = []

def run_experiment(exp_nr, purpose, experiments_list, alpha=0.5, beta=0.5, confidence=0.95, adjust_machines=True, num_coffee_machines=1, num_ovens=1, seeds=exp_seeds):
    '''
    Runs experiments nr. {exp_nr}, which is an experiment with a schedule {purpose} that we're testing.
    The amount of machines (ovens, coffee machines) may be adjusted to be optimal for the amount of staff throughout the day.
    The experiment's parameters and results will be added to {experiments_list}.

    Inputs:
        exp_nr:             Int.    The experiment's ID number.
        purpose:            Shifts. The schedule we're testing out.
        experiments_list:   List.   A list to keep track of experiments.
        adjust_machines:    Bool.   Whether or not to adjust the number of machines to the staffing.

    Outputs:
        The experiment.
    '''

    exp = Experiment(exp_nr)
    exp.shifts = purpose

    machinery = [(1,1), (2,1), (3,1), (1,2), (2,2)] # See table of coffee machines and ovens in the section "The Experimental Setup".

    if (adjust_machines):
        best_machines = find_optimal_setup_for_staffing(
            experiment=exp, 
            machinery_list=machinery, 
            alpha=alpha,
            beta=beta,
            confidence=CONFIDENCE,
            epsilon=TIE_BREAKER,
            seed=seeds[exp_nr],
            print_comparison=True
        )
        exp.num_coffee_machines = best_machines["coffee"]
        exp.num_ovens = best_machines["ovens"]
    else:
        exp.num_coffee_machines = num_coffee_machines
        exp.num_ovens = num_ovens

    exp.results = {"replications": run_simulations(exp, seeds[exp_nr])}
    compute_experiment_metrics(
        exp=exp,
        alpha=alpha,
        beta=beta,
        confidence=confidence
    )
    experiments_list.append(exp)

    return exp
    

exp_nr      = 0
purpose     = BARE_MINIMUM
exp_0       = run_experiment(
                exp_nr=exp_nr, 
                purpose=purpose, 
                experiments_list=all_experiments, 
                alpha=WAIT_WEIGHT,
                beta=UTILIZATION_WEIGHT,
                confidence=CONFIDENCE,
                adjust_machines=False       # With just 1 barista, it's pointless to add machines.
            )

plot_all_multi(exp_0.results['replications'])

exp_nr      = 1
purpose     = AVERAGESTAFFED
exp_1       = run_experiment(
                exp_nr=exp_nr, 
                purpose=purpose, 
                experiments_list=all_experiments, 
                alpha=WAIT_WEIGHT,
                beta=UTILIZATION_WEIGHT,
                confidence=CONFIDENCE,
                adjust_machines=False       # Not adding machines since this is the base case.
            )

plot_all_multi(exp_1.results['replications'])

exp_nr      = 2
purpose     = HIGHSTAFFED
exp_2       = run_experiment(
                exp_nr=exp_nr, 
                purpose=purpose, 
                experiments_list=all_experiments, 
                alpha=WAIT_WEIGHT,
                beta=UTILIZATION_WEIGHT,
                confidence=CONFIDENCE,
                adjust_machines=True
            )

plot_all_multi(exp_2.results['replications'])

exp_nr      = 3
purpose     = FULLSTAFFED_1
exp_3       = run_experiment(
                exp_nr=exp_nr, 
                purpose=purpose, 
                experiments_list=all_experiments, 
                alpha=WAIT_WEIGHT,
                beta=UTILIZATION_WEIGHT,
                confidence=CONFIDENCE,
                adjust_machines=True
            )

plot_all_multi(exp_3.results['replications'])

exp_nr      = 4
purpose     = FULLSTAFFED_2
exp_4       = run_experiment(
                exp_nr=exp_nr, 
                purpose=purpose, 
                experiments_list=all_experiments, 
                alpha=WAIT_WEIGHT,
                beta=UTILIZATION_WEIGHT,
                confidence=CONFIDENCE,
                adjust_machines=True
            )

plot_all_multi(exp_4.results['replications'])

for exp in all_experiments:
    compute_experiment_metrics(
        exp=exp,
        alpha=WAIT_WEIGHT,
        beta=UTILIZATION_WEIGHT,
        confidence=CONFIDENCE
    )

ranked = compare_all(
    experiments=all_experiments, 
    alpha=WAIT_WEIGHT,
    beta=UTILIZATION_WEIGHT,
    epsilon=TIE_BREAKER,
    print_comparison=True
)
most_successful_exp_ID = ranked[0]["ID"]
most_successful_experiment = next(exp for exp in all_experiments if (exp.ID == most_successful_exp_ID))
print_the_winner(most_successful_experiment)

get_ci_matrix_of = ['mean_total_time', 'mean_register_queue', 'register_utilization', 'barista_utilization', 'staff_utilization']

compare_3 = [exp_0, exp_1, exp_4]

for i in get_ci_matrix_of:
    exp_nr = 0
    get_ci_overlap_matrix = ci_overlap_matrix(compare_3, metric_name=(i))
    print(f"CI Overlap Matrix for {i}:")
    print("        0    1    4")
    for exp_nr, row in enumerate(get_ci_overlap_matrix):
        tf_row = ["T" if x else "F" for x in row]
        print(f"   {exp_nr}  {tf_row}")
    print("")