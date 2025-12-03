import simpy
import numpy as np

# =============== OPENING HOURS ===============
OPENING_HOUR = 6      # opnum kl. 06:30
OPENING_MINUTE = 30
DAY_LENGTH_MIN = 9 * 60     # 9 klst opið (06:30–15:30)
CLOSING_TIME_MIN = DAY_LENGTH_MIN
SECONDS = 1/60              # 1 sekúnda í mínútum

# =============== ARRIVAL RATE ===============
def arrival_rate(time_min):
    """
    Arrival-rate (customers per minute) með morgunrush.
    time_min er mínútur frá opnun (0 = 06:30).
    """
    baseline = 0.06      # ~4 á klst
    peak = 0.33          # ~20 á klst
    rush = 120           # toppur í kringum 2 klst eftir opnun (≈ 08:30)
    rush_length = 60
    return baseline + peak * np.exp(-((time_min - rush) ** 2) / (2 * rush_length**2))

# =============== LÍKUR Á VÉLUM ===============
CHANCE_THAT_COFFEE_MACHINE_NEEDED = 0.9
CHANCE_THAT_OVEN_NEEDED = 0.15

# =============== HÉR BREYTIR ÞÚ VAKTAPLANINU ===============
# Hver færsla: "HH:MM" : fjöldi barista á vakt í þeirri klukkustund
# 06:30–07:30, 07:30–08:30, 08:30–09:30, ..., 14:30–15:30
barista_schedule_clock = {
    "06:30": 1,
    "07:30": 1,
    "08:30": 2,
    "09:30": 3,
    "10:30": 3,
    "11:30": 2,
    "12:30": 2,
    "13:30": 1,
    "14:30": 1,
}

# =============== STATISTICS ===============
class Stats:
    def __init__(self):
        self.register_queue_time = []
        self.barista_queue_time = []
        self.pickup_queue_time = []
        self.total_time_in_system = []
        self.time_until_prep_begins = []

# =============== DREIFINGAR (allt í mínútum) ===============
def order_time(rng):
    # Gamma + shift, 10–120 sek -> mín
    shift = 10
    k = 4.5
    theta = 10
    sample = shift + rng.gamma(k, theta)
    sample = min(sample, 120)
    return sample * SECONDS

def barista_prep_time(rng):
    # Triangular 15s–3min–10min
    left = 15
    mode = 180
    right = 600
    sample = rng.triangular(left, mode, right)
    return sample * SECONDS

def pickup_time(rng):
    # um 5 sek
    return max(1, rng.normal(5, 1)) * SECONDS

def coffee_machine_time(rng):
    return rng.uniform(60, 180) * SECONDS  # 1–3 min

def oven_time(rng):
    return rng.uniform(60, 300) * SECONDS  # 1–5 min

# =============== VÉLAR ===============
def use_coffee_machine(env, coffee_machine, rng):
    with coffee_machine.request() as req:
        yield req
        yield env.timeout(coffee_machine_time(rng))

def use_oven(env, oven, rng):
    with oven.request() as req:
        yield req
        yield env.timeout(oven_time(rng))

# =============== HJÁLPARFALL FYRIR TÍMA ===============
def clock_to_offset_min(clock_str):
    """
    Tekur 'HH:MM' og skilar mínútum frá opnun (06:30 -> 0, 07:30 -> 60, o.s.frv.)
    """
    h, m = map(int, clock_str.split(":"))
    total_min = h * 60 + m
    opening_total_min = OPENING_HOUR * 60 + OPENING_MINUTE
    return total_min - opening_total_min

def build_barista_schedule_minutes(barista_schedule_clock):
    """
    Tekur dict {"HH:MM": fjöldi} og skilar lista (offset_min, fjöldi), sortað eftir tíma.
    """
    schedule = []
    for clock, count in barista_schedule_clock.items():
        t_min = clock_to_offset_min(clock)
        schedule.append((t_min, count))
    schedule.sort(key=lambda x: x[0])
    return schedule

# =============== STAFF SCHEDULE PROCESS ===============
def barista_staff_scheduler(env, barista_staff, schedule_minutes):
    """
    schedule_minutes: listi af (time_min_from_open, count_baristas).
    Notum simpy.Container til að tákna fjölda barista á vakt hverju sinni.
    """
    # Fyrsta gildi er upphafslevel, rest eru breytingar
    for (t, count) in schedule_minutes[1:]:
        if t > env.now:
            yield env.timeout(t - env.now)
        diff = count - barista_staff.level
        if diff > 0:
            yield barista_staff.put(diff)
        elif diff < 0:
            yield barista_staff.get(-diff)
        print(f"[{env.now:6.1f} min] baristar á vakt: {barista_staff.level}")

# =============== CUSTOMER PROCESS ===============
def customer(env, stats, ID,
             counter, barista_staff, pickup,
             coffee_machine, oven, rng):
    arrival = env.now

    # --- Afgreiðslustaður (ordering) ---
    with counter.request() as req:
        yield req
        start_ordering = env.now
        yield env.timeout(order_time(rng))
        finish_ordering = env.now

    # --- Baristar (með staff schedule) ---
    yield barista_staff.get(1)  # bíður eftir lausum barista
    prep_start = env.now

    yield env.timeout(barista_prep_time(rng))

    if rng.random() < CHANCE_THAT_COFFEE_MACHINE_NEEDED:
        yield env.process(use_coffee_machine(env, coffee_machine, rng))

    if rng.random() < CHANCE_THAT_OVEN_NEEDED:
        yield env.process(use_oven(env, oven, rng))

    order_ready = env.now
    yield barista_staff.put(1)  # baristi losnar

    # --- Pickup ---
    with pickup.request() as req:
        yield req
        start_pickup = env.now
        yield env.timeout(pickup_time(rng))
        end_time = env.now

    # --- Stats ---
    stats.register_queue_time.append(start_ordering - arrival)
    stats.barista_queue_time.append(prep_start - finish_ordering)
    stats.pickup_queue_time.append(start_pickup - order_ready)
    stats.total_time_in_system.append(end_time - arrival)
    stats.time_until_prep_begins.append(prep_start - finish_ordering)

# =============== ARRIVALS ===============
def arrivals_generator(env, stats,
                       counter, barista_staff, pickup,
                       coffee_machine, oven,
                       arrival_rate, closing_time_min, seed):
    rng = np.random.default_rng(seed)
    cust_id = 0
    inter_arrival = 0

    while True:
        if env.now + inter_arrival >= closing_time_min:
            return

        rate = arrival_rate(env.now)
        inter_arrival = rng.exponential(1 / rate)
        yield env.timeout(inter_arrival)

        env.process(customer(env, stats, cust_id,
                             counter, barista_staff, pickup,
                             coffee_machine, oven, rng))
        cust_id += 1

# =============== SIMULATION WRAPPER ===============
def simulate(num_counter_staff, num_pickup_spots,
             num_coffee_machines, num_ovens,
             barista_schedule_clock,
             closing_time_min, ttl, seed=0):

    env = simpy.Environment()
    stats = Stats()

    # 1 kassi/borð, en capacity = fjöldi sem geta tekið pöntun á sama tíma
    counter = simpy.Resource(env, capacity=num_counter_staff)

    pickup = simpy.Resource(env, capacity=num_pickup_spots)
    coffee_machine = simpy.Resource(env, capacity=num_coffee_machines)
    oven = simpy.Resource(env, capacity=num_ovens)

    schedule_min = build_barista_schedule_minutes(barista_schedule_clock)
    initial_baristas = schedule_min[0][1]
    barista_staff = simpy.Container(env, init=initial_baristas, capacity=20)

    env.process(barista_staff_scheduler(env, barista_staff, schedule_min))

    env.process(arrivals_generator(
        env, stats,
        counter, barista_staff, pickup,
        coffee_machine, oven,
        arrival_rate, closing_time_min, seed
    ))

    env.run(until=ttl)
    return stats

# =============== MAIN: DÆMI ===============
if __name__ == "__main__":
    stats = simulate(
        num_counter_staff=1,          # hversu margir að afgreiða pantanir
        num_pickup_spots=1,           # pickup stöðvar
        num_coffee_machines=1,        # kaffivél
        num_ovens=1,                  # ofn
        barista_schedule_clock=barista_schedule_clock,
        closing_time_min=CLOSING_TIME_MIN,
        ttl=CLOSING_TIME_MIN + 60,    # keyrum aðeins lengur
        seed=42
    )

    print(f"Fjöldi viðskiptavina: {len(stats.total_time_in_system)}")
    print(f"Meðaltími í kerfi (mín): {np.mean(stats.total_time_in_system):.2f}")
