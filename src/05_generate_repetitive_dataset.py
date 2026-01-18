#!/usr/bin/env python3
"""
04_generate_repetitive_dataset.py
"The Prompt Repetition Specialist" - Implements arXiv 2512.14982 techniques

Based on: https://arxiv.org/html/2512.14982v1
- Repeats prompts to improve non-reasoning LLM performance
- No latency increase (prefill stage parallelization)
- Variants: 2x, 3x repetition with explicit markers

THIS IS SYNTHETIC DATA - Required for factual knowledge grounding.
Cannot be replaced with real datasets.
"""

import os
import sys
import json
import random
import time
import hashlib
import multiprocessing
import string
import math
from pathlib import Path
from typing import Dict, Tuple, Set

sys.path.insert(0, str(Path(__file__).parent))
from utils.logging_config import setup_logger, log_progress, log_header, log_completion

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════
CONFIG = {
    "target_samples": 200_000_000,  # HARD LIMIT
    "samples_per_file": 1_000_000,
    "output_dir": "/mnt/e/data/repetitive-prompt-dataset",
    "train_ratio": 0.95,
    "val_ratio": 0.025,
    "test_ratio": 0.025,
}

logger = setup_logger(__name__, "logs/gen_repetitive.log")

# ═══════════════════════════════════════════════════════════════
# DEDUPLICATION
# ═══════════════════════════════════════════════════════════════
class DeduplicatedGenerator:
    def __init__(self):
        self.seen_hashes: Set[str] = set()
        self.duplicates_skipped = 0

    def is_duplicate(self, sample: Dict) -> bool:
        if "messages" in sample and len(sample["messages"]) >= 2:
            user = sample["messages"][0].get("content", "")
            answer = sample["messages"][1].get("content", "")
            content = f"{user}|||{answer}"
        else:
            content = str(sample)
        h = hashlib.md5(content.encode()).hexdigest()
        if h in self.seen_hashes:
            self.duplicates_skipped += 1
            return True
        self.seen_hashes.add(h)
        return False


# ═══════════════════════════════════════════════════════════════
# GENERATOR CATEGORIES (50 types - 4M samples each)
# ═══════════════════════════════════════════════════════════════
GENERATOR_WEIGHTS = {
    # Math (10 types)
    "basic_arithmetic": 4_000_000, "percentage_calc": 4_000_000,
    "unit_conversion": 4_000_000, "geometry": 4_000_000,
    "statistics": 4_000_000, "algebra": 4_000_000,
    "compound_interest": 4_000_000, "distance_calc": 4_000_000,
    "time_calc": 4_000_000, "currency_convert": 4_000_000,
    
    # Science (10 types)
    "chemistry": 4_000_000, "physics": 4_000_000,
    "biology": 4_000_000, "astronomy": 4_000_000,
    "periodic_table": 4_000_000, "scientific_notation": 4_000_000,
    "energy_calc": 4_000_000, "density_calc": 4_000_000,
    "ph_calc": 4_000_000, "speed_calc": 4_000_000,
    
    # Geography & History (10 types)
    "capital_cities": 4_000_000, "population": 4_000_000,
    "historical_events": 4_000_000, "time_zones": 4_000_000,
    "country_facts": 4_000_000, "language_facts": 4_000_000,
    "currency_info": 4_000_000, "coordinates": 4_000_000,
    "area_calc": 4_000_000, "historical_dates": 4_000_000,
    
    # Technology (10 types)
    "file_size_convert": 4_000_000, "bandwidth_calc": 4_000_000,
    "storage_calc": 4_000_000, "programming_basics": 4_000_000,
    "algorithm_complexity": 4_000_000, "data_structures": 4_000_000,
    "networking": 4_000_000, "encoding": 4_000_000,
    "hash_functions": 4_000_000, "binary_operations": 4_000_000,
    
    # Business & Daily Life (10 types)
    "bmi_calc": 4_000_000, "calorie_burn": 4_000_000,
    "tip_calc": 4_000_000, "tax_calc": 4_000_000,
    "budget_calc": 4_000_000, "recipe_scale": 4_000_000,
    "temp_conversion": 4_000_000, "sports_stats": 4_000_000,
    "age_calc": 4_000_000, "date_diff": 4_000_000,
}

# Data pools
COUNTRIES = ["USA", "Canada", "UK", "France", "Germany", "Japan", "China", "India", "Brazil", "Australia"]
CITIES = ["New York", "London", "Paris", "Tokyo", "Beijing", "Mumbai", "Sydney", "Toronto", "Berlin", "Rome"]
ELEMENTS = ["Hydrogen", "Helium", "Carbon", "Nitrogen", "Oxygen", "Sodium", "Iron", "Copper", "Zinc", "Gold"]
PLANETS = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]

def rnum(a, b): return random.randint(a, b)
def rfloat(a, b): return round(random.uniform(a, b), 2)
def rstr(n): return ''.join(random.choices(string.ascii_lowercase + string.digits, k=n))


# ═══════════════════════════════════════════════════════════════
# REPETITIVE PROMPTING ENGINE (arXiv 2512.14982)
# ═══════════════════════════════════════════════════════════════
class PromptRepetitionEngine:
    def __init__(self):
        self.deduplicator = DeduplicatedGenerator()
        self.category_counters = {k: 0 for k in GENERATOR_WEIGHTS.keys()}
        
    def apply_repetition(self, query: str, context: str, style: str) -> str:
        """
        4 repetition variants from arXiv 2512.14982:
        1. Baseline (no repetition)
        2. 2x (simple repeat)
        3. Verbose (with "Let me repeat that:")
        4. 3x (triple with markers)
        """
        full_query = f"{context}\n{query}" if context else query
        
        if style == "baseline":
            return full_query
        elif style == "2x":
            return f"{full_query} {full_query}"
        elif style == "verbose":
            return f"{full_query} Let me repeat that: {full_query}"
        elif style == "3x":
            return f"{full_query} Let me repeat that: {full_query} Let me repeat that one more time: {full_query}"
        return full_query
    
    # ═══ MATH GENERATORS ═══
    def gen_basic_arithmetic(self) -> Tuple[str, str, str]:
        a, b = rnum(1, 1000), rnum(1, 1000)
        op = random.choice(["+", "-", "*"])
        result = eval(f"{a} {op} {b}")
        return f"Calculate {a} {op} {b}", f"Numbers: {a}, {b}\nOperation: {op}", str(result)
    
    def gen_percentage(self) -> Tuple[str, str, str]:
        total, pct = rnum(100, 10000), rnum(5, 95)
        result = round(total * pct / 100, 2)
        return f"What is {pct}% of {total}?", f"Total: {total}\nPercentage: {pct}%", str(result)
    
    def gen_unit_conversion(self) -> Tuple[str, str, str]:
        km = rnum(1, 500)
        miles = round(km * 0.621371, 2)
        return f"Convert {km} kilometers to miles", f"Distance: {km} km", f"{miles} miles"
    
    def gen_geometry(self) -> Tuple[str, str, str]:
        r = rnum(1, 50)
        area = round(math.pi * r ** 2, 2)
        return f"Area of circle with radius {r}?", f"Radius: {r}", f"{area} sq units"
    
    def gen_statistics(self) -> Tuple[str, str, str]:
        nums = [rnum(10, 100) for _ in range(5)]
        mean = round(sum(nums) / len(nums), 2)
        return f"Mean of {nums}?", f"Numbers: {nums}", str(mean)
    
    def gen_algebra(self) -> Tuple[str, str, str]:
        a, b = rnum(1, 20), rnum(1, 50)
        x = rnum(1, 10)
        result = a * x + b
        return f"If {a}x + {b} = {result}, find x", f"Equation: {a}x + {b} = {result}", f"x = {x}"
    
    def gen_compound_interest(self) -> Tuple[str, str, str]:
        principal, rate, years = rnum(1000, 100000), rnum(2, 12), rnum(1, 30)
        amount = round(principal * (1 + rate/100) ** years, 2)
        return f"${principal} at {rate}% for {years} years?", f"P: ${principal}\nR: {rate}%\nT: {years}y", f"${amount}"
    
    def gen_distance(self) -> Tuple[str, str, str]:
        speed, time = rnum(30, 150), rnum(1, 10)
        distance = speed * time
        return f"Distance at {speed} km/h for {time} hours?", f"Speed: {speed} km/h\nTime: {time}h", f"{distance} km"
    
    def gen_time(self) -> Tuple[str, str, str]:
        h1, m1 = rnum(0, 23), rnum(0, 59)
        h2, m2 = rnum(0, 23), rnum(0, 59)
        diff_min = abs((h2 * 60 + m2) - (h1 * 60 + m1))
        h, m = diff_min // 60, diff_min % 60
        return f"Time from {h1:02d}:{m1:02d} to {h2:02d}:{m2:02d}?", f"Start: {h1:02d}:{m1:02d}\nEnd: {h2:02d}:{m2:02d}", f"{h}h {m}m"
    
    def gen_currency(self) -> Tuple[str, str, str]:
        amount, rate = rnum(100, 10000), rfloat(0.5, 2.0)
        result = round(amount * rate, 2)
        return f"Convert ${amount} at rate {rate}?", f"Amount: ${amount}\nRate: {rate}", f"${result}"
    
    # ═══ SCIENCE GENERATORS ═══
    def gen_chemistry(self) -> Tuple[str, str, str]:
        elem = random.choice(ELEMENTS)
        symbols = {"Hydrogen": "H", "Helium": "He", "Carbon": "C", "Nitrogen": "N", "Oxygen": "O", 
                   "Sodium": "Na", "Iron": "Fe", "Copper": "Cu", "Zinc": "Zn", "Gold": "Au"}
        return f"Chemical symbol for {elem}?", f"Element: {elem}", symbols.get(elem, "X")
    
    def gen_physics(self) -> Tuple[str, str, str]:
        m, a = rnum(1, 100), rnum(1, 20)
        force = m * a
        return f"Force on {m}kg mass with {a} m/s² acceleration?", f"Mass: {m}kg\nAccel: {a} m/s²", f"{force} N"
    
    def gen_biology(self) -> Tuple[str, str, str]:
        organelles = ["Mitochondria", "Nucleus", "Ribosome", "Chloroplast", "Golgi Apparatus"]
        functions = {"Mitochondria": "Energy production", "Nucleus": "Genetic control", "Ribosome": "Protein synthesis", 
                     "Chloroplast": "Photosynthesis", "Golgi Apparatus": "Protein processing"}
        org = random.choice(organelles)
        return f"Function of {org}?", f"Organelle: {org}", functions[org]
    
    def gen_astronomy(self) -> Tuple[str, str, str]:
        planet = random.choice(PLANETS)
        positions = {"Mercury": 1, "Venus": 2, "Earth": 3, "Mars": 4, "Jupiter": 5, "Saturn": 6, "Uranus": 7, "Neptune": 8}
        return f"Position of {planet} from Sun?", f"Planet: {planet}", f"{positions[planet]}th"
    
    def gen_periodic_table(self) -> Tuple[str, str, str]:
        elem = random.choice(ELEMENTS)
        atomic_nums = {"Hydrogen": 1, "Helium": 2, "Carbon": 6, "Nitrogen": 7, "Oxygen": 8, 
                       "Sodium": 11, "Iron": 26, "Copper": 29, "Zinc": 30, "Gold": 79}
        return f"Atomic number of {elem}?", f"Element: {elem}", str(atomic_nums.get(elem, 0))
    
    def gen_scientific_notation(self) -> Tuple[str, str, str]:
        num = rnum(1000, 999999)
        exp = len(str(num)) - 1
        mantissa = num / (10 ** exp)
        return f"Scientific notation for {num}?", f"Number: {num}", f"{mantissa:.2f} × 10^{exp}"
    
    def gen_energy(self) -> Tuple[str, str, str]:
        m, v = rnum(1, 100), rnum(1, 50)
        ke = 0.5 * m * v ** 2
        return f"Kinetic energy: {m}kg at {v} m/s?", f"Mass: {m}kg\nVelocity: {v} m/s", f"{ke} J"
    
    def gen_density(self) -> Tuple[str, str, str]:
        mass, volume = rnum(10, 500), rnum(5, 100)
        density = round(mass / volume, 2)
        return f"Density: {mass}g in {volume}cm³?", f"Mass: {mass}g\nVolume: {volume}cm³", f"{density} g/cm³"
    
    def gen_ph(self) -> Tuple[str, str, str]:
        ph = round(random.uniform(0, 14), 1)
        nature = "Acidic" if ph < 7 else ("Neutral" if ph == 7 else "Basic")
        return f"Nature of solution with pH {ph}?", f"pH: {ph}", nature
    
    def gen_speed(self) -> Tuple[str, str, str]:
        distance, time = rnum(10, 500), rnum(1, 10)
        speed = round(distance / time, 2)
        return f"Speed: {distance}m in {time}s?", f"Distance: {distance}m\nTime: {time}s", f"{speed} m/s"
    
    # ═══ GEOGRAPHY & HISTORY ═══
    def gen_capital(self) -> Tuple[str, str, str]:
        capitals = {"USA": "Washington DC", "UK": "London", "France": "Paris", "Germany": "Berlin", "Japan": "Tokyo"}
        country = random.choice(list(capitals.keys()))
        return f"Capital of {country}?", f"Country: {country}", capitals[country]
    
    def gen_population(self) -> Tuple[str, str, str]:
        city = random.choice(CITIES)
        pop = rnum(1, 40) * 1_000_000
        return f"Approximate population of {city}?", f"City: {city}", f"~{pop//1_000_000}M"
    
    def gen_historical_event(self) -> Tuple[str, str, str]:
        events = {"World War I": 1914, "Moon Landing": 1969, "Fall of Berlin Wall": 1989, "French Revolution": 1789}
        event = random.choice(list(events.keys()))
        return f"Year of {event}?", f"Event: {event}", str(events[event])
    
    def gen_timezone(self) -> Tuple[str, str, str]:
        zones = {"New York": -5, "London": 0, "Tokyo": 9, "Sydney": 10}
        city1, city2 = random.sample(list(zones.keys()), 2)
        diff = zones[city2] - zones[city1]
        return f"Time difference: {city1} to {city2}?", f"{city1} to {city2}", f"{diff:+d} hours"
    
    def gen_country_fact(self) -> Tuple[str, str, str]:
        facts = {"USA": "English", "France": "French", "Germany": "German", "Japan": "Japanese"}
        country = random.choice(list(facts.keys()))
        return f"Official language of {country}?", f"Country: {country}", facts[country]
    
    def gen_language(self) -> Tuple[str, str, str]:
        speakers = {"English": "1.5B", "Spanish": "500M", "French": "280M", "Chinese": "1.3B"}
        lang = random.choice(list(speakers.keys()))
        return f"Speakers of {lang}?", f"Language: {lang}", speakers[lang]
    
    def gen_currency_info(self) -> Tuple[str, str, str]:
        currencies = {"USA": "USD", "UK": "GBP", "Japan": "JPY", "India": "INR"}
        country = random.choice(list(currencies.keys()))
        return f"Currency of {country}?", f"Country: {country}", currencies[country]
    
    def gen_coordinates(self) -> Tuple[str, str, str]:
        lat, lon = round(random.uniform(-90, 90), 2), round(random.uniform(-180, 180), 2)
        hemisphere = ("N" if lat >= 0 else "S", "E" if lon >= 0 else "W")
        return f"Hemisphere for ({lat}, {lon})?", f"Lat: {lat}\nLon: {lon}", f"{hemisphere[0]}, {hemisphere[1]}"
    
    def gen_area(self) -> Tuple[str, str, str]:
        length, width = rnum(10, 100), rnum(10, 100)
        area = length * width
        return f"Area: {length}m × {width}m?", f"L: {length}m\nW: {width}m", f"{area} m²"
    
    def gen_historical_date(self) -> Tuple[str, str, str]:
        events = {"Internet": 1991, "DNA Discovery": 1953, "Penicillin": 1928}
        event = random.choice(list(events.keys()))
        return f"Year of {event} discovery?", f"Discovery: {event}", str(events[event])
    
    # ═══ TECHNOLOGY ═══
    def gen_file_size(self) -> Tuple[str, str, str]:
        mb = rnum(1, 10000)
        gb = round(mb / 1024, 2)
        return f"Convert {mb} MB to GB?", f"Size: {mb} MB", f"{gb} GB"
    
    def gen_bandwidth(self) -> Tuple[str, str, str]:
        mbps, file_mb = rnum(10, 1000), rnum(100, 5000)
        seconds = round(file_mb * 8 / mbps, 1)
        return f"Download time: {file_mb}MB at {mbps} Mbps?", f"File: {file_mb}MB\nSpeed: {mbps} Mbps", f"{seconds}s"
    
    def gen_storage(self) -> Tuple[str, str, str]:
        total, used = rnum(256, 2000), rnum(50, 1500)
        free = max(0, total - used)
        return f"Free space: {total}GB total, {used}GB used?", f"Total: {total}GB\nUsed: {used}GB", f"{free}GB"
    
    def gen_programming(self) -> Tuple[str, str, str]:
        data_types = {"int": "Integer", "str": "String", "list": "Array", "dict": "Object"}
        dtype = random.choice(list(data_types.keys()))
        return f"Python type {dtype} represents?", f"Type: {dtype}", data_types[dtype]
    
    def gen_algorithm_complexity(self) -> Tuple[str, str, str]:
        algos = {"Binary Search": "O(log n)", "Merge Sort": "O(n log n)", "Linear Search": "O(n)"}
        algo = random.choice(list(algos.keys()))
        return f"Time complexity of {algo}?", f"Algorithm: {algo}", algos[algo]
    
    def gen_data_structures(self) -> Tuple[str, str, str]:
        structures = {"Stack": "LIFO", "Queue": "FIFO", "Heap": "Priority Queue"}
        struct = random.choice(list(structures.keys()))
        return f"Ordering principle of {struct}?", f"Structure: {struct}", structures[struct]
    
    def gen_networking(self) -> Tuple[str, str, str]:
        ports = {"HTTP": 80, "HTTPS": 443, "SSH": 22, "FTP": 21}
        protocol = random.choice(list(ports.keys()))
        return f"Default port for {protocol}?", f"Protocol: {protocol}", str(ports[protocol])
    
    def gen_encoding(self) -> Tuple[str, str, str]:
        text = rstr(4)
        encoded = text.encode().hex()
        return f"Hex encoding of '{text}'?", f"Text: {text}", encoded
    
    def gen_hash(self) -> Tuple[str, str, str]:
        text = rstr(8)
        hashed = hashlib.md5(text.encode()).hexdigest()[:8]
        return f"MD5 hash (first 8 chars) of '{text}'?", f"Input: {text}", hashed
    
    def gen_binary(self) -> Tuple[str, str, str]:
        num = rnum(0, 255)
        binary = bin(num)[2:].zfill(8)
        return f"Binary of {num}?", f"Number: {num}", binary
    
    # ═══ BUSINESS & DAILY LIFE ═══
    def gen_bmi(self) -> Tuple[str, str, str]:
        weight, height = rnum(50, 120), rfloat(1.5, 2.0)
        bmi = round(weight / (height ** 2), 1)
        return f"BMI: {weight}kg, {height}m tall?", f"Weight: {weight}kg\nHeight: {height}m", str(bmi)
    
    def gen_calorie(self) -> Tuple[str, str, str]:
        activity = random.choice(["Running", "Swimming", "Cycling", "Walking"])
        rates = {"Running": 10, "Swimming": 8, "Cycling": 7, "Walking": 4}
        mins = rnum(15, 120)
        burned = rates[activity] * mins
        return f"Calories burned: {activity} for {mins} mins?", f"Activity: {activity}\nDuration: {mins}min", f"{burned} cal"
    
    def gen_tip(self) -> Tuple[str, str, str]:
        bill, tip_pct = rnum(20, 500), random.choice([15, 18, 20, 22, 25])
        tip = round(bill * tip_pct / 100, 2)
        return f"{tip_pct}% tip on ${bill}?", f"Bill: ${bill}\nTip: {tip_pct}%", f"${tip}"
    
    def gen_tax(self) -> Tuple[str, str, str]:
        amount, tax_rate = rnum(100, 10000), rnum(5, 25)
        tax = round(amount * tax_rate / 100, 2)
        return f"Tax on ${amount} at {tax_rate}%?", f"Amount: ${amount}\nRate: {tax_rate}%", f"${tax}"
    
    def gen_budget(self) -> Tuple[str, str, str]:
        income, expenses = rnum(3000, 10000), rnum(1000, 8000)
        savings = max(0, income - expenses)
        return f"Savings: ${income} income, ${expenses} expenses?", f"Income: ${income}\nExpenses: ${expenses}", f"${savings}"
    
    def gen_recipe(self) -> Tuple[str, str, str]:
        original, from_serv, to_serv = rnum(1, 5), rnum(4, 8), rnum(8, 24)
        scaled = round(original * to_serv / from_serv, 2)
        return f"Scale {original} cups from {from_serv} to {to_serv} servings?", f"Original: {original}\nFrom: {from_serv}\nTo: {to_serv}", f"{scaled} cups"
    
    def gen_temp(self) -> Tuple[str, str, str]:
        celsius = rnum(0, 250)
        fahrenheit = round(celsius * 9/5 + 32, 1)
        return f"Convert {celsius}°C to Fahrenheit?", f"Temperature: {celsius}°C", f"{fahrenheit}°F"
    
    def gen_sports(self) -> Tuple[str, str, str]:
        wins, losses = rnum(20, 100), rnum(10, 80)
        pct = round(wins / (wins + losses) * 100, 1)
        return f"Win percentage: {wins} wins, {losses} losses?", f"W: {wins}\nL: {losses}", f"{pct}%"
    
    def gen_age(self) -> Tuple[str, str, str]:
        birth_year = rnum(1950, 2020)
        age = 2026 - birth_year
        return f"Age in 2026 if born in {birth_year}?", f"Birth Year: {birth_year}", f"{age} years"
    
    def gen_date_diff(self) -> Tuple[str, str, str]:
        days = rnum(1, 365)
        weeks = days // 7
        return f"How many weeks in {days} days?", f"Days: {days}", f"{weeks} weeks"
    
    def generate_trajectory(self) -> Dict:
        """Generate a single trajectory with prompt repetition."""
        available = [c for c, t in GENERATOR_WEIGHTS.items() if self.category_counters[c] < t]
        if not available:
            return None
        
        category = random.choice(available)
        
        # Map to generator
        gen_map = {
            "basic_arithmetic": self.gen_basic_arithmetic, "percentage_calc": self.gen_percentage,
            "unit_conversion": self.gen_unit_conversion, "geometry": self.gen_geometry,
            "statistics": self.gen_statistics, "algebra": self.gen_algebra,
            "compound_interest": self.gen_compound_interest, "distance_calc": self.gen_distance,
            "time_calc": self.gen_time, "currency_convert": self.gen_currency,
            "chemistry": self.gen_chemistry, "physics": self.gen_physics,
            "biology": self.gen_biology, "astronomy": self.gen_astronomy,
            "periodic_table": self.gen_periodic_table, "scientific_notation": self.gen_scientific_notation,
            "energy_calc": self.gen_energy, "density_calc": self.gen_density,
            "ph_calc": self.gen_ph, "speed_calc": self.gen_speed,
            "capital_cities": self.gen_capital, "population": self.gen_population,
            "historical_events": self.gen_historical_event, "time_zones": self.gen_timezone,
            "country_facts": self.gen_country_fact, "language_facts": self.gen_language,
            "currency_info": self.gen_currency_info, "coordinates": self.gen_coordinates,
            "area_calc": self.gen_area, "historical_dates": self.gen_historical_date,
            "file_size_convert": self.gen_file_size, "bandwidth_calc": self.gen_bandwidth,
            "storage_calc": self.gen_storage, "programming_basics": self.gen_programming,
            "algorithm_complexity": self.gen_algorithm_complexity, "data_structures": self.gen_data_structures,
            "networking": self.gen_networking, "encoding": self.gen_encoding,
            "hash_functions": self.gen_hash, "binary_operations": self.gen_binary,
            "bmi_calc": self.gen_bmi, "calorie_burn": self.gen_calorie,
            "tip_calc": self.gen_tip, "tax_calc": self.gen_tax,
            "budget_calc": self.gen_budget, "recipe_scale": self.gen_recipe,
            "temp_conversion": self.gen_temp, "sports_stats": self.gen_sports,
            "age_calc": self.gen_age, "date_diff": self.gen_date_diff,
        }
        
        query, context, answer = gen_map[category]()
        
        # Apply repetition (equal distribution)
        style = random.choice(["baseline", "2x", "verbose", "3x"])
        repeated_prompt = self.apply_repetition(query, context, style)
        
        sample = {
            "messages": [
                {"role": "user", "content": repeated_prompt},
                {"role": "assistant", "content": answer}
            ],
            "domain": "factual_knowledge",
            "category": category,
            "repetition_style": style,
            "id": f"rep_{category}_{rstr(8)}"
        }
        
        if self.deduplicator.is_duplicate(sample):
            return None
        
        self.category_counters[category] += 1
        return sample


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    log_header(logger, "PROMPT REPETITION DATASET (arXiv 2512.14982)", {
        "Target": CONFIG["target_samples"],
        "Categories": len(GENERATOR_WEIGHTS),
        "Output": CONFIG["output_dir"]
    })
    
    base_dir = Path(CONFIG["output_dir"])
    for split in ["train", "val", "test"]:
        (base_dir / split).mkdir(parents=True, exist_ok=True)
    
    engine = PromptRepetitionEngine()
    samples = []
    count = 0
    batch_num = 0
    
    for i in range(CONFIG["target_samples"]):
        sample = engine.generate_trajectory()
        if sample:
            samples.append(sample)
            count += 1
            
            if len(samples) >= CONFIG["samples_per_file"]:
                r = random.random()
                split = "train" if r < CONFIG["train_ratio"] else ("val" if r < CONFIG["train_ratio"] + CONFIG["val_ratio"] else "test")
                
                out_file = base_dir / split / f"part_{batch_num:04d}.jsonl"
                with open(out_file, 'w') as f:
                    for s in samples:
                        f.write(json.dumps(s) + "\n")
                
                logger.info(f"Wrote {len(samples)} to {out_file}")
                samples = []
                batch_num += 1
        
        if count % 100_000 == 0 and count > 0:
            log_progress(logger, count, CONFIG["target_samples"])
    
    # Write remaining
    if samples:
        out_file = base_dir / "train" / f"part_{batch_num:04d}.jsonl"
        with open(out_file, 'w') as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
    
    log_completion(logger, "Repetitive Prompting Dataset", {"Total": count})


if __name__ == "__main__":
    main()
