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

def check_env():
    """Verify environment dependencies."""
    if os.environ.get("CONDA_DEFAULT_ENV") != "nexus":
        print("[ERROR] Must be run in 'nexus' conda environment.")
        return False
    return True

# logger will be initialized in main()
logger = None
CONFIG = {}

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
# --- Fullstack / Software Engineering Focused (new) ---

GENERATOR_WEIGHTS.update({
    # Architecture & high-level reasoning
    "fs_arch_monolith_vs_microservices": 4_000_000,
    "fs_arch_layered": 4_000_000,
    "fs_arch_clean_hexagonal": 4_000_000,
    "fs_arch_event_driven": 4_000_000,
    "fs_arch_scalability_patterns": 4_000_000,
    "fs_arch_observability": 4_000_000,

    # Backend / API
    "fs_api_rest_crud": 4_000_000,
    "fs_api_rest_errors": 4_000_000,
    "fs_api_pagination": 4_000_000,
    "fs_api_graphql_schema": 4_000_000,
    "fs_api_async_jobs": 4_000_000,
    "fs_api_validation_schemas": 4_000_000,
    "fs_api_file_uploads": 4_000_000,
    "fs_api_rate_limiting": 4_000_000,

    # Database / schema
    "fs_db_schema_design": 4_000_000,
    "fs_db_relations": 4_000_000,
    "fs_db_migrations": 4_000_000,
    "fs_db_indexes": 4_000_000,
    "fs_db_multi_tenancy": 4_000_000,
    "fs_db_transactions": 4_000_000,

    # Frontend / UI
    "fs_ui_crud_forms": 4_000_000,
    "fs_ui_data_tables": 4_000_000,
    "fs_ui_state_management": 4_000_000,
    "fs_ui_routing": 4_000_000,
    "fs_ui_accessibility": 4_000_000,
    "fs_ui_design_systems": 4_000_000,
    "fs_ui_client_fetching": 4_000_000,

    # Auth & security
    "fs_auth_session_vs_jwt": 4_000_000,
    "fs_auth_rbac_abac": 4_000_000,
    "fs_auth_input_sanitization": 4_000_000,
    "fs_auth_password_flows": 4_000_000,
    "fs_auth_oauth_oidc": 4_000_000,
    "fs_auth_audit_logging": 4_000_000,

    # DevOps / deployment
    "fs_devops_dockerization": 4_000_000,
    "fs_devops_compose_k8s": 4_000_000,
    "fs_devops_ci_cd": 4_000_000,
    "fs_devops_env_config": 4_000_000,
    "fs_devops_monitoring": 4_000_000,
    "fs_devops_zero_downtime": 4_000_000,

    # Testing / quality
    "fs_test_unit": 4_000_000,
    "fs_test_integration": 4_000_000,
    "fs_test_e2e": 4_000_000,
    "fs_test_fixtures": 4_000_000,
    "fs_test_performance": 4_000_000,
    "fs_test_quality_guidelines": 4_000_000,

    # Refactoring & maintenance
    "fs_refactor_extract_function": 4_000_000,
    "fs_refactor_extract_module": 4_000_000,
    "fs_refactor_rename": 4_000_000,
    "fs_refactor_reduce_duplication": 4_000_000,
    "fs_refactor_api_cleanup": 4_000_000,

    # Project scaffolding
    "fs_proj_readme": 4_000_000,
    "fs_proj_folder_structure": 4_000_000,
    "fs_proj_coding_guidelines": 4_000_000,
    "fs_proj_onboarding_docs": 4_000_000,
    "fs_proj_release_process": 4_000_000,

    # TIER 1: HIGH PRIORITY NEW CATEGORIES
    "fs_api_websockets": 4_000_000,
    "fs_error_handling_patterns": 4_000_000,
    "fs_tracing_observability": 4_000_000,
    "fs_caching_strategies": 4_000_000,
    "fs_message_queues": 4_000_000,

    # TIER 2: MEDIUM PRIORITY NEW CATEGORIES
    "fs_search_indexing": 4_000_000,
    "fs_data_validation_pipelines": 4_000_000,
    "fs_rate_limiting_throttling": 4_000_000,
    "fs_monitoring_alerting": 4_000_000,
    "fs_feature_flags_ab_testing": 4_000_000,

    # TIER 3: ADDITIONAL CATEGORIES
    "fs_backwards_compatibility": 4_000_000,
    "fs_capacity_planning": 4_000_000,
})


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
    
    # ═══════════════════════════════════════════════════════════════
    # FULLSTACK: ARCHITECTURE
    # ═══════════════════════════════════════════════════════════════

    def gen_fs_arch_monolith_vs_microservices(self) -> Tuple[str, str, str]:
        q = "You are designing a SaaS CRM. Compare monolith vs microservices for this use case."
        ctx = "Discuss deployment complexity, team structure, performance, and failure isolation."
        a = (
            "Monolith:\n"
            "- Pros: Simpler deployment, easier local dev, fewer network hops.\n"
            "- Cons: Hard to scale parts independently, tight coupling, slower releases.\n\n"
            "Microservices:\n"
            "- Pros: Independent scaling, tech stack flexibility, bounded contexts.\n"
            "- Cons: Network overhead, distributed tracing, ops complexity.\n"
            "For an early‑stage CRM with one team, start with a well‑modularized monolith."
        )
        return q, ctx, a

    def gen_fs_arch_layered(self) -> Tuple[str, str, str]:
        q = "Explain the controller‑service‑repository pattern for a typical web backend."
        ctx = "Use an example of a User resource with create and fetch operations."
        a = (
            "Controller: Handles HTTP, parses/validates input, calls service.\n"
            "Service: Encodes business rules (e.g., unique email, password policy).\n"
            "Repository: Talks to the DB (CRUD for User rows).\n"
            "The controller never touches SQL directly; the service does not know SQL details."
        )
        return q, ctx, a

    def gen_fs_arch_clean_hexagonal(self) -> Tuple[str, str, str]:
        q = "Describe hexagonal (ports & adapters) architecture for an e‑commerce service."
        ctx = "Map domain core, ports, and adapters explicitly."
        a = (
            "Domain core: Entities like Order, Customer, Product; use cases like PlaceOrder.\n"
            "Inbound ports: Interfaces used by HTTP controllers, CLI, or queues.\n"
            "Outbound ports: Interfaces for payment, inventory, email.\n"
            "Adapters: Concrete implementations (StripePaymentAdapter, PostgresOrderRepo).\n"
            "This keeps framework/IO details outside the core domain."
        )
        return q, ctx, a

    def gen_fs_arch_event_driven(self) -> Tuple[str, str, str]:
        q = "Design an event‑driven flow when an order is placed in an online store."
        ctx = "Describe events, consumers, and idempotency handling."
        a = (
            "1) 'OrderPlaced' event emitted by the order service.\n"
            "2) Inventory service reserves stock on OrderPlaced.\n"
            "3) Payment service charges the customer.\n"
            "4) Notification service sends confirmation.\n"
            "Consumers store processed event IDs to avoid double‑processing on retry."
        )
        return q, ctx, a

    def gen_fs_arch_scalability_patterns(self) -> Tuple[str, str, str]:
        q = "Explain when to use caching, queues, and read replicas in a web app."
        ctx = "Use a high‑traffic API as an example."
        a = (
            "Caching: Use for expensive reads with high reuse (e.g., product catalog).\n"
            "Queues: Use for slow but non‑interactive work (emails, PDF generation).\n"
            "Read replicas: Serve heavy read traffic without stressing the primary DB.\n"
            "Combine them carefully to keep consistency acceptable for the product."
        )
        return q, ctx, a

    def gen_fs_arch_observability(self) -> Tuple[str, str, str]:
        q = "What should a fullstack team log and trace for a checkout API?"
        ctx = "Think logs, metrics, and traces end‑to‑end."
        a = (
            "Logs: Request IDs, user ID, order ID, error stacks, external call failures.\n"
            "Metrics: Request latency, error rate, payment gateway failures, inventory failures.\n"
            "Traces: Span from HTTP handler → service → DB → payment API.\n"
            "This lets you debug slow checkouts and failed payments quickly."
        )
        return q, ctx, a

    # ═══════════════════════════════════════════════════════════════
    # FULLSTACK: BACKEND & API
    # ═══════════════════════════════════════════════════════════════

    def gen_fs_api_rest_crud(self) -> Tuple[str, str, str]:
        q = "Design REST CRUD endpoints for a BlogPost resource."
        ctx = "Include URIs and HTTP methods, and briefly state what each does."
        a = (
            "Typical CRUD endpoints for BlogPost:\n"
            "- POST /posts: Create a new post. Body includes title, content, author_id.\n"
            "- GET /posts: List posts, optionally with pagination and filters.\n"
            "- GET /posts/{id}: Fetch a single post by ID.\n"
            "- PUT /posts/{id}: Replace an existing post (full update).\n"
            "- PATCH /posts/{id}: Partially update fields (e.g., title only).\n"
            "- DELETE /posts/{id}: Soft or hard delete a post.\n"
            "Use consistent JSON shapes and return created/updated objects with IDs."
        )
        return q, ctx, a

    def gen_fs_api_rest_errors(self) -> Tuple[str, str, str]:
        q = "Describe a consistent error response format for a public JSON API."
        ctx = "Mention HTTP status codes and a JSON structure clients can rely on."
        a = (
            "Use HTTP status codes for broad class (4xx client, 5xx server) and a JSON body like:\n"
            "{\n"
            '  "error": {\n'
            '    "code": "RESOURCE_NOT_FOUND",\n'
            '    "message": "Post not found",\n'
            '    "details": {"post_id": "123"}\n'
            "  }\n"
            "}\n"
            "This lets clients branch on 'code' programmatically and show 'message' to users.\n"
            "Always document possible error codes per endpoint."
        )
        return q, ctx, a

    def gen_fs_api_pagination(self) -> Tuple[str, str, str]:
        q = "Compare offset and cursor pagination for a high-traffic listing endpoint."
        ctx = "Use an example of listing orders for an admin dashboard."
        a = (
            "Offset pagination (?page=3&limit=50) is simple and works well for small datasets,\n"
            "but large OFFSETs become slow and can show duplicates/missing rows when data changes.\n\n"
            "Cursor pagination (?cursor=abc&limit=50) uses a stable, opaque token (e.g., last order ID)\n"
            "to fetch the next page efficiently with 'WHERE id > last_id'. It scales better and handles\n"
            "live updates with fewer glitches, so it's usually preferred for large order tables."
        )
        return q, ctx, a

    def gen_fs_api_graphql_schema(self) -> Tuple[str, str, str]:
        q = "Sketch a simple GraphQL schema for Users and their Posts."
        ctx = "Show type definitions and an example query."
        a = (
            "Types:\n"
            "type User { id: ID!, name: String!, email: String!, posts: [Post!]! }\n"
            "type Post { id: ID!, title: String!, body: String!, author: User! }\n"
            "type Query {\n"
            "  me: User\n"
            "  user(id: ID!): User\n"
            "  posts(limit: Int, offset: Int): [Post!]!\n"
            "}\n\n"
            "Example query:\n"
            "{ me { id name posts { id title } } }\n"
            "The server resolves relationships via resolvers that call your data layer."
        )
        return q, ctx, a

    def gen_fs_api_async_jobs(self) -> Tuple[str, str, str]:
        q = "Explain when to use background jobs instead of synchronous HTTP responses."
        ctx = "Use sending emails and generating PDFs as examples."
        a = (
            "Use background jobs when work is slow, unreliable, or not essential to the immediate response.\n"
            "For example, order placement can return 200 OK once the order is stored, then enqueue jobs to:\n"
            "- send confirmation emails,\n"
            "- generate and store a PDF invoice,\n"
            "- notify external systems.\n"
            "The HTTP handler writes a job to a queue (e.g., Redis, SQS), and workers process it with retries."
        )
        return q, ctx, a

    def gen_fs_api_validation_schemas(self) -> Tuple[str, str, str]:
        q = "How would you validate API request bodies using a schema library?"
        ctx = "Explain the idea using Pydantic or Zod-style schemas."
        a = (
            "Define a schema that encodes expected fields and types, e.g. Pydantic:\n"
            "class CreateUser(BaseModel):\n"
            "    email: EmailStr\n"
            "    name: constr(min_length=1, max_length=100)\n"
            "    age: Optional[int] = None\n\n"
            "The framework parses JSON into this model, raising validation errors automatically.\n"
            "This centralizes validation logic and keeps controllers thin and predictable."
        )
        return q, ctx, a

    def gen_fs_api_file_uploads(self) -> Tuple[str, str, str]:
        q = "Describe a secure file upload flow for user profile pictures."
        ctx = "Mention limits, content-type checks, and storage strategy."
        a = (
            "Typical flow:\n"
            "1) Client requests an upload URL; server issues a signed URL to object storage (e.g., S3).\n"
            "2) Client uploads directly to storage, respecting size limits and allowed content types.\n"
            "3) Server stores only the file key/URL in DB.\n\n"
            "On the server, verify:\n"
            "- Size < configured max (e.g., 5 MB),\n"
            "- Type is image/* and extension is reasonable,\n"
            "- File is scanned if needed for malware.\n"
            "Serve images via a CDN with proper cache and auth rules."
        )
        return q, ctx, a

    def gen_fs_api_rate_limiting(self) -> Tuple[str, str, str]:
        q = "Explain basic rate limiting strategies for a public API."
        ctx = "Cover per-IP and per-API-key limits."
        a = (
            "Common strategies:\n"
            "- Fixed window: Allow N requests per minute per key/IP.\n"
            "- Sliding window: More accurate count over rolling time.\n"
            "- Token bucket: Refill tokens over time, each request consumes one.\n\n"
            "You can store counters in Redis keyed by 'api_key' or 'ip'. When the limit is exceeded,\n"
            "return 429 Too Many Requests with retry-after hints."
        )
        return q, ctx, a

    # ═══════════════════════════════════════════════════════════════
    # FULLSTACK: DATABASE & SCHEMA
    # ═══════════════════════════════════════════════════════════════

    def gen_fs_db_schema_design(self) -> Tuple[str, str, str]:
        q = "Design a relational schema for a simple task management app."
        ctx = "Include users, projects, and tasks."
        a = (
            "Example tables:\n"
            "- users(id PK, email UNIQUE, name, created_at)\n"
            "- projects(id PK, owner_id FK→users.id, name, created_at)\n"
            "- tasks(id PK, project_id FK→projects.id, title, status, assignee_id FK→users.id, due_date)\n\n"
            "This allows each project to have many tasks, and tasks can be assigned to users.\n"
            "Add indexes on (project_id, status) and (assignee_id, status) for common queries."
        )
        return q, ctx, a

    def gen_fs_db_relations(self) -> Tuple[str, str, str]:
        q = "Explain 1-to-many and many-to-many relations with examples."
        ctx = "Use blog posts and tags as a concrete case."
        a = (
            "1‑to‑many: A user has many posts, each post belongs to exactly one user.\n"
            "Table posts has user_id FK→users.id.\n\n"
            "Many‑to‑many: Posts can have many tags, tags can belong to many posts.\n"
            "Use a join table post_tags(post_id FK→posts.id, tag_id FK→tags.id, PRIMARY KEY(post_id, tag_id)).\n"
            "This pattern generalizes to any symmetric N‑to‑N relationships."
        )
        return q, ctx, a

    def gen_fs_db_migrations(self) -> Tuple[str, str, str]:
        q = "Describe a safe process to add a non-nullable column to a large table."
        ctx = "Avoid downtime and failing writes."
        a = (
            "Safe sequence:\n"
            "1) Add the column as NULLable with a default (or no constraint yet).\n"
            "2) Backfill it in small batches to avoid long locks.\n"
            "3) Once backfilled, add NOT NULL constraint and default in a separate migration.\n"
            "4) Deploy code that writes this column from the application.\n"
            "This avoids long table locks and failed inserts during the transition."
        )
        return q, ctx, a

    def gen_fs_db_indexes(self) -> Tuple[str, str, str]:
        q = "When and how should you add indexes to a table?"
        ctx = "Use an orders table with queries by user_id and created_at."
        a = (
            "Start from real queries: if you often run 'SELECT * FROM orders WHERE user_id = ? ORDER BY created_at DESC',\n"
            "add an index on (user_id, created_at DESC).\n"
            "Indexes speed lookups but slow writes and consume memory.\n"
            "Monitor slow queries and add targeted indexes, avoiding indexing every column blindly."
        )
        return q, ctx, a

    def gen_fs_db_multi_tenancy(self) -> Tuple[str, str, str]:
        q = "Compare strategies for multi-tenant SaaS databases."
        ctx = "Discuss single DB with tenant_id vs separate DB per tenant."
        a = (
            "Single DB with tenant_id column:\n"
            "- Pros: Simpler operations, shared schema, easy to onboard new tenants.\n"
            "- Cons: Noisy neighbor risk, complex row-level security.\n\n"
            "Separate DB per tenant:\n"
            "- Pros: Strong isolation, easier data export/retire, custom schema per big client.\n"
            "- Cons: More operational overhead, migrations across many DBs.\n"
            "Hybrids exist (per-region DB, big tenants isolated, small tenants shared)."
        )
        return q, ctx, a

    def gen_fs_db_transactions(self) -> Tuple[str, str, str]:
        q = "Give an example of when to use a database transaction."
        ctx = "Use transferring money between two accounts."
        a = (
            "Transferring money requires atomicity:\n"
            "1) Subtract amount from source account.\n"
            "2) Add amount to destination account.\n"
            "If one succeeds and the other fails, balances are inconsistent.\n\n"
            "Wrap both updates in a transaction; on failure, rollback both so the system remains consistent."
        )
        return q, ctx, a

    # ═══════════════════════════════════════════════════════════════
    # FULLSTACK: FRONTEND & UI
    # ═══════════════════════════════════════════════════════════════

    def gen_fs_ui_crud_forms(self) -> Tuple[str, str, str]:
        q = "Describe best practices for a Create/Edit form in a web app."
        ctx = "Consider validation, UX, and API integration."
        a = (
            "Use labeled inputs with clear placeholders and inline validation messages.\n"
            "Disable submit while a request is in-flight and show a spinner.\n"
            "On error, display field-specific messages (e.g., invalid email) plus a generic banner.\n"
            "On success, either clear the form or navigate to a detail page and show a toast."
        )
        return q, ctx, a

    def gen_fs_ui_data_tables(self) -> Tuple[str, str, str]:
        q = "What should a good data table component provide?"
        ctx = "Think about UX for large lists like orders."
        a = (
            "Key features:\n"
            "- Sortable columns (by date, status, amount).\n"
            "- Text filter and status filters.\n"
            "- Pagination or infinite scroll.\n"
            "- Clear empty/loading/error states.\n"
            "Use responsive design so the table works on smaller screens (stack columns or show detail drawer)."
        )
        return q, ctx, a

    def gen_fs_ui_state_management(self) -> Tuple[str, str, str]:
        q = "When should you use local vs global state on the frontend?"
        ctx = "Example: React app with filters, modals, and authenticated user."
        a = (
            "Local state is ideal for component-scoped concerns (input values, open/closed modals).\n"
            "Global state is better for cross-cutting data like 'current user', feature flags, or cart contents.\n"
            "Avoid putting everything in global state; it harms performance and traceability.\n"
            "Use server-state libraries (React Query/SWR) for data fetched from APIs."
        )
        return q, ctx, a

    def gen_fs_ui_routing(self) -> Tuple[str, str, str]:
        q = "Sketch a route structure for a simple dashboard app."
        ctx = "Include public and private routes."
        a = (
            "Example routes:\n"
            "- /login, /signup (public)\n"
            "- / (redirect to /dashboard)\n"
            "- /dashboard (overview)\n"
            "- /projects, /projects/:id\n"
            "- /settings/profile, /settings/security\n"
            "Wrap private routes in an auth guard that redirects unauthenticated users to /login."
        )
        return q, ctx, a

    def gen_fs_ui_accessibility(self) -> Tuple[str, str, str]:
        q = "List key accessibility practices for forms and buttons."
        ctx = "Assume basic HTML/React app."
        a = (
            "Use <label> with 'for' pointing to input 'id'.\n"
            "Ensure sufficient color contrast for text and buttons.\n"
            "Make all interactive elements keyboard-focusable with visible focus rings.\n"
            "Provide aria-labels where necessary and use semantic HTML elements (button, nav, main)."
        )
        return q, ctx, a

    def gen_fs_ui_design_systems(self) -> Tuple[str, str, str]:
        q = "What is a design system and how does it help frontend teams?"
        ctx = "Mention components, tokens, and consistency."
        a = (
            "A design system is a shared set of components, styles, and guidelines.\n"
            "It includes design tokens (colors, spacing, typography), reusable components (Button, Card, Modal),\n"
            "and usage guidelines. It speeds up development, enforces consistency, and makes global redesigns easier.\n"
            "Teams can implement it in Storybook and reuse components across apps."
        )
        return q, ctx, a

    def gen_fs_ui_client_fetching(self) -> Tuple[str, str, str]:
        q = "Explain good patterns for fetching data on the client."
        ctx = "Include caching and error handling."
        a = (
            "Use a data-fetching library (React Query, SWR) that handles caching, deduplication, and retries.\n"
            "Keep fetching logic near the components that need it, or in hooks like useUser() and useOrders().\n"
            "Show skeletons during loading and friendly messages on errors, with a retry button.\n"
            "Refetch in the background to keep data fresh without jarring reloads."
        )
        return q, ctx, a

    # ═══════════════════════════════════════════════════════════════
    # FULLSTACK: AUTH & SECURITY
    # ═══════════════════════════════════════════════════════════════

    def gen_fs_auth_session_vs_jwt(self) -> Tuple[str, str, str]:
        q = "Compare cookie-based session auth and stateless JWT auth."
        ctx = "Focus on web backends."
        a = (
            "Sessions:\n"
            "- Server stores session data keyed by cookie ID.\n"
            "- Easy to revoke; change on server and old sessions break.\n"
            "- Requires sticky session or shared session store in multi-instance setups.\n\n"
            "JWTs:\n"
            "- Encoded user claims signed by server, stored client-side.\n"
            "- Easy horizontal scaling; server just verifies signature.\n"
            "- Revocation is harder; often require short lifetimes + refresh tokens."
        )
        return q, ctx, a

    def gen_fs_auth_rbac_abac(self) -> Tuple[str, str, str]:
        q = "Explain RBAC vs ABAC for authorization."
        ctx = "Give examples in a project management app."
        a = (
            "RBAC (Role-Based Access Control):\n"
            "- Users have roles like admin, manager, member.\n"
            "- Policies are defined by role (e.g., admins can delete projects).\n\n"
            "ABAC (Attribute-Based Access Control):\n"
            "- Decisions use attributes: user.department, resource.owner_id, time of day.\n"
            "- Example: user can edit a task if user.id == task.assignee_id or user.role == 'admin'.\n"
            "ABAC is more flexible but more complex to reason about."
        )
        return q, ctx, a

    def gen_fs_auth_input_sanitization(self) -> Tuple[str, str, str]:
        q = "Summarize common web security risks and basic mitigations."
        ctx = "Include SQL injection, XSS, and CSRF."
        a = (
            "SQL injection: Use parameterized queries or ORM; never string-concatenate user input into SQL.\n"
            "XSS: Escape user-generated content, use frameworks that auto-escape, and apply CSP.\n"
            "CSRF: Use same-site cookies and CSRF tokens for state-changing requests.\n"
            "Centralize input validation and avoid eval/exec on user input."
        )
        return q, ctx, a

    def gen_fs_auth_password_flows(self) -> Tuple[str, str, str]:
        q = "Describe a secure password reset flow."
        ctx = "Mention tokens and expiry."
        a = (
            "Flow:\n"
            "1) User requests reset with their email.\n"
            "2) Server generates a random, single-use token with short expiry and stores its hash.\n"
            "3) Email a link containing the token; don't show whether the email exists.\n"
            "4) User clicks link, sets new password, server verifies token and updates password hash.\n"
            "5) Invalidate the token and all existing sessions."
        )
        return q, ctx, a

    def gen_fs_auth_oauth_oidc(self) -> Tuple[str, str, str]:
        q = "Explain high-level OAuth/OIDC login (\"Sign in with Google\")."
        ctx = "Skip low-level protocol details."
        a = (
            "User clicks 'Sign in with Google'.\n"
            "App redirects them to Google with a client_id and redirect_uri.\n"
            "After login/consent, Google redirects back with a code.\n"
            "The backend exchanges the code for tokens, verifies them, and creates or finds a local user.\n"
            "Subsequent requests use your own session/JWT; you generally don't pass Google tokens to the frontend."
        )
        return q, ctx, a

    def gen_fs_auth_audit_logging(self) -> Tuple[str, str, str]:
        q = "What events should be included in security/audit logs?"
        ctx = "Consider an admin panel for a SaaS product."
        a = (
            "Log security-sensitive actions:\n"
            "- Logins (success/failure), password changes, 2FA enroll/disable.\n"
            "- Role or permission changes.\n"
            "- Data exports and bulk deletes.\n"
            "- Changes to billing or subscription.\n"
            "Include who (user id), what, when, and from where (IP/user-agent). Store logs immutably."
        )
        return q, ctx, a

    # ═══════════════════════════════════════════════════════════════
    # FULLSTACK: DEVOPS & DEPLOYMENT
    # ═══════════════════════════════════════════════════════════════

    def gen_fs_devops_dockerization(self) -> Tuple[str, str, str]:
        q = "Describe best practices for Dockerizing a Python web app."
        ctx = "Think image size and reproducibility."
        a = (
            "Use a multi-stage Dockerfile: build dependencies in one stage, copy only needed artifacts into a slim runtime image.\n"
            "Pin Python and dependency versions.\n"
            "Set a working directory, copy only necessary files, and avoid dev-only artifacts.\n"
            "Run the app with a non-root user and define a clear ENTRYPOINT/CMD."
        )
        return q, ctx, a

    def gen_fs_devops_compose_k8s(self) -> Tuple[str, str, str]:
        q = "Compare docker-compose and Kubernetes for running services."
        ctx = "Use a small team vs growing platform example."
        a = (
            "docker-compose:\n"
            "- Great for local dev and small deployments.\n"
            "- Simple YAML with services, volumes, networks.\n\n"
            "Kubernetes:\n"
            "- Designed for large-scale, highly-available deployments.\n"
            "- More complex concepts: Deployments, Services, Ingress, StatefulSets.\n"
            "Start with docker-compose for dev, move to K8s when you need autoscaling, rollouts, and robust orchestration."
        )
        return q, ctx, a

    def gen_fs_devops_ci_cd(self) -> Tuple[str, str, str]:
        q = "Outline a simple CI/CD pipeline for a monorepo web app."
        ctx = "Include branches, tests, and deployments."
        a = (
            "Typical pipeline:\n"
            "1) On pull request: run linting, unit tests, and build checks.\n"
            "2) On merge to main: run full test suite, build images, push to registry.\n"
            "3) Deploy to staging automatically; run smoke tests.\n"
            "4) Promote to production via manual approval or tags.\n"
            "Store config in code and keep the pipeline definition versioned."
        )
        return q, ctx, a

    def gen_fs_devops_env_config(self) -> Tuple[str, str, str]:
        q = "Explain 12-factor style configuration for a web service."
        ctx = "Mention environment variables and secrets."
        a = (
            "Configuration (DB URLs, API keys, feature flags) should be provided via environment variables or an external config system.\n"
            "Code stays the same across environments; only config changes.\n"
            "Secrets should not be committed to Git; use a secrets manager or encrypted storage.\n"
            "This makes builds reproducible and deployments safer."
        )
        return q, ctx, a

    def gen_fs_devops_monitoring(self) -> Tuple[str, str, str]:
        q = "What should you monitor for a production API?"
        ctx = "Mention metrics, logs, and alerting."
        a = (
            "Monitor:\n"
            "- Latency (p50/p95/p99) per endpoint.\n"
            "- Error rates and specific error codes (5xx, 4xx spikes).\n"
            "- Resource usage (CPU, memory, disk, DB connections).\n"
            "Set alerts when SLOs are breached (e.g., 5xx > 1% for 5 minutes) and route them to on-call channels."
        )
        return q, ctx, a

    def gen_fs_devops_zero_downtime(self) -> Tuple[str, str, str]:
        q = "Describe zero-downtime deployment strategies."
        ctx = "Mention rolling and blue-green."
        a = (
            "Rolling deployments replace instances gradually, taking some out of rotation while new ones start.\n"
            "Blue-green uses two environments: blue (live) and green (new). You switch traffic to green once healthy.\n"
            "Both require health checks and readiness probes so traffic only hits healthy instances.\n"
            "DB migrations must also be compatible with both old and new code during rollout."
        )
        return q, ctx, a

    # ═══════════════════════════════════════════════════════════════
    # FULLSTACK: TESTING & QUALITY
    # ═══════════════════════════════════════════════════════════════

    def gen_fs_test_unit(self) -> Tuple[str, str, str]:
        q = "Explain the Arrange-Act-Assert pattern for unit tests."
        ctx = "Use a simple function as example."
        a = (
            "Arrange: Set up input data and dependencies.\n"
            "Act: Call the function under test.\n"
            "Assert: Check the result and side effects.\n\n"
            "Example:\n"
            "arrange: x = 2, y = 3\n"
            "act: result = add(x, y)\n"
            "assert: result == 5"
        )
        return q, ctx, a

    def gen_fs_test_integration(self) -> Tuple[str, str, str]:
        q = "What is an integration test in a web backend?"
        ctx = "Contrast with unit tests."
        a = (
            "Integration tests verify multiple components working together: HTTP layer, business logic, and DB.\n"
            "For example, hitting POST /users with JSON and asserting that the user exists in the DB afterward.\n"
            "They are slower and more brittle than unit tests but catch wiring and configuration issues."
        )
        return q, ctx, a

    def gen_fs_test_e2e(self) -> Tuple[str, str, str]:
        q = "Describe end-to-end tests for a user signup flow."
        ctx = "Assume Playwright/Cypress-like tooling."
        a = (
            "E2E tests simulate a real user:\n"
            "1) Open the signup page.\n"
            "2) Fill in email/password and submit.\n"
            "3) Assert redirect to dashboard and that a welcome message appears.\n"
            "4) Optionally check that a new user record exists via an API or DB fixture.\n"
            "These tests validate UI, API, and data layer together."
        )
        return q, ctx, a

    def gen_fs_test_fixtures(self) -> Tuple[str, str, str]:
        q = "What are test fixtures and why are they useful?"
        ctx = "Use database seed data as an example."
        a = (
            "Fixtures are reusable setup data or objects for tests.\n"
            "For example, seeding a test DB with a demo user and project so multiple tests can rely on them.\n"
            "They keep test code DRY and make scenarios easier to express ('given an existing project with tasks')."
        )
        return q, ctx, a

    def gen_fs_test_performance(self) -> Tuple[str, str, str]:
        q = "What do you look for in basic performance/load testing?"
        ctx = "Consider an API endpoint with many concurrent users."
        a = (
            "Check how latency and error rate behave as concurrent users increase.\n"
            "Look for saturation points where CPU or DB connections max out.\n"
            "Identify slow endpoints, N+1 queries, and inadequate indexes.\n"
            "Use load tools (k6, Locust, JMeter) and measure before/after optimizations."
        )
        return q, ctx, a

    def gen_fs_test_quality_guidelines(self) -> Tuple[str, str, str]:
        q = "List key code quality guidelines for a backend team."
        ctx = "Provide concise, practical points."
        a = (
            "Guidelines:\n"
            "- Small, focused functions and modules.\n"
            "- Clear naming, avoiding cleverness.\n"
            "- Tests for critical paths and bugfixes.\n"
            "- Consistent formatting via automated tools.\n"
            "- Code review focused on correctness, security, and maintainability."
        )
        return q, ctx, a

    # ═══════════════════════════════════════════════════════════════
    # FULLSTACK: REFACTORING & MAINTENANCE
    # ═══════════════════════════════════════════════════════════════

    def gen_fs_refactor_extract_function(self) -> Tuple[str, str, str]:
        q = "Explain the 'extract function' refactor."
        ctx = "Use a controller with duplicated validation logic as example."
        a = (
            "When duplicate or complex logic appears in multiple places, move it into a dedicated function.\n"
            "Example: validate_user_payload(request.json) defined once and reused in several endpoints.\n"
            "This reduces duplication, makes tests easier to write, and clarifies intent."
        )
        return q, ctx, a

    def gen_fs_refactor_extract_module(self) -> Tuple[str, str, str]:
        q = "When should you split a large module into smaller ones?"
        ctx = "Consider a 'utils.py' file that has grown too big."
        a = (
            "If a single file has many unrelated functions or classes, it's harder to navigate and reuse.\n"
            "Group related logic into modules (e.g., string_utils.py, time_utils.py).\n"
            "This improves discoverability and limits the surface area each import brings in."
        )
        return q, ctx, a

    def gen_fs_refactor_rename(self) -> Tuple[str, str, str]:
        q = "Why and how should you rename functions or variables?"
        ctx = "Consider ambiguous names like 'doStuff' or 'data'."
        a = (
            "Names should describe purpose, not implementation.\n"
            "Rename 'doStuff' to 'send_welcome_email' or 'calculate_invoice_total'.\n"
            "Use IDE-assisted rename to update all references safely.\n"
            "Clear names make reviews and maintenance significantly easier."
        )
        return q, ctx, a

    def gen_fs_refactor_reduce_duplication(self) -> Tuple[str, str, str]:
        q = "Give an example of reducing duplication in request handling code."
        ctx = "Think about similar try/except or logging blocks."
        a = (
            "If many handlers log errors and return JSON in the same way, centralize this in middleware.\n"
            "For example, an error-handling middleware catches exceptions, logs, and returns a uniform response.\n"
            "Controllers then focus on business logic instead of boilerplate."
        )
        return q, ctx, a

    def gen_fs_refactor_api_cleanup(self) -> Tuple[str, str, str]:
        q = "What does it mean to clean up an API surface?"
        ctx = "Consider a service class with many rarely used methods."
        a = (
            "API cleanup involves removing unused or redundant endpoints and methods,\n"
            "standardizing signatures, and deprecating confusing behavior.\n"
            "A smaller, clearer surface is easier to test and less likely to break callers.\n"
            "Deprecate old methods with warnings before removing them."
        )
        return q, ctx, a

    # ═══════════════════════════════════════════════════════════════
    # FULLSTACK: PROJECT SCAFFOLDING
    # ═══════════════════════════════════════════════════════════════

    def gen_fs_proj_readme(self) -> Tuple[str, str, str]:
        q = "What sections should a good README contain for a web app?"
        ctx = "Focus on practical content for new contributors."
        a = (
            "Typical sections:\n"
            "- Project overview and goals.\n"
            "- Tech stack and architecture summary.\n"
            "- Setup instructions (prereqs, env vars, commands).\n"
            "- Running tests and linting.\n"
            "- Deployment notes.\n"
            "- Contribution guidelines and code of conduct."
        )
        return q, ctx, a

    def gen_fs_proj_folder_structure(self) -> Tuple[str, str, str]:
        q = "Suggest a folder structure for a monorepo with frontend and backend."
        ctx = "Keep it simple but scalable."
        a = (
            "Example:\n"
            "- /apps\n"
            "  - /frontend\n"
            "  - /backend\n"
            "- /packages\n"
            "  - /ui (shared components)\n"
            "  - /core (shared domain logic)\n"
            "- /infra (IaC, deployment configs)\n"
            "- /scripts (maintenance scripts)\n"
            "This separates apps and shared libraries clearly."
        )
        return q, ctx, a

    def gen_fs_proj_coding_guidelines(self) -> Tuple[str, str, str]:
        q = "What should team coding guidelines typically cover?"
        ctx = "Aim for a concise list."
        a = (
            "Guidelines usually specify:\n"
            "- Preferred language features and patterns.\n"
            "- Error handling style (exceptions vs error codes).\n"
            "- Logging and metrics conventions.\n"
            "- Commenting and documentation expectations.\n"
            "- Performance and security considerations.\n"
            "They should be maintained and agreed on by the team."
        )
        return q, ctx, a

    def gen_fs_proj_onboarding_docs(self) -> Tuple[str, str, str]:
        q = "What should an onboarding doc include for new engineers?"
        ctx = "Think beyond just 'clone repo and run'."
        a = (
            "Include:\n"
            "- High-level architecture and key components.\n"
            "- How to set up dev environment step by step.\n"
            "- How to run tests and a typical dev workflow.\n"
            "- Access to staging/production (if applicable).\n"
            "- Who to contact for questions in each area.\n"
            "This reduces onboarding time and repeated explanations."
        )
        return q, ctx, a

    def gen_fs_proj_release_process(self) -> Tuple[str, str, str]:
        q = "Describe a simple, robust release process."
        ctx = "Include tagging and changelogs."
        a = (
            "Common flow:\n"
            "1) Merge features into main after review and passing CI.\n"
            "2) Create a version tag (e.g., v1.4.0) and generate a changelog from commits.\n"
            "3) Build and publish artifacts (images, packages).\n"
            "4) Deploy to staging, verify, then deploy to production.\n"
            "5) Record release notes and any manual steps or rollbacks.\n"
            "Automate as much of this as possible in CI/CD."
        )
        return q, ctx, a

    # ═══════════════════════════════════════════════════════════════
    # FULLSTACK: TIER 1 - HIGH PRIORITY NEW CATEGORIES
    # ═══════════════════════════════════════════════════════════════

    def gen_fs_api_websockets(self) -> Tuple[str, str, str]:
        q = "Design a WebSocket-based real-time notification system."
        ctx = "Cover connection handling, message formats, and reconnection."
        a = (
            "Key components:\n"
            "- Connection manager: Track active connections per user with heartbeat/ping.\n"
            "- Message format: JSON with {type, payload, timestamp, id}.\n"
            "- Channels/rooms: Allow subscribing to topics (e.g., order:123).\n"
            "- Reconnection: Client auto-reconnects with exponential backoff.\n"
            "- Server sends missed events since last seen ID on reconnect.\n"
            "Use Redis pub/sub for horizontal scaling across server instances."
        )
        return q, ctx, a

    def gen_fs_error_handling_patterns(self) -> Tuple[str, str, str]:
        q = "Design a layered error handling strategy for a web application."
        ctx = "Cover domain errors, HTTP errors, and logging."
        a = (
            "Error hierarchy:\n"
            "- DomainError: Base class for app-specific errors (NotFound, Unauthorized).\n"
            "- Map domain errors to HTTP status in middleware.\n"
            "- Catch unexpected exceptions at top level, log with context, return 500.\n"
            "- Include correlation IDs in logs and responses for debugging.\n"
            "- Avoid leaking stack traces in production responses.\n"
            "- Use structured logging (JSON) with error codes for alerting."
        )
        return q, ctx, a

    def gen_fs_tracing_observability(self) -> Tuple[str, str, str]:
        q = "How would you implement distributed tracing in a microservices architecture?"
        ctx = "Use OpenTelemetry as an example."
        a = (
            "Distributed tracing flow:\n"
            "1) Generate trace_id at entry point (API gateway or first service).\n"
            "2) Propagate trace_id and span_id in headers (e.g., traceparent).\n"
            "3) Each service creates child spans for its operations.\n"
            "4) Export spans to collector (Jaeger, Zipkin, Tempo).\n"
            "5) Correlate logs using trace_id for unified debugging.\n"
            "OpenTelemetry SDKs handle most of this with auto-instrumentation."
        )
        return q, ctx, a

    def gen_fs_caching_strategies(self) -> Tuple[str, str, str]:
        q = "Compare cache-aside, write-through, and write-behind caching patterns."
        ctx = "Use Redis and a product catalog as an example."
        a = (
            "Cache-aside (lazy loading):\n"
            "- App checks cache first, on miss reads from DB and populates cache.\n"
            "- Pros: Simple, only caches accessed data. Cons: First request slow.\n\n"
            "Write-through:\n"
            "- Writes go to cache AND DB synchronously.\n"
            "- Pros: Cache always consistent. Cons: Write latency.\n\n"
            "Write-behind (write-back):\n"
            "- Writes go to cache, async batch to DB.\n"
            "- Pros: Fast writes. Cons: Risk of data loss if cache fails.\n"
            "Use cache-aside for read-heavy catalogs with TTL for freshness."
        )
        return q, ctx, a

    def gen_fs_message_queues(self) -> Tuple[str, str, str]:
        q = "When would you choose RabbitMQ vs Kafka vs SQS?"
        ctx = "Compare for different use cases."
        a = (
            "RabbitMQ:\n"
            "- Best for: Task queues, RPC, complex routing (exchanges).\n"
            "- Push-based, message acknowledgment.\n\n"
            "Kafka:\n"
            "- Best for: Event streaming, high throughput, replay capability.\n"
            "- Append-only log, consumer groups, data retention.\n\n"
            "SQS:\n"
            "- Best for: Serverless, managed service, simple FIFO queues.\n"
            "- No infrastructure to manage, pay-per-message.\n\n"
            "Use RabbitMQ for task distribution, Kafka for event sourcing, SQS for Lambda triggers."
        )
        return q, ctx, a

    # ═══════════════════════════════════════════════════════════════
    # FULLSTACK: TIER 2 - MEDIUM PRIORITY NEW CATEGORIES
    # ═══════════════════════════════════════════════════════════════

    def gen_fs_search_indexing(self) -> Tuple[str, str, str]:
        q = "Design a product search feature with Elasticsearch."
        ctx = "Cover indexing, querying, and relevance tuning."
        a = (
            "Indexing:\n"
            "- Map product fields: title (text), description (text), price (float), category (keyword).\n"
            "- Use analyzers for stemming, synonyms (sneakers → shoes).\n\n"
            "Querying:\n"
            "- Multi-match on title^3, description^1 for relevance weighting.\n"
            "- Filters for category, price range (cached, doesn't affect score).\n\n"
            "Sync:\n"
            "- Use CDC (change data capture) or batch jobs to keep index fresh.\n"
            "- Alias-based zero-downtime reindexing for schema changes."
        )
        return q, ctx, a

    def gen_fs_data_validation_pipelines(self) -> Tuple[str, str, str]:
        q = "How would you build a data validation pipeline for CSV imports?"
        ctx = "Cover schema validation, error reporting, and partial success."
        a = (
            "Pipeline stages:\n"
            "1) Schema validation: Check required columns, data types per field.\n"
            "2) Row-level validation: Business rules (email format, date ranges).\n"
            "3) Cross-row validation: Uniqueness, referential integrity.\n\n"
            "Error handling:\n"
            "- Collect all errors per row, continue validation.\n"
            "- Return detailed error report (row number, column, message).\n\n"
            "Partial success: Allow importing valid rows, skip invalid.\n"
            "Use streaming for large files to avoid memory issues."
        )
        return q, ctx, a

    def gen_fs_rate_limiting_throttling(self) -> Tuple[str, str, str]:
        q = "Design an advanced rate limiting system with multiple tiers."
        ctx = "Cover per-user, per-IP, and global limits."
        a = (
            "Multi-tier limiting:\n"
            "1) Global: Protect entire system (e.g., 10K req/sec total).\n"
            "2) Per-API-key: Different limits for free/paid tiers.\n"
            "3) Per-endpoint: Expensive operations get lower limits.\n"
            "4) Per-IP: Block abuse from anonymous users.\n\n"
            "Implementation:\n"
            "- Use Redis with Lua scripts for atomic operations.\n"
            "- Sliding window or token bucket algorithms.\n"
            "- Return 429 with Retry-After header and rate limit headers.\n"
            "- Consider graceful degradation vs hard blocking."
        )
        return q, ctx, a

    def gen_fs_monitoring_alerting(self) -> Tuple[str, str, str]:
        q = "Design a monitoring and alerting strategy for a production service."
        ctx = "Cover metrics, SLOs, and alert routing."
        a = (
            "Key metrics (RED method):\n"
            "- Rate: Requests per second.\n"
            "- Errors: 4xx/5xx rate.\n"
            "- Duration: Latency percentiles (p50, p95, p99).\n\n"
            "SLOs and alerting:\n"
            "- Define SLOs: 99.9% availability, p95 latency < 200ms.\n"
            "- Alert on error budget burn rate, not raw thresholds.\n"
            "- Page on-call for critical SLO breaches.\n\n"
            "Tools: Prometheus for metrics, Grafana for dashboards, PagerDuty for alerts.\n"
            "Runbooks for each alert with actionable steps."
        )
        return q, ctx, a

    def gen_fs_feature_flags_ab_testing(self) -> Tuple[str, str, str]:
        q = "Design a feature flag system with A/B testing support."
        ctx = "Cover flag types, assignment, and analytics."
        a = (
            "Flag types:\n"
            "- Boolean: On/off for all users.\n"
            "- Percentage rollout: Gradually enable for N% of users.\n"
            "- User targeting: Enable for specific user segments.\n\n"
            "A/B testing:\n"
            "- Assign users to variants deterministically (hash(user_id + experiment_id)).\n"
            "- Track exposure events and conversion metrics.\n"
            "- Use statistical significance testing before declaring winners.\n\n"
            "Implementation: LaunchDarkly, Flagsmith, or custom with Redis + DB.\n"
            "Always have kill switches for fast rollback."
        )
        return q, ctx, a

    # ═══════════════════════════════════════════════════════════════
    # FULLSTACK: TIER 3 - ADDITIONAL CATEGORIES
    # ═══════════════════════════════════════════════════════════════

    def gen_fs_backwards_compatibility(self) -> Tuple[str, str, str]:
        q = "How do you maintain backwards compatibility when evolving an API?"
        ctx = "Cover versioning, deprecation, and migration strategies."
        a = (
            "Versioning strategies:\n"
            "- URL versioning: /api/v1/, /api/v2/\n"
            "- Header versioning: Accept: application/vnd.api.v2+json\n"
            "- Query parameter: ?version=2\n\n"
            "Backwards compatible changes:\n"
            "- Add optional fields (don't remove or rename existing).\n"
            "- Add new endpoints alongside old ones.\n\n"
            "Deprecation process:\n"
            "1) Announce deprecation with sunset date.\n"
            "2) Add Deprecation header to responses.\n"
            "3) Log usage to track migration progress.\n"
            "4) Remove after sufficient migration period."
        )
        return q, ctx, a

    def gen_fs_capacity_planning(self) -> Tuple[str, str, str]:
        q = "How would you approach capacity planning for a growing service?"
        ctx = "Cover current analysis, growth projections, and scaling decisions."
        a = (
            "Current state analysis:\n"
            "- Measure: RPS, CPU, memory, DB connections, queue depth.\n"
            "- Identify bottlenecks under load testing.\n\n"
            "Growth projection:\n"
            "- Historical growth rate (users, requests).\n"
            "- Planned features that affect load.\n\n"
            "Scaling calculations:\n"
            "- If current pod handles 100 RPS at 70% CPU, need X pods for Y RPS.\n"
            "- DB: Read replicas, connection pooling, sharding thresholds.\n"
            "- Cache: Size based on working set, hit rate targets.\n\n"
            "Plan for 2-3x headroom above projected peak."
        )
        return q, ctx, a


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

            # Fullstack architecture
            "fs_arch_monolith_vs_microservices": self.gen_fs_arch_monolith_vs_microservices,
            "fs_arch_layered": self.gen_fs_arch_layered,
            "fs_arch_clean_hexagonal": self.gen_fs_arch_clean_hexagonal,
            "fs_arch_event_driven": self.gen_fs_arch_event_driven,
            "fs_arch_scalability_patterns": self.gen_fs_arch_scalability_patterns,
            "fs_arch_observability": self.gen_fs_arch_observability,

            # Fullstack backend/API
            "fs_api_rest_crud": self.gen_fs_api_rest_crud,
            "fs_api_rest_errors": self.gen_fs_api_rest_errors,
            "fs_api_pagination": self.gen_fs_api_pagination,
            "fs_api_graphql_schema": self.gen_fs_api_graphql_schema,
            "fs_api_async_jobs": self.gen_fs_api_async_jobs,
            "fs_api_validation_schemas": self.gen_fs_api_validation_schemas,
            "fs_api_file_uploads": self.gen_fs_api_file_uploads,
            "fs_api_rate_limiting": self.gen_fs_api_rate_limiting,

            # Fullstack DB
            "fs_db_schema_design": self.gen_fs_db_schema_design,
            "fs_db_relations": self.gen_fs_db_relations,
            "fs_db_migrations": self.gen_fs_db_migrations,
            "fs_db_indexes": self.gen_fs_db_indexes,
            "fs_db_multi_tenancy": self.gen_fs_db_multi_tenancy,
            "fs_db_transactions": self.gen_fs_db_transactions,

            # Fullstack UI
            "fs_ui_crud_forms": self.gen_fs_ui_crud_forms,
            "fs_ui_data_tables": self.gen_fs_ui_data_tables,
            "fs_ui_state_management": self.gen_fs_ui_state_management,
            "fs_ui_routing": self.gen_fs_ui_routing,
            "fs_ui_accessibility": self.gen_fs_ui_accessibility,
            "fs_ui_design_systems": self.gen_fs_ui_design_systems,
            "fs_ui_client_fetching": self.gen_fs_ui_client_fetching,

            # Fullstack auth
            "fs_auth_session_vs_jwt": self.gen_fs_auth_session_vs_jwt,
            "fs_auth_rbac_abac": self.gen_fs_auth_rbac_abac,
            "fs_auth_input_sanitization": self.gen_fs_auth_input_sanitization,
            "fs_auth_password_flows": self.gen_fs_auth_password_flows,
            "fs_auth_oauth_oidc": self.gen_fs_auth_oauth_oidc,
            "fs_auth_audit_logging": self.gen_fs_auth_audit_logging,

            # Fullstack devops
            "fs_devops_dockerization": self.gen_fs_devops_dockerization,
            "fs_devops_compose_k8s": self.gen_fs_devops_compose_k8s,
            "fs_devops_ci_cd": self.gen_fs_devops_ci_cd,
            "fs_devops_env_config": self.gen_fs_devops_env_config,
            "fs_devops_monitoring": self.gen_fs_devops_monitoring,
            "fs_devops_zero_downtime": self.gen_fs_devops_zero_downtime,

            # Fullstack testing
            "fs_test_unit": self.gen_fs_test_unit,
            "fs_test_integration": self.gen_fs_test_integration,
            "fs_test_e2e": self.gen_fs_test_e2e,
            "fs_test_fixtures": self.gen_fs_test_fixtures,
            "fs_test_performance": self.gen_fs_test_performance,
            "fs_test_quality_guidelines": self.gen_fs_test_quality_guidelines,

            # Fullstack refactoring
            "fs_refactor_extract_function": self.gen_fs_refactor_extract_function,
            "fs_refactor_extract_module": self.gen_fs_refactor_extract_module,
            "fs_refactor_rename": self.gen_fs_refactor_rename,
            "fs_refactor_reduce_duplication": self.gen_fs_refactor_reduce_duplication,
            "fs_refactor_api_cleanup": self.gen_fs_refactor_api_cleanup,

            # Fullstack project scaffolding
            "fs_proj_readme": self.gen_fs_proj_readme,
            "fs_proj_folder_structure": self.gen_fs_proj_folder_structure,
            "fs_proj_coding_guidelines": self.gen_fs_proj_coding_guidelines,
            "fs_proj_onboarding_docs": self.gen_fs_proj_onboarding_docs,
            "fs_proj_release_process": self.gen_fs_proj_release_process,

            # Fullstack TIER 1 - High Priority
            "fs_api_websockets": self.gen_fs_api_websockets,
            "fs_error_handling_patterns": self.gen_fs_error_handling_patterns,
            "fs_tracing_observability": self.gen_fs_tracing_observability,
            "fs_caching_strategies": self.gen_fs_caching_strategies,
            "fs_message_queues": self.gen_fs_message_queues,

            # Fullstack TIER 2 - Medium Priority
            "fs_search_indexing": self.gen_fs_search_indexing,
            "fs_data_validation_pipelines": self.gen_fs_data_validation_pipelines,
            "fs_rate_limiting_throttling": self.gen_fs_rate_limiting_throttling,
            "fs_monitoring_alerting": self.gen_fs_monitoring_alerting,
            "fs_feature_flags_ab_testing": self.gen_fs_feature_flags_ab_testing,

            # Fullstack TIER 3 - Additional
            "fs_backwards_compatibility": self.gen_fs_backwards_compatibility,
            "fs_capacity_planning": self.gen_fs_capacity_planning,
            }
        
        query, context, answer = gen_map[category]()
        
        # Apply repetition (equal distribution)
        style = random.choice(["baseline", "2x", "verbose", "3x"])
        repeated_prompt = self.apply_repetition(query, context, style)
        
        domain = "fullstack_engineering" if category.startswith("fs_") else "factual_knowledge"

        sample = {
            "messages": [
                {"role": "user", "content": repeated_prompt},
                {"role": "assistant", "content": answer}
            ],
            "domain": domain,
            "category": category,
            "repetition_style": style,
            "id": f"rep_{category}_{rstr(8)}",
        }

        
        if self.deduplicator.is_duplicate(sample):
            return None
        
        self.category_counters[category] += 1
        return sample


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

CONFIG = {
    "target_samples": 200_000_000,  # HARD LIMIT
    "samples_per_file": 1_000_000,
    "output_dir": "/mnt/e/data/repetitive-prompt-dataset",
    "train_ratio": 0.95,
    "val_ratio": 0.025,
    "test_ratio": 0.025,
}

def main():
    global logger
    
    if not check_env():
         sys.exit(1)
         
    logger = setup_logger(__name__, "logs/gen_repetitive.log")

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
