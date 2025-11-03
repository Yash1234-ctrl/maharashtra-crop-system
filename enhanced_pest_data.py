"""
Enhanced Pest and Disease Database
Comprehensive pest lifecycle data, management strategies, and crop-specific information
"""

PEST_DATABASE = {
    'Cotton': {
        'Bollworm': {
            'scientific_name': 'Helicoverpa armigera',
            'lifecycle_days': 35,
            'eggs_per_female': 500,
            'favorable_conditions': {
                'temperature': (25, 30),
                'humidity': (60, 80),
                'rainfall': 'moderate'
            },
            'damage_stages': ['Flowering', 'Fruit Development'],
            'symptoms': [
                'Round holes in squares and bolls',
                'Frass (green/brown droppings) near entry holes',
                'Premature flower drop',
                'Damaged developing bolls'
            ],
            'organic_control': [
                'Neem oil spray @ 3-5ml/L every 7 days',
                'Bt (Bacillus thuringiensis) @ 1-2g/L',
                'NPV (Nuclear Polyhedrosis Virus) @ 250 LE/hectare',
                'Release Trichogramma wasps @ 50,000/hectare'
            ],
            'chemical_control': [
                'Emamectin Benzoate 5% SG @ 0.4g/L',
                'Chlorantraniliprole 18.5% SC @ 0.3ml/L',
                'Spinosad 45% SC @ 0.3ml/L'
            ],
            'economic_threshold': '10% plants showing damage or 2 larvae per plant',
            'monitoring': 'Check 10 plants randomly per acre, 3 times per week'
        },
        'Aphids': {
            'scientific_name': 'Aphis gossypii',
            'lifecycle_days': 7,
            'eggs_per_female': 80,
            'favorable_conditions': {
                'temperature': (20, 30),
                'humidity': (50, 70),
                'rainfall': 'low'
            },
            'damage_stages': ['Germination', 'Vegetative', 'Flowering'],
            'symptoms': [
                'Curled, distorted leaves',
                'Sticky honeydew on leaves',
                'Yellowing of leaves',
                'Sooty mold growth on honeydew',
                'Stunted plant growth'
            ],
            'organic_control': [
                'Neem oil @ 5ml/L',
                'Soap solution (Insecticidal soap) @ 5ml/L',
                'Garlic + chili extract spray',
                'Release ladybird beetles and lacewings'
            ],
            'chemical_control': [
                'Imidacloprid 17.8% SL @ 0.3ml/L',
                'Thiamethoxam 25% WG @ 0.3g/L',
                'Acetamiprid 20% SP @ 0.3g/L'
            ],
            'economic_threshold': '5-10 aphids per leaf on 25% of plants',
            'monitoring': 'Check undersides of leaves daily during warm weather'
        },
        'Whitefly': {
            'scientific_name': 'Bemisia tabaci',
            'lifecycle_days': 21,
            'eggs_per_female': 160,
            'favorable_conditions': {
                'temperature': (27, 32),
                'humidity': (60, 80),
                'rainfall': 'low'
            },
            'damage_stages': ['Vegetative', 'Flowering'],
            'symptoms': [
                'Small white flies on leaf undersides',
                'Yellowing and curling of leaves',
                'Honeydew secretion',
                'Leaf curl virus transmission',
                'Reduced plant vigor'
            ],
            'organic_control': [
                'Yellow sticky traps @ 4-6 per acre',
                'Neem oil @ 5ml/L',
                'Reflective mulches to deter adults',
                'Release Encarsia formosa parasitoids'
            ],
            'chemical_control': [
                'Spiromesifen 240 SC @ 0.75ml/L',
                'Diafenthiuron 50% WP @ 1.5g/L',
                'Pyriproxyfen 10% EC @ 1ml/L'
            ],
            'economic_threshold': '5-8 adults per leaf on 25% of plants',
            'monitoring': 'Use yellow sticky traps and check weekly'
        }
    },
    'Rice': {
        'Brown Plant Hopper': {
            'scientific_name': 'Nilaparvata lugens',
            'lifecycle_days': 25,
            'eggs_per_female': 300,
            'favorable_conditions': {
                'temperature': (25, 31),
                'humidity': (75, 90),
                'rainfall': 'high'
            },
            'damage_stages': ['Vegetative', 'Flowering', 'Grain Filling'],
            'symptoms': [
                'Hopper burn - yellowing and drying of plants',
                'Brown patches in field',
                'Plants wilting from base',
                'Sticky honeydew at plant base',
                'Virus transmission (Grassy stunt, Ragged stunt)'
            ],
            'organic_control': [
                'Neem cake application in water @ 250kg/hectare',
                'Maintain 2-3cm water depth to drown nymphs',
                'Release spiders and mirid bugs',
                'Avoid excessive nitrogen fertilization'
            ],
            'chemical_control': [
                'Imidacloprid 17.8% SL @ 0.25ml/L',
                'Buprofezin 25% SC @ 1ml/L',
                'Pymetrozine 50% WG @ 0.6g/L'
            ],
            'economic_threshold': '10-15 hoppers per hill at vegetative stage',
            'monitoring': 'Sweep net sampling weekly, check plant base'
        },
        'Stem Borer': {
            'scientific_name': 'Scirpophaga incertulas',
            'lifecycle_days': 45,
            'eggs_per_female': 200,
            'favorable_conditions': {
                'temperature': (20, 32),
                'humidity': (70, 85),
                'rainfall': 'moderate to high'
            },
            'damage_stages': ['Vegetative', 'Flowering'],
            'symptoms': [
                'Dead hearts at vegetative stage',
                'White heads at reproductive stage',
                'Entry holes on stems',
                'Frass near holes',
                'Central leaf whorl drying'
            ],
            'organic_control': [
                'Release Trichogramma japonicum @ 50,000/hectare weekly',
                'Apply neem cake @ 250kg/hectare',
                'Remove and destroy egg masses',
                'Clip leaf tips containing eggs'
            ],
            'chemical_control': [
                'Cartap Hydrochloride 50% SP @ 2g/L',
                'Chlorantraniliprole 0.4% GR @ 10kg/hectare',
                'Fipronil 0.3% GR @ 20kg/hectare'
            ],
            'economic_threshold': '5% dead hearts or 2 egg masses per square meter',
            'monitoring': 'Check for egg masses on leaves weekly'
        }
    },
    'Tomato': {
        'Fruit Borer': {
            'scientific_name': 'Helicoverpa armigera',
            'lifecycle_days': 30,
            'eggs_per_female': 500,
            'favorable_conditions': {
                'temperature': (25, 30),
                'humidity': (60, 75),
                'rainfall': 'low to moderate'
            },
            'damage_stages': ['Flowering', 'Fruit Development'],
            'symptoms': [
                'Small entry holes in fruits',
                'Brown frass near holes',
                'Hollowed fruits with larvae inside',
                'Premature fruit drop',
                'Secondary fungal infections in damaged fruits'
            ],
            'organic_control': [
                'Pheromone traps @ 10 per hectare',
                'NPV @ 250 LE/hectare',
                'Bt spray @ 1-2g/L',
                'Neem oil @ 3ml/L',
                'Hand-pick and destroy infested fruits'
            ],
            'chemical_control': [
                'Indoxacarb 15.8% EC @ 0.5ml/L',
                'Spinosad 45% SC @ 0.3ml/L',
                'Emamectin Benzoate 5% SG @ 0.4g/L',
                'Alternate chemicals to prevent resistance'
            ],
            'economic_threshold': '5% fruits with damage or 2 larvae per 10 plants',
            'monitoring': 'Check fruits daily during flowering and fruiting stage'
        },
        'Whitefly': {
            'scientific_name': 'Bemisia tabaci',
            'lifecycle_days': 18,
            'eggs_per_female': 160,
            'favorable_conditions': {
                'temperature': (27, 32),
                'humidity': (55, 75),
                'rainfall': 'low'
            },
            'damage_stages': ['Vegetative', 'Flowering', 'Fruit Development'],
            'symptoms': [
                'Small white flies on leaf undersides',
                'Yellowing and curling of leaves',
                'Sooty mold on honeydew',
                'Leaf curl virus transmission',
                'Reduced fruit quality and yield'
            ],
            'organic_control': [
                'Yellow sticky traps @ 8-10 per acre',
                'Neem oil @ 5ml/L + sticking agent',
                'Reflective silver mulch',
                'Remove and destroy heavily infested leaves',
                'Release Encarsia formosa @ 10,000 per hectare'
            ],
            'chemical_control': [
                'Diafenthiuron 50% WP @ 1.5g/L',
                'Spiromesifen 240 SC @ 0.75ml/L',
                'Buprofezin 25% SC @ 1ml/L',
                'Apply when temperature is below 30°C'
            ],
            'economic_threshold': '5-8 adults per leaf',
            'monitoring': 'Weekly monitoring using yellow sticky traps'
        },
        'Leaf Miner': {
            'scientific_name': 'Liriomyza trifolii',
            'lifecycle_days': 21,
            'eggs_per_female': 300,
            'favorable_conditions': {
                'temperature': (22, 30),
                'humidity': (50, 70),
                'rainfall': 'low'
            },
            'damage_stages': ['Vegetative', 'Flowering'],
            'symptoms': [
                'Serpentine white mines on leaves',
                'Stippling marks where adults feed',
                'Yellowing of affected leaves',
                'Reduced photosynthesis',
                'Premature leaf drop in severe cases'
            ],
            'organic_control': [
                'Remove and destroy affected leaves',
                'Neem oil @ 5ml/L',
                'Yellow sticky traps for adult flies',
                'Release parasitoid wasps Diglyphus isaea',
                'Apply neem cake to soil @ 250kg/hectare'
            ],
            'chemical_control': [
                'Abamectin 1.9% EC @ 0.5ml/L',
                'Cyromazine 75% WP @ 0.5g/L',
                'Spinosad 45% SC @ 0.3ml/L'
            ],
            'economic_threshold': '2-3 mines per leaf on 50% of plants',
            'monitoring': 'Check new leaves for mines twice weekly'
        }
    },
    'Potato': {
        'Tuber Moth': {
            'scientific_name': 'Phthorimaea operculella',
            'lifecycle_days': 28,
            'eggs_per_female': 200,
            'favorable_conditions': {
                'temperature': (20, 28),
                'humidity': (40, 60),
                'rainfall': 'low'
            },
            'damage_stages': ['Vegetative', 'Maturity', 'Storage'],
            'symptoms': [
                'Mines in leaves (early stage)',
                'Tunnels in tubers',
                'Frass-filled galleries in tubers',
                'Entry holes in stored potatoes',
                'Secondary rot in damaged tubers'
            ],
            'organic_control': [
                'Deep planting (15cm depth)',
                'Earth-up plants to cover tubers',
                'Immediate harvest after crop maturity',
                'Store potatoes in dark, cool place',
                'Pheromone traps @ 20 per hectare',
                'Sand/ash covering for storage'
            ],
            'chemical_control': [
                'Cartap Hydrochloride 50% SP @ 2g/L',
                'Chlorantraniliprole 18.5% SC @ 0.3ml/L',
                'Spray foliage before tuber formation'
            ],
            'economic_threshold': '2-3 moths per pheromone trap per day',
            'monitoring': 'Use pheromone traps weekly during growing season'
        },
        'Aphids': {
            'scientific_name': 'Myzus persicae',
            'lifecycle_days': 8,
            'eggs_per_female': 80,
            'favorable_conditions': {
                'temperature': (18, 28),
                'humidity': (50, 70),
                'rainfall': 'low to moderate'
            },
            'damage_stages': ['Vegetative', 'Flowering'],
            'symptoms': [
                'Curled, puckered leaves',
                'Sticky honeydew on plants',
                'Yellowing of leaves',
                'Virus transmission (Leaf roll, PVY)',
                'Stunted growth'
            ],
            'organic_control': [
                'Neem oil @ 5ml/L',
                'Soap water spray @ 10ml/L',
                'Mulching with reflective material',
                'Release ladybird beetles',
                'Remove alternate weed hosts'
            ],
            'chemical_control': [
                'Imidacloprid 17.8% SL @ 0.3ml/L',
                'Thiamethoxam 25% WG @ 0.3g/L',
                'Acetamiprid 20% SP @ 0.3g/L',
                'Apply as soon as first aphids appear'
            ],
            'economic_threshold': '10 aphids per leaf on 20% of plants',
            'monitoring': 'Check undersides of leaves twice weekly'
        }
    },
    'Sugarcane': {
        'Early Shoot Borer': {
            'scientific_name': 'Chilo infuscatellus',
            'lifecycle_days': 40,
            'eggs_per_female': 300,
            'favorable_conditions': {
                'temperature': (25, 32),
                'humidity': (70, 85),
                'rainfall': 'moderate'
            },
            'damage_stages': ['Germination', 'Tillering'],
            'symptoms': [
                'Dead hearts in young shoots',
                'Wilting of central leaves',
                'Pin holes on leaves',
                'Reduced tillering',
                'Stunted plant growth'
            ],
            'organic_control': [
                'Remove and destroy egg masses',
                'Release Trichogramma @ 50,000/hectare',
                'Trash mulching',
                'Avoid ratooning of heavily infested crops'
            ],
            'chemical_control': [
                'Chlorantraniliprole 0.4% GR @ 15kg/hectare',
                'Fipronil 0.3% GR @ 25kg/hectare',
                'Carbofuran 3% CG @ 33kg/hectare (at planting)'
            ],
            'economic_threshold': '5% dead hearts',
            'monitoring': 'Check for dead hearts weekly during first 3 months'
        }
    },
    'Soybean': {
        'Pod Borer': {
            'scientific_name': 'Helicoverpa armigera',
            'lifecycle_days': 30,
            'eggs_per_female': 500,
            'favorable_conditions': {
                'temperature': (25, 32),
                'humidity': (60, 80),
                'rainfall': 'moderate'
            },
            'damage_stages': ['Flowering', 'Pod Development'],
            'symptoms': [
                'Holes in pods',
                'Damaged seeds inside pods',
                'Frass on pods',
                'Premature pod drop',
                'Reduced seed quality'
            ],
            'organic_control': [
                'NPV @ 250 LE/hectare',
                'Bt spray @ 2g/L',
                'Neem oil @ 3ml/L',
                'Pheromone traps @ 10 per hectare',
                'Hand-pick larvae early morning'
            ],
            'chemical_control': [
                'Quinalphos 25% EC @ 2ml/L',
                'Indoxacarb 15.8% EC @ 0.5ml/L',
                'Emamectin Benzoate 5% SG @ 0.4g/L'
            ],
            'economic_threshold': '2-3 larvae per meter row or 10% pod damage',
            'monitoring': 'Check pods daily during pod-filling stage'
        }
    }
}

# Disease severity assessment based on confidence
def get_disease_severity(confidence, disease_name):
    """Get disease severity level based on confidence and disease type"""
    if 'healthy' in disease_name.lower():
        return {
            'level': 'Healthy',
            'color': '#44AA44',
            'icon': '✅',
            'action': 'Maintenance Only'
        }
    elif confidence > 85:
        return {
            'level': 'Severe',
            'color': '#FF4444',
            'icon': '🚨',
            'action': 'Immediate Treatment Required'
        }
    elif confidence > 70:
        return {
            'level': 'Moderate',
            'color': '#FFAA00',
            'icon': '⚠️',
            'action': 'Treatment Recommended'
        }
    else:
        return {
            'level': 'Mild/Uncertain',
            'color': '#FFA726',
            'icon': '⚡',
            'action': 'Monitor Closely'
        }
