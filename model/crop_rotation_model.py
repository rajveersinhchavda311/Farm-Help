import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

print("="*70)
print("ENHANCED AGRICULTURAL CROP ROTATION EXPERT SYSTEM")
print("="*70)

def create_advanced_crop_rotation_rules():
    """
    Create comprehensive crop rotation rules based on advanced agricultural science
    """
    
    # Enhanced rotation rules with multiple options and rationale
    rotation_rules = {
        # Cereal crops - rotate to legumes for nitrogen fixation
        'Wheat': {
            'primary': ['Soybean', 'Pea', 'Lentil', 'Chickpea'],
            'secondary': ['Mustard', 'Canola', 'Sunflower'],
            'rationale': 'After cereals, legumes restore soil nitrogen'
        },
        'Rice': {
            'primary': ['Soybean', 'Lentil', 'Mung Bean'],
            'secondary': ['Wheat', 'Mustard', 'Potato'],
            'rationale': 'Legumes after rice improve soil structure and fertility'
        },
        'Maize': {
            'primary': ['Soybean', 'Pea', 'Bean'],
            'secondary': ['Wheat', 'Sunflower', 'Sorghum'],
            'rationale': 'Nitrogen-fixing crops balance nutrient depletion from corn'
        },
        'Barley': {
            'primary': ['Soybean', 'Pea', 'Lentil'],
            'secondary': ['Canola', 'Flax', 'Mustard'],
            'rationale': 'Legumes restore nitrogen after cereal harvest'
        },
        
        # Root/Tuber crops - rotate to cereals or leafy crops
        'Potato': {
            'primary': ['Wheat', 'Barley', 'Oats'],
            'secondary': ['Maize', 'Cabbage', 'Onion'],
            'rationale': 'Cereals break disease cycles and improve soil structure'
        },
        'Sweet Potato': {
            'primary': ['Wheat', 'Rice', 'Maize'],
            'secondary': ['Soybean', 'Pea'],
            'rationale': 'Grains help break root disease cycles'
        },
        
        # Solanaceae family - avoid same family, prefer cereals
        'Tomato': {
            'primary': ['Wheat', 'Rice', 'Maize'],
            'secondary': ['Lettuce', 'Spinach', 'Carrot'],
            'rationale': 'Different families break solanaceae disease cycles'
        },
        'Eggplant': {
            'primary': ['Wheat', 'Barley', 'Oats'],
            'secondary': ['Cabbage', 'Broccoli', 'Pea'],
            'rationale': 'Non-solanaceae crops prevent disease buildup'
        },
        
        # Legumes - rotate to heavy feeders (cereals/brassicas)
        'Soybean': {
            'primary': ['Wheat', 'Rice', 'Maize'],
            'secondary': ['Potato', 'Cabbage', 'Sunflower'],
            'rationale': 'Heavy feeders utilize nitrogen fixed by legumes'
        },
        'Pea': {
            'primary': ['Wheat', 'Rice', 'Maize'],
            'secondary': ['Potato', 'Tomato', 'Cabbage'],
            'rationale': 'Cereals and heavy feeders benefit from nitrogen fixation'
        },
        'Lentil': {
            'primary': ['Wheat', 'Barley', 'Maize'],
            'secondary': ['Sunflower', 'Safflower', 'Mustard'],
            'rationale': 'Nitrogen-demanding crops follow nitrogen-fixing legumes'
        },
        'Chickpea': {
            'primary': ['Wheat', 'Barley', 'Sorghum'],
            'secondary': ['Mustard', 'Safflower', 'Cotton'],
            'rationale': 'Cereals utilize residual nitrogen from chickpea'
        },
        
        # Brassicas/Oil seeds
        'Mustard': {
            'primary': ['Wheat', 'Rice', 'Potato'],
            'secondary': ['Pea', 'Lentil', 'Chickpea'],
            'rationale': 'Mustard improves soil health; rotate to diverse families'
        },
        'Canola': {
            'primary': ['Wheat', 'Barley', 'Pea'],
            'secondary': ['Lentil', 'Flax', 'Oats'],
            'rationale': 'Alternate with cereals and legumes for balanced nutrition'
        },
        
        # Cash crops
        'Cotton': {
            'primary': ['Wheat', 'Sorghum', 'Chickpea'],
            'secondary': ['Mustard', 'Safflower', 'Pearl Millet'],
            'rationale': 'Rotate heavy feeder cotton with soil-improving crops'
        },
        'Sugarcane': {
            'primary': ['Soybean', 'Wheat', 'Potato'],
            'secondary': ['Mustard', 'Pea', 'Onion'],
            'rationale': 'Long-duration sugarcane requires soil restoration crops'
        }
    }
    
    # Advanced disease management with severity levels
    disease_restrictions = {
        'Late_blight': {
            'high_risk': ['Potato', 'Tomato', 'Eggplant'],
            'medium_risk': ['Pepper', 'Sweet Potato'],
            'safe': ['Wheat', 'Rice', 'Maize', 'Soybean', 'Mustard'],
            'beneficial': ['Wheat', 'Barley', 'Mustard'],
            'management': 'Avoid solanaceae family for 2-3 years'
        },
        'Early_blight': {
            'high_risk': ['Potato', 'Tomato', 'Eggplant'],
            'medium_risk': ['Pepper'],
            'safe': ['Wheat', 'Rice', 'Maize', 'Soybean', 'Pea'],
            'beneficial': ['Maize', 'Soybean', 'Mustard'],
            'management': 'Rotate out of solanaceae, improve air circulation'
        },
        'Black_rot': {
            'high_risk': ['Tomato', 'Pepper', 'Eggplant'],
            'medium_risk': ['Potato'],
            'safe': ['Wheat', 'Rice', 'Maize', 'Soybean', 'Pea', 'Mustard'],
            'beneficial': ['Wheat', 'Maize', 'Soybean'],
            'management': 'Non-host crops reduce pathogen survival'
        },
        'Common_rust': {
            'high_risk': ['Maize', 'Wheat', 'Barley'],
            'medium_risk': ['Oats', 'Rye'],
            'safe': ['Soybean', 'Pea', 'Potato', 'Tomato', 'Mustard'],
            'beneficial': ['Soybean', 'Pea', 'Lentil'],
            'management': 'Rotate to non-grass family crops'
        },
        'Powdery_mildew': {
            'high_risk': ['Wheat', 'Barley', 'Pea'],
            'medium_risk': ['Oats', 'Lentil'],
            'safe': ['Maize', 'Rice', 'Potato', 'Tomato', 'Soybean'],
            'beneficial': ['Maize', 'Rice', 'Mustard'],
            'management': 'Improve air circulation, avoid susceptible varieties'
        },
        'Fusarium_wilt': {
            'high_risk': ['Tomato', 'Potato', 'Eggplant', 'Cotton'],
            'medium_risk': ['Pepper', 'Chickpea'],
            'safe': ['Wheat', 'Rice', 'Maize', 'Mustard', 'Onion'],
            'beneficial': ['Wheat', 'Rice', 'Mustard', 'Maize'],
            'management': 'Long rotation with non-host crops'
        },
        'healthy': {
            'high_risk': [],
            'medium_risk': [],
            'safe': [],
            'beneficial': [],
            'management': 'Continue preventive rotation practices'
        }
    }
    
    # Comprehensive soil-based recommendations
    soil_recommendations = {
        'very_low_K': {  # K < 15
            'preferred': ['Soybean', 'Pea', 'Lentil', 'Chickpea', 'Bean'],
            'bonus_score': 0.8,
            'reason': 'Legumes fix nitrogen and improve K availability'
        },
        'low_K': {  # K 15-25
            'preferred': ['Soybean', 'Pea', 'Lentil', 'Alfalfa'],
            'bonus_score': 0.5,
            'reason': 'Nitrogen-fixing crops help with nutrient balance'
        },
        'very_acidic': {  # pH < 5.5
            'preferred': ['Potato', 'Sweet Potato', 'Blueberry'],
            'bonus_score': 0.6,
            'reason': 'Acid-tolerant crops thrive in low pH soils'
        },
        'acidic': {  # pH 5.5-6.0
            'preferred': ['Potato', 'Lentil', 'Oats', 'Rye'],
            'bonus_score': 0.4,
            'reason': 'Moderately acid-tolerant crops'
        },
        'alkaline': {  # pH > 7.5
            'preferred': ['Wheat', 'Barley', 'Mustard', 'Spinach'],
            'bonus_score': 0.4,
            'reason': 'Alkaline-tolerant crops perform well'
        },
        'very_alkaline': {  # pH > 8.0
            'preferred': ['Barley', 'Sugar Beet', 'Asparagus'],
            'bonus_score': 0.6,
            'reason': 'High alkaline tolerance needed'
        },
        'very_dry': {  # rainfall < 100
            'preferred': ['Sorghum', 'Pearl Millet', 'Chickpea', 'Mustard'],
            'bonus_score': 0.7,
            'reason': 'Drought-resistant crops for arid conditions'
        },
        'dry': {  # rainfall 100-200
            'preferred': ['Wheat', 'Barley', 'Chickpea', 'Mustard', 'Safflower'],
            'bonus_score': 0.4,
            'reason': 'Moderate drought tolerance'
        },
        'wet': {  # rainfall 300-500
            'preferred': ['Rice', 'Soybean', 'Maize', 'Sugarcane'],
            'bonus_score': 0.4,
            'reason': 'Water-loving crops for high rainfall'
        },
        'very_wet': {  # rainfall > 500
            'preferred': ['Rice', 'Sugarcane', 'Taro'],
            'bonus_score': 0.6,
            'reason': 'Flood-tolerant crops for excessive moisture'
        },
        'high_temp': {  # temperature > 30
            'preferred': ['Sorghum', 'Pearl Millet', 'Cotton', 'Okra'],
            'bonus_score': 0.5,
            'reason': 'Heat-tolerant crops for hot climates'
        },
        'low_temp': {  # temperature < 15
            'preferred': ['Wheat', 'Barley', 'Oats', 'Pea', 'Cabbage'],
            'bonus_score': 0.5,
            'reason': 'Cool-season crops for low temperatures'
        }
    }
    
    return rotation_rules, disease_restrictions, soil_recommendations

def advanced_soil_analysis(soil_data):
    """
    Comprehensive soil condition analysis with multiple parameters
    """
    conditions = []
    analysis = {}
    
    # Potassium analysis
    K = soil_data.get('K', 25)
    if K < 15:
        conditions.append('very_low_K')
        analysis['K_status'] = 'Very Low - Critical'
    elif K < 25:
        conditions.append('low_K')
        analysis['K_status'] = 'Low - Needs Improvement'
    elif K > 45:
        analysis['K_status'] = 'High - Adequate'
    else:
        analysis['K_status'] = 'Normal - Good'
    
    # pH analysis
    ph = soil_data.get('ph', 6.5)
    if ph < 5.5:
        conditions.append('very_acidic')
        analysis['pH_status'] = 'Very Acidic - Lime Needed'
    elif ph < 6.0:
        conditions.append('acidic')
        analysis['pH_status'] = 'Acidic - Consider Liming'
    elif ph > 8.0:
        conditions.append('very_alkaline')
        analysis['pH_status'] = 'Very Alkaline - Sulfur Needed'
    elif ph > 7.5:
        conditions.append('alkaline')
        analysis['pH_status'] = 'Alkaline - Monitor'
    else:
        analysis['pH_status'] = 'Optimal - Good'
    
    # Rainfall analysis
    rainfall = soil_data.get('rainfall', 200)
    if rainfall < 100:
        conditions.append('very_dry')
        analysis['moisture_status'] = 'Very Dry - Irrigation Critical'
    elif rainfall < 200:
        conditions.append('dry')
        analysis['moisture_status'] = 'Dry - Supplemental Irrigation'
    elif rainfall > 500:
        conditions.append('very_wet')
        analysis['moisture_status'] = 'Very Wet - Drainage Needed'
    elif rainfall > 300:
        conditions.append('wet')
        analysis['moisture_status'] = 'Wet - Good for Water Crops'
    else:
        analysis['moisture_status'] = 'Optimal - Well Balanced'
    
    # Temperature analysis
    temp = soil_data.get('temperature', 25)
    if temp > 30:
        conditions.append('high_temp')
        analysis['temp_status'] = 'High - Heat Stress Risk'
    elif temp < 15:
        conditions.append('low_temp')
        analysis['temp_status'] = 'Low - Cool Season'
    else:
        analysis['temp_status'] = 'Optimal - Good Growing'
    
    # Humidity analysis
    humidity = soil_data.get('humidity', 60)
    if humidity > 85:
        analysis['humidity_status'] = 'Very High - Disease Risk'
    elif humidity > 70:
        analysis['humidity_status'] = 'High - Monitor Disease'
    elif humidity < 40:
        analysis['humidity_status'] = 'Low - Water Stress Risk'
    else:
        analysis['humidity_status'] = 'Optimal - Good'
    
    return conditions, analysis

def advanced_crop_scoring(possible_crops, soil_conditions, detected_disease, soil_recommendations, disease_restrictions, soil_analysis):
    """
    Advanced scoring system with multiple factors and weights
    """
    scores = {}
    explanations = {}
    
    for crop in possible_crops:
        score = 1.0
        explanation = []
        
        # Base rotation score
        explanation.append(f"Base rotation score: 1.0")
        
        # Soil condition bonuses
        soil_bonus = 0
        for condition in soil_conditions:
            if condition in soil_recommendations:
                if crop in soil_recommendations[condition]['preferred']:
                    bonus = soil_recommendations[condition]['bonus_score']
                    soil_bonus += bonus
                    explanation.append(f"+{bonus} for {condition} ({soil_recommendations[condition]['reason']})")
        
        score += soil_bonus
        
        # Disease penalties and bonuses
        disease_info = disease_restrictions.get(detected_disease, {})
        if crop in disease_info.get('high_risk', []):
            penalty = -0.8
            score += penalty
            explanation.append(f"{penalty} - High disease risk for {detected_disease}")
        elif crop in disease_info.get('medium_risk', []):
            penalty = -0.4
            score += penalty
            explanation.append(f"{penalty} - Medium disease risk for {detected_disease}")
        elif crop in disease_info.get('beneficial', []):
            bonus = 0.5
            score += bonus
            explanation.append(f"+{bonus} - Helps manage {detected_disease}")
        elif crop in disease_info.get('safe', []):
            bonus = 0.2
            score += bonus
            explanation.append(f"+{bonus} - Safe from {detected_disease}")
        
        # Ensure minimum score
        score = max(score, 0.1)
        
        scores[crop] = score
        explanations[crop] = explanation
    
    return scores, explanations

def predict_crop_rotation_advanced(soil_data, previous_crop, detected_disease='healthy'):
    """
    Advanced crop rotation prediction with comprehensive analysis
    """
    # Get advanced rules
    rotation_rules, disease_restrictions, soil_recommendations = create_advanced_crop_rotation_rules()
    
    # Get base rotation options
    crop_info = rotation_rules.get(previous_crop, {
        'primary': ['Wheat', 'Rice', 'Soybean'],
        'secondary': ['Maize', 'Pea'],
        'rationale': 'General rotation principles'
    })
    
    possible_crops = crop_info['primary'] + crop_info['secondary']
    
    # Advanced soil analysis
    soil_conditions, soil_analysis = advanced_soil_analysis(soil_data)
    
    # Advanced scoring
    scores, explanations = advanced_crop_scoring(
        possible_crops, soil_conditions, detected_disease, 
        soil_recommendations, disease_restrictions, soil_analysis
    )
    
    # Add additional recommended crops based on conditions
    disease_info = disease_restrictions.get(detected_disease, {})
    for beneficial_crop in disease_info.get('beneficial', []):
        if beneficial_crop not in scores:
            scores[beneficial_crop] = 1.2
            explanations[beneficial_crop] = [f"+1.2 - Specifically beneficial for {detected_disease}"]
    
    # Add soil-specific recommendations
    for condition in soil_conditions:
        if condition in soil_recommendations:
            for preferred_crop in soil_recommendations[condition]['preferred']:
                if preferred_crop not in scores:
                    bonus = soil_recommendations[condition]['bonus_score']
                    scores[preferred_crop] = 0.8 + bonus
                    explanations[preferred_crop] = [f"+{0.8 + bonus} - {soil_recommendations[condition]['reason']}"]
    
    # Sort and normalize scores
    sorted_recommendations = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Get top recommendation
    top_crop = sorted_recommendations[0][0]
    confidence = sorted_recommendations[0][1]
    
    # Normalize to probabilities
    total_score = sum(scores.values())
    probabilities = {crop: round(score/total_score, 3) for crop, score in sorted_recommendations}
    
    # Create comprehensive result
    result = {
        'recommended_crop': top_crop,
        'confidence': round(confidence, 3),
        'probability': probabilities.get(top_crop, 0),
        'top_5_recommendations': dict(sorted_recommendations[:5]),
        'all_probabilities': probabilities,
        'detailed_analysis': {
            'previous_crop_info': crop_info,
            'soil_analysis': soil_analysis,
            'soil_conditions': soil_conditions,
            'disease_management': disease_restrictions.get(detected_disease, {}).get('management', 'Standard practices'),
            'explanations': {crop: explanations[crop] for crop in dict(sorted_recommendations[:3]).keys()}
        },
        'recommendations': {
            'immediate': f"Plant {top_crop} with {probabilities.get(top_crop, 0)*100:.1f}% suitability",
            'soil_improvement': _get_soil_recommendations(soil_analysis),
            'disease_prevention': disease_restrictions.get(detected_disease, {}).get('management', 'Continue monitoring'),
            'alternative_crops': [crop for crop, _ in sorted_recommendations[1:4]]
        }
    }
    
    return result

def _get_soil_recommendations(soil_analysis):
    """Generate soil improvement recommendations"""
    recommendations = []
    
    if 'Critical' in soil_analysis.get('K_status', ''):
        recommendations.append("Apply potassium-rich fertilizer or compost")
    
    if 'Lime Needed' in soil_analysis.get('pH_status', ''):
        recommendations.append("Apply agricultural lime to raise pH")
    elif 'Sulfur Needed' in soil_analysis.get('pH_status', ''):
        recommendations.append("Apply sulfur to lower pH")
    
    if 'Irrigation' in soil_analysis.get('moisture_status', ''):
        recommendations.append("Install or improve irrigation system")
    elif 'Drainage' in soil_analysis.get('moisture_status', ''):
        recommendations.append("Improve field drainage systems")
    
    if 'Disease Risk' in soil_analysis.get('humidity_status', ''):
        recommendations.append("Improve air circulation and consider fungicide schedule")
    
    return recommendations if recommendations else ["Maintain current soil management practices"]

def create_enhanced_dataset(n_samples=50):
    """
    Create enhanced dataset with realistic scenarios (reduced for demo)
    """
    crops = ['Wheat', 'Rice', 'Maize', 'Potato', 'Tomato', 'Soybean', 'Pea', 'Lentil', 'Cotton', 'Mustard']
    diseases = ['healthy', 'Late_blight', 'Early_blight', 'Black_rot', 'Common_rust', 'Fusarium_wilt']
    
    data = []
    
    for i in range(n_samples):
        # Create realistic soil combinations
        if np.random.random() < 0.3:  # 30% challenging conditions
            soil_data = {
                'K': np.random.uniform(8, 20),
                'temperature': np.random.uniform(32, 40),
                'humidity': np.random.uniform(25, 45),
                'ph': np.random.choice([np.random.uniform(4.5, 5.5), np.random.uniform(8.0, 8.8)]),
                'rainfall': np.random.choice([np.random.uniform(50, 120), np.random.uniform(400, 600)])
            }
        else:  # 70% normal conditions
            soil_data = {
                'K': np.random.uniform(20, 45),
                'temperature': np.random.uniform(18, 32),
                'humidity': np.random.uniform(45, 75),
                'ph': np.random.uniform(6.0, 7.5),
                'rainfall': np.random.uniform(150, 350)
            }
        
        prev_crop = np.random.choice(crops)
        detected_disease = np.random.choice(diseases, p=[0.4, 0.15, 0.15, 0.1, 0.1, 0.1])
        
        # Get recommendation
        result = predict_crop_rotation_advanced(soil_data, prev_crop, detected_disease)
        
        data.append({
            'scenario_id': i + 1,
            'K': round(soil_data['K'], 1),
            'temperature': round(soil_data['temperature'], 1),
            'humidity': round(soil_data['humidity'], 1),
            'ph': round(soil_data['ph'], 2),
            'rainfall': round(soil_data['rainfall'], 1),
            'previous_crop': prev_crop,
            'detected_disease': detected_disease,
            'recommended_crop': result['recommended_crop'],
            'confidence': result['confidence'],
            'probability': result['probability'],
            'soil_k_status': result['detailed_analysis']['soil_analysis'].get('K_status', 'Unknown'),
            'soil_ph_status': result['detailed_analysis']['soil_analysis'].get('pH_status', 'Unknown'),
            'soil_moisture_status': result['detailed_analysis']['soil_analysis'].get('moisture_status', 'Unknown'),
            'alternative_1': result['recommendations']['alternative_crops'][0] if len(result['recommendations']['alternative_crops']) > 0 else '',
            'alternative_2': result['recommendations']['alternative_crops'][1] if len(result['recommendations']['alternative_crops']) > 1 else '',
            'alternative_3': result['recommendations']['alternative_crops'][2] if len(result['recommendations']['alternative_crops']) > 2 else '',
        })
    
    df = pd.DataFrame(data)
    return df

def save_enhanced_results():
    """
    Save comprehensive results and analytics - FIXED VERSION
    """
    print("\n💾 SAVING ENHANCED RESULTS...")
    
    # Create datasets
    df = create_enhanced_dataset(50)  # Reduced for demo
    
    # Save main dataset
    os.makedirs('enhanced_data', exist_ok=True)
    df.to_csv('enhanced_data/crop_rotation_enhanced.csv', index=False)
    
    # Create transition matrix with string keys - FIXED
    transition_matrix = {}
    for (prev_crop, rec_crop), count in df.groupby(['previous_crop', 'recommended_crop']).size().items():
        key = f"{prev_crop} → {rec_crop}"
        transition_matrix[key] = int(count)
    
    # Create summary analytics - FIXED
    analytics = {
        'dataset_summary': {
            'total_scenarios': int(len(df)),
            'unique_previous_crops': int(df['previous_crop'].nunique()),
            'unique_recommended_crops': int(df['recommended_crop'].nunique()),
            'average_confidence': float(round(df['confidence'].mean(), 3)),
            'high_confidence_scenarios': int(len(df[df['confidence'] > 1.5])),
            'disease_scenarios': int(len(df[df['detected_disease'] != 'healthy']))
        },
        'crop_transition_matrix': transition_matrix,
        'soil_condition_analysis': {
            'low_k_scenarios': int(len(df[df['soil_k_status'].str.contains('Low', na=False)])),
            'acidic_soil_scenarios': int(len(df[df['soil_ph_status'].str.contains('Acidic', na=False)])),
            'drought_scenarios': int(len(df[df['soil_moisture_status'].str.contains('Dry', na=False)])),
        },
        'disease_management_stats': {str(k): int(v) for k, v in df['detected_disease'].value_counts().to_dict().items()},
        'recommendation_frequency': {str(k): int(v) for k, v in df['recommended_crop'].value_counts().to_dict().items()}
    }
    
    # Save analytics
    with open('enhanced_data/rotation_analytics.json', 'w') as f:
        json.dump(analytics, f, indent=2)
    
    print(f"✓ Enhanced dataset saved: {len(df)} scenarios")
    print(f"✓ Analytics saved with {len(analytics)} metrics")
    print(f"✓ Files location: enhanced_data/")
    
    # Print summary
    print(f"\n📈 DATASET SUMMARY:")
    for key, value in analytics['dataset_summary'].items():
        print(f"   • {key.replace('_', ' ').title()}: {value}")
    
    # Print top transitions
    print(f"\n🔄 TOP CROP TRANSITIONS:")
    sorted_transitions = sorted(transition_matrix.items(), key=lambda x: x[1], reverse=True)
    for transition, count in sorted_transitions[:5]:
        print(f"   • {transition}: {count} times")
    
    return df, analytics

def test_quick_scenarios():
    """
    Quick test scenarios
    """
    print("\n🧪 TESTING CROP ROTATION SCENARIOS")
    print("="*50)
    
    test_cases = [
        {
            'name': 'Ideal After Wheat',
            'soil': {'K': 35, 'temperature': 22, 'humidity': 60, 'ph': 6.8, 'rainfall': 180},
            'prev_crop': 'Wheat',
            'disease': 'healthy'
        },
        {
            'name': 'Disease Crisis',
            'soil': {'K': 25, 'temperature': 28, 'humidity': 85, 'ph': 6.2, 'rainfall': 280},
            'prev_crop': 'Potato',
            'disease': 'Late_blight'
        },
        {
            'name': 'Poor Soil',
            'soil': {'K': 12, 'temperature': 35, 'humidity': 35, 'ph': 5.2, 'rainfall': 80},
            'prev_crop': 'Cotton',
            'disease': 'healthy'
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n🌱 Test {i}: {case['name']}")
        result = predict_crop_rotation_advanced(case['soil'], case['prev_crop'], case['disease'])
        print(f"   Recommended: {result['recommended_crop']} ({result['probability']*100:.1f}%)")
        print(f"   Alternatives: {', '.join(result['recommendations']['alternative_crops'][:2])}")

def main():
    """
    Main function - FINAL VERSION
    """
    print("🚀 FINAL AGRICULTURAL CROP ROTATION EXPERT SYSTEM")
    print()
    
    try:
        # Quick test
        test_quick_scenarios()
        
        # Save results
        df, analytics = save_enhanced_results()
        
        print("\n" + "="*70)
        print("🎉 SUCCESS! FINAL SYSTEM READY!")
        print("="*70)
        print("✅ Advanced agricultural logic")
        print("✅ Multi-factor scoring system")
        print("✅ Comprehensive soil analysis")
        print("✅ Disease management integration")
        print("✅ JSON export working correctly")
        print()
        print("📊 USAGE:")
        print("result = predict_crop_rotation_advanced(soil_data, 'Wheat', 'healthy')")
        print("print(f\"Plant {result['recommended_crop']} next!\")")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
