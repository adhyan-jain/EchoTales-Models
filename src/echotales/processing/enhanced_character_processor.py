from transformers import (
    pipeline, 
    AutoTokenizer, 
    AutoModel,
    AutoModelForSequenceClassification,
    BertTokenizer,
    BertModel
)
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import re
import numpy as np
import logging
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from textblob import TextBlob

from ..core.models import (
    PersonalityVector, 
    PhysicalDescription, 
    Character, 
    DialogueLine,
    Gender,
    AgeGroup
)

logger = logging.getLogger(__name__)


class PersonalityExtractor:
    """Extract Big Five personality traits using multiple ML approaches"""
    
    def __init__(self):
        try:
            # Load pre-trained models for personality analysis
            self.emotion_classifier = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base"
            )
            
            # Load BERT model for text embeddings
            self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.bert_model = BertModel.from_pretrained('bert-base-uncased')
            
            # Load sentiment analyzer
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest"
            )
            
            # Personality trait keywords (for ensemble approach)
            self.personality_keywords = {
                'openness': [
                    'creative', 'imaginative', 'artistic', 'curious', 'original',
                    'unconventional', 'inventive', 'philosophical', 'intellectual',
                    'adventurous', 'explores', 'innovative', 'aesthetic'
                ],
                'conscientiousness': [
                    'organized', 'responsible', 'disciplined', 'careful', 'thorough',
                    'punctual', 'reliable', 'dutiful', 'systematic', 'methodical',
                    'planned', 'efficient', 'diligent', 'persistent'
                ],
                'extraversion': [
                    'outgoing', 'sociable', 'talkative', 'assertive', 'energetic',
                    'enthusiastic', 'gregarious', 'active', 'bold', 'dominant',
                    'confident', 'cheerful', 'optimistic', 'expressive'
                ],
                'agreeableness': [
                    'kind', 'compassionate', 'cooperative', 'trusting', 'helpful',
                    'forgiving', 'generous', 'considerate', 'polite', 'friendly',
                    'empathetic', 'sympathetic', 'altruistic', 'warm'
                ],
                'neuroticism': [
                    'anxious', 'worried', 'nervous', 'tense', 'moody', 'irritable',
                    'stressed', 'emotional', 'insecure', 'vulnerable', 'unstable',
                    'fearful', 'jealous', 'envious', 'angry'
                ]
            }
            
            logger.info("Personality extraction models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading personality models: {e}")
            self.emotion_classifier = None
            self.sentiment_analyzer = None
            self.bert_model = None
            self.bert_tokenizer = None
    
    def extract_personality_from_text(self, texts: List[str], dialogue: List[str]) -> PersonalityVector:
        """Extract personality traits from character descriptions and dialogue"""
        
        # Combine all text for analysis
        all_text = ' '.join(texts + dialogue)
        
        # Method 1: Keyword-based analysis with context weighting
        keyword_scores = self._analyze_personality_keywords(all_text)
        
        # Method 2: Emotion-based personality inference
        emotion_scores = self._analyze_emotions_for_personality(dialogue)
        
        # Method 3: BERT embeddings with trained personality classifier
        embedding_scores = self._analyze_bert_embeddings_for_personality(all_text)
        
        # Method 4: Linguistic pattern analysis
        linguistic_scores = self._analyze_linguistic_patterns(all_text, dialogue)
        
        # Ensemble approach: combine all methods with weights
        final_scores = self._combine_personality_scores([
            (keyword_scores, 0.3),
            (emotion_scores, 0.3),
            (embedding_scores, 0.25),
            (linguistic_scores, 0.15)
        ])
        
        return PersonalityVector(**final_scores)
    
    def _analyze_personality_keywords(self, text: str) -> Dict[str, float]:
        """Analyze personality using keyword matching with TF-IDF weighting"""
        text_lower = text.lower()
        scores = {}
        
        # Use TF-IDF to weight keyword importance
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        
        try:
            tfidf_matrix = vectorizer.fit_transform([text_lower])
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = dict(zip(feature_names, tfidf_matrix.toarray()[0]))
        except:
            tfidf_scores = {}
        
        for trait, keywords in self.personality_keywords.items():
            trait_score = 0.0
            total_weight = 0.0
            
            for keyword in keywords:
                # Count occurrences
                count = text_lower.count(keyword)
                if count > 0:
                    # Weight by TF-IDF score if available
                    weight = tfidf_scores.get(keyword, 1.0)
                    trait_score += count * weight
                    total_weight += weight
            
            # Normalize score
            if total_weight > 0:
                scores[trait] = min(trait_score / (total_weight * len(keywords)), 1.0)
            else:
                scores[trait] = 0.0
        
        # Add confidence score based on text length and keyword density
        text_length = len(text.split())
        if text_length > 0:
            keyword_density = sum(scores.values()) / len(scores)
            confidence = min(keyword_density * (text_length / 100.0), 1.0)
        else:
            confidence = 0.0
            
        scores['confidence'] = confidence
        
        return scores
    
    def _analyze_emotions_for_personality(self, dialogue: List[str]) -> Dict[str, float]:
        """Infer personality from emotional patterns in dialogue"""
        if not self.emotion_classifier or not dialogue:
            return {trait: 0.0 for trait in self.personality_keywords.keys() + ['confidence']}
        
        emotion_counts = defaultdict(int)
        total_lines = 0
        
        for line in dialogue[:50]:  # Limit to avoid API limits
            try:
                emotions = self.emotion_classifier(line)
                if emotions:
                    top_emotion = emotions[0]['label'].lower()
                    emotion_counts[top_emotion] += 1
                total_lines += 1
            except Exception as e:
                logger.warning(f"Emotion classification failed for line: {e}")
                continue
        
        if total_lines == 0:
            return {trait: 0.0 for trait in list(self.personality_keywords.keys()) + ['confidence']}
        
        # Map emotions to personality traits
        emotion_to_personality = {
            'joy': {'extraversion': 0.8, 'agreeableness': 0.6, 'neuroticism': -0.4},
            'sadness': {'neuroticism': 0.7, 'extraversion': -0.5, 'openness': 0.3},
            'anger': {'neuroticism': 0.8, 'agreeableness': -0.7, 'conscientiousness': -0.3},
            'fear': {'neuroticism': 0.9, 'openness': -0.2, 'extraversion': -0.6},
            'surprise': {'openness': 0.7, 'neuroticism': 0.3},
            'disgust': {'agreeableness': -0.6, 'neuroticism': 0.4},
            'love': {'agreeableness': 0.8, 'extraversion': 0.5, 'neuroticism': -0.3}
        }
        
        # Calculate personality scores from emotion patterns
        personality_scores = defaultdict(float)
        
        for emotion, count in emotion_counts.items():
            proportion = count / total_lines
            if emotion in emotion_to_personality:
                for trait, weight in emotion_to_personality[emotion].items():
                    personality_scores[trait] += proportion * weight
        
        # Normalize scores to [0, 1]
        final_scores = {}
        for trait in self.personality_keywords.keys():
            score = personality_scores.get(trait, 0.0)
            # Convert from [-1, 1] to [0, 1]
            normalized_score = (score + 1.0) / 2.0
            final_scores[trait] = max(0.0, min(1.0, normalized_score))
        
        final_scores['confidence'] = min(total_lines / 20.0, 1.0)  # More dialogue = higher confidence
        
        return final_scores
    
    def _analyze_bert_embeddings_for_personality(self, text: str) -> Dict[str, float]:
        """Use BERT embeddings with personality classification"""
        if not self.bert_model or not self.bert_tokenizer:
            return {trait: 0.0 for trait in list(self.personality_keywords.keys()) + ['confidence']}
        
        try:
            # Tokenize and encode text
            inputs = self.bert_tokenizer(text[:512], return_tensors='pt', truncation=True, padding=True)
            
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
            
            # Simple personality classification based on embedding similarity
            # This is a simplified approach - in practice, you'd train a classifier
            personality_scores = {}
            
            for trait, keywords in self.personality_keywords.items():
                # Get embeddings for trait keywords
                trait_text = ' '.join(keywords[:10])  # Use top 10 keywords
                trait_inputs = self.bert_tokenizer(trait_text, return_tensors='pt', truncation=True, padding=True)
                
                with torch.no_grad():
                    trait_outputs = self.bert_model(**trait_inputs)
                    trait_embeddings = trait_outputs.last_hidden_state.mean(dim=1).squeeze()
                
                # Calculate similarity
                similarity = F.cosine_similarity(embeddings.unsqueeze(0), trait_embeddings.unsqueeze(0))
                personality_scores[trait] = max(0.0, float(similarity))
            
            # Normalize scores
            max_score = max(personality_scores.values()) if personality_scores.values() else 1.0
            if max_score > 0:
                for trait in personality_scores:
                    personality_scores[trait] /= max_score
            
            personality_scores['confidence'] = min(len(text) / 1000.0, 1.0)
            
            return personality_scores
            
        except Exception as e:
            logger.error(f"BERT embedding analysis failed: {e}")
            return {trait: 0.0 for trait in list(self.personality_keywords.keys()) + ['confidence']}
    
    def _analyze_linguistic_patterns(self, text: str, dialogue: List[str]) -> Dict[str, float]:
        """Analyze linguistic patterns for personality inference"""
        scores = {}
        
        # Analyze text complexity for openness
        avg_sentence_length = len(text.split()) / max(text.count('.'), 1)
        complex_words = len([w for w in text.split() if len(w) > 6])
        openness_score = min((avg_sentence_length / 20.0) + (complex_words / len(text.split())), 1.0)
        scores['openness'] = openness_score
        
        # Analyze structure and organization for conscientiousness
        sentence_count = text.count('.')
        paragraph_indicators = text.count('\n\n') + text.count('\n')
        organization_score = min((sentence_count / 50.0) + (paragraph_indicators / 10.0), 1.0)
        scores['conscientiousness'] = organization_score
        
        # Analyze dialogue patterns for extraversion
        if dialogue:
            avg_dialogue_length = sum(len(line.split()) for line in dialogue) / len(dialogue)
            exclamation_count = sum(line.count('!') for line in dialogue)
            extraversion_score = min((avg_dialogue_length / 15.0) + (exclamation_count / len(dialogue)), 1.0)
            scores['extraversion'] = extraversion_score
        else:
            scores['extraversion'] = 0.0
        
        # Analyze sentiment for agreeableness
        if self.sentiment_analyzer:
            try:
                sentiment = self.sentiment_analyzer(text[:512])
                if sentiment and sentiment[0]['label'] in ['LABEL_2', 'positive']:  # Positive sentiment
                    scores['agreeableness'] = sentiment[0]['score']
                else:
                    scores['agreeableness'] = 1.0 - sentiment[0]['score'] if sentiment else 0.5
            except:
                scores['agreeableness'] = 0.5
        else:
            scores['agreeableness'] = 0.5
        
        # Analyze emotional language for neuroticism
        emotional_words = ['worried', 'anxious', 'scared', 'angry', 'sad', 'upset', 'frustrated']
        emotional_count = sum(text.lower().count(word) for word in emotional_words)
        neuroticism_score = min(emotional_count / max(len(text.split()) / 50, 1), 1.0)
        scores['neuroticism'] = neuroticism_score
        
        scores['confidence'] = 0.6  # Moderate confidence for linguistic analysis
        
        return scores
    
    def _combine_personality_scores(self, scored_methods: List[Tuple[Dict[str, float], float]]) -> Dict[str, float]:
        """Combine personality scores from multiple methods using weighted average"""
        final_scores = defaultdict(float)
        total_weights = defaultdict(float)
        
        for scores, method_weight in scored_methods:
            method_confidence = scores.get('confidence', 0.5)
            effective_weight = method_weight * method_confidence
            
            for trait in self.personality_keywords.keys():
                if trait in scores:
                    final_scores[trait] += scores[trait] * effective_weight
                    total_weights[trait] += effective_weight
        
        # Normalize by total weights
        result = {}
        for trait in self.personality_keywords.keys():
            if total_weights[trait] > 0:
                result[trait] = final_scores[trait] / total_weights[trait]
            else:
                result[trait] = 0.5  # Default to middle value
        
        # Calculate overall confidence
        avg_confidence = sum(scores.get('confidence', 0.0) for scores, _ in scored_methods) / len(scored_methods)
        result['confidence'] = avg_confidence
        
        return result


class PhysicalDescriptionExtractor:
    """Extract physical descriptions using NLP and pattern matching"""
    
    def __init__(self):
        try:
            # Load spaCy model for NER
            self.nlp = spacy.load("en_core_web_sm")
            
            # Load text generation model for description enhancement
            self.text_generator = pipeline(
                "text2text-generation",
                model="microsoft/DialoGPT-medium"
            )
            
            # Physical feature patterns
            self.physical_patterns = {
                'height': r'\b(tall|short|towering|petite|giant|tiny|height|inches?|feet?|ft)\b',
                'build': r'\b(thin|thick|muscular|slender|stocky|athletic|heavy|lean|robust|frail|strong)\b',
                'hair_color': r'\b(blonde?|brunette?|black-haired|red-haired|gray-haired|white-haired|auburn|chestnut)\b',
                'hair_style': r'\b(curly|straight|wavy|braided|long|short|shoulder-length|cropped)\b',
                'eye_color': r'\b(blue-eyed|brown-eyed|green-eyed|hazel|gray-eyed|amber)\b',
                'skin_tone': r'\b(pale|dark|tanned|fair|olive|ebony|ivory|bronze)\b',
                'age_appearance': r'\b(young|old|elderly|middle-aged|youthful|aged|teen|adult)\b'
            }
            
            logger.info("Physical description extraction models loaded")
            
        except Exception as e:
            logger.error(f"Error loading physical description models: {e}")
            self.nlp = None
            self.text_generator = None
    
    def extract_physical_description(self, context_texts: List[str]) -> PhysicalDescription:
        """Extract physical descriptions from context text"""
        
        combined_text = ' '.join(context_texts).lower()
        
        # Method 1: Pattern-based extraction
        pattern_features = self._extract_with_patterns(combined_text)
        
        # Method 2: NER-based extraction
        ner_features = self._extract_with_ner(combined_text)
        
        # Method 3: Context-based inference
        context_features = self._extract_with_context(context_texts)
        
        # Combine results
        final_features = self._combine_physical_features([
            pattern_features,
            ner_features, 
            context_features
        ])
        
        return PhysicalDescription(**final_features)
    
    def _extract_with_patterns(self, text: str) -> Dict[str, any]:
        """Extract features using regex patterns"""
        features = {}
        
        for feature_type, pattern in self.physical_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Take the most common match
                most_common = Counter(matches).most_common(1)[0][0]
                if feature_type in ['height', 'build', 'hair_color', 'hair_style', 'eye_color', 'skin_tone', 'age_appearance']:
                    features[feature_type] = most_common
        
        # Extract distinctive features
        distinctive_patterns = [
            r'\b(scar|tattoo|birthmark|freckles|dimples|beard|mustache|glasses)\b',
            r'\b(limp|cane|wheelchair|missing|prosthetic)\b'
        ]
        
        distinctive_features = []
        for pattern in distinctive_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            distinctive_features.extend(matches)
        
        if distinctive_features:
            features['distinctive_features'] = list(set(distinctive_features))
        
        return features
    
    def _extract_with_ner(self, text: str) -> Dict[str, any]:
        """Extract features using Named Entity Recognition"""
        if not self.nlp:
            return {}
        
        features = {}
        doc = self.nlp(text)
        
        # Look for person-related entities and their descriptions
        person_descriptions = []
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                # Get surrounding context
                start_idx = max(0, ent.start - 10)
                end_idx = min(len(doc), ent.end + 10)
                context = doc[start_idx:end_idx].text
                person_descriptions.append(context)
        
        # Analyze person descriptions for physical features
        combined_descriptions = ' '.join(person_descriptions).lower()
        
        # Use pattern matching on the extracted person descriptions
        pattern_results = self._extract_with_patterns(combined_descriptions)
        features.update(pattern_results)
        
        return features
    
    def _extract_with_context(self, context_texts: List[str]) -> Dict[str, any]:
        """Extract features using contextual analysis"""
        features = {}
        
        # Analyze clothing and style mentions
        clothing_patterns = r'\b(wore|wearing|dressed|outfit|clothes|shirt|dress|suit|jacket)\b'
        clothing_context = []
        
        for text in context_texts:
            if re.search(clothing_patterns, text.lower()):
                clothing_context.append(text)
        
        if clothing_context:
            # Infer clothing style from context
            formal_indicators = ['suit', 'tie', 'dress', 'formal', 'elegant']
            casual_indicators = ['jeans', 'shirt', 'casual', 'comfortable', 'simple']
            
            clothing_text = ' '.join(clothing_context).lower()
            
            formal_count = sum(clothing_text.count(word) for word in formal_indicators)
            casual_count = sum(clothing_text.count(word) for word in casual_indicators)
            
            if formal_count > casual_count:
                features['clothing_style'] = 'formal'
            elif casual_count > formal_count:
                features['clothing_style'] = 'casual'
        
        return features
    
    def _combine_physical_features(self, feature_dicts: List[Dict[str, any]]) -> Dict[str, any]:
        """Combine physical features from multiple extraction methods"""
        combined = {}
        
        # For single-valued features, take the first non-None value
        single_features = ['height', 'build', 'hair_color', 'hair_style', 'eye_color', 'skin_tone', 'clothing_style', 'age_appearance']
        
        for feature in single_features:
            for feature_dict in feature_dicts:
                if feature in feature_dict and feature_dict[feature]:
                    combined[feature] = feature_dict[feature]
                    break
        
        # For list features, combine all unique values
        distinctive_features = []
        for feature_dict in feature_dicts:
            if 'distinctive_features' in feature_dict:
                distinctive_features.extend(feature_dict['distinctive_features'])
        
        if distinctive_features:
            combined['distinctive_features'] = list(set(distinctive_features))
        
        return combined


class RelationshipAnalyzer:
    """Analyze character relationships from dialogue and narrative"""
    
    def __init__(self):
        try:
            # Load models for relationship analysis
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest"
            )
            
            self.emotion_classifier = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base"
            )
            
            # Relationship keywords
            self.relationship_patterns = {
                'romantic': ['love', 'romance', 'dating', 'boyfriend', 'girlfriend', 'husband', 'wife', 'partner'],
                'family': ['mother', 'father', 'sister', 'brother', 'son', 'daughter', 'parent', 'child', 'family'],
                'friendship': ['friend', 'buddy', 'pal', 'companion', 'ally', 'teammate'],
                'professional': ['colleague', 'boss', 'employee', 'coworker', 'partner', 'client', 'customer'],
                'antagonistic': ['enemy', 'rival', 'opponent', 'foe', 'adversary', 'competitor'],
                'mentor': ['teacher', 'mentor', 'guide', 'advisor', 'instructor', 'coach']
            }
            
        except Exception as e:
            logger.error(f"Error initializing relationship analyzer: {e}")
    
    def analyze_character_relationships(self, characters: List[Character], dialogue_data: List[DialogueLine]) -> Dict[str, Dict[str, str]]:
        """Analyze relationships between characters"""
        
        relationships = defaultdict(lambda: defaultdict(str))
        
        # Group dialogue by character pairs
        character_interactions = defaultdict(list)
        
        for dialogue in dialogue_data:
            char_id = dialogue.character_id
            # Find other characters mentioned in the same scene/context
            for other_char in characters:
                if other_char.character_id != char_id:
                    # Check if other character is mentioned in dialogue
                    if other_char.name.lower() in dialogue.text.lower():
                        character_interactions[(char_id, other_char.character_id)].append(dialogue.text)
        
        # Analyze each character pair interaction
        for (char1_id, char2_id), interactions in character_interactions.items():
            relationship_type = self._analyze_interaction_sentiment(interactions)
            relationships[char1_id][char2_id] = relationship_type
            relationships[char2_id][char1_id] = relationship_type  # Symmetric relationship
        
        return dict(relationships)
    
    def _analyze_interaction_sentiment(self, interactions: List[str]) -> str:
        """Analyze the sentiment and emotion of character interactions"""
        if not interactions:
            return 'unknown'
        
        # Analyze sentiment
        positive_count = 0
        negative_count = 0
        
        # Check for relationship keywords
        relationship_scores = defaultdict(int)
        
        combined_text = ' '.join(interactions).lower()
        
        for rel_type, keywords in self.relationship_patterns.items():
            for keyword in keywords:
                if keyword in combined_text:
                    relationship_scores[rel_type] += combined_text.count(keyword)
        
        # If clear relationship type detected, return it
        if relationship_scores:
            return max(relationship_scores.keys(), key=relationship_scores.get)
        
        # Otherwise, use sentiment analysis
        try:
            for interaction in interactions[:10]:  # Limit to avoid API limits
                sentiment = self.sentiment_analyzer(interaction)
                if sentiment:
                    if sentiment[0]['label'] in ['LABEL_2', 'positive']:
                        positive_count += 1
                    elif sentiment[0]['label'] in ['LABEL_0', 'negative']:
                        negative_count += 1
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
        
        # Determine relationship based on sentiment
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'


class EnhancedCharacterProcessor:
    """Advanced character analysis with personality vectors and physical descriptions"""
    
    def __init__(self):
        self.personality_extractor = PersonalityExtractor()
        self.physical_extractor = PhysicalDescriptionExtractor()
        self.relationship_analyzer = RelationshipAnalyzer()
        
        logger.info("Enhanced character processor initialized")
    
    def extract_enhanced_personality(self, character_text: List[str], dialogue: List[str]) -> PersonalityVector:
        """Extract detailed personality using multiple models and techniques"""
        return self.personality_extractor.extract_personality_from_text(character_text, dialogue)
    
    def extract_physical_description(self, context_texts: List[str]) -> PhysicalDescription:
        """Extract detailed physical descriptions from text"""
        return self.physical_extractor.extract_physical_description(context_texts)
    
    def analyze_character_relationships(self, characters: List[Character], dialogue_data: List[DialogueLine]) -> Dict[str, Dict[str, str]]:
        """Analyze relationships between characters"""
        return self.relationship_analyzer.analyze_character_relationships(characters, dialogue_data)
    
    def process_character_arc(self, character: Character, all_dialogue: List[DialogueLine]) -> List[str]:
        """Analyze character development arc throughout the story"""
        
        # Get character's dialogue chronologically
        char_dialogue = [d for d in all_dialogue if d.character_id == character.character_id]
        char_dialogue.sort(key=lambda x: (x.chapter, x.paragraph))
        
        if not char_dialogue:
            return []
        
        # Divide dialogue into story segments for arc analysis
        total_chapters = max(d.chapter for d in char_dialogue)
        segment_size = max(1, total_chapters // 5)  # 5 segments
        
        arc_points = []
        
        for i in range(0, total_chapters, segment_size):
            segment_dialogue = [d for d in char_dialogue if i <= d.chapter < i + segment_size]
            
            if segment_dialogue:
                # Analyze personality in this segment
                dialogue_texts = [d.text for d in segment_dialogue]
                segment_personality = self.personality_extractor.extract_personality_from_text([], dialogue_texts)
                
                # Create arc description based on dominant traits
                dominant_traits = sorted(
                    [(trait, score) for trait, score in segment_personality.dict().items() if trait != 'confidence'],
                    key=lambda x: x[1],
                    reverse=True
                )[:2]
                
                arc_description = f"Chapters {i+1}-{min(i+segment_size, total_chapters)}: "
                arc_description += f"Primarily {dominant_traits[0][0]} ({dominant_traits[0][1]:.2f})"
                if len(dominant_traits) > 1:
                    arc_description += f" and {dominant_traits[1][0]} ({dominant_traits[1][1]:.2f})"
                
                arc_points.append(arc_description)
        
        return arc_points