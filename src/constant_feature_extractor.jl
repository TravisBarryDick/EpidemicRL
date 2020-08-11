"Feature extractor that always returns [1.0]"
struct ConstantFeatureExtractor end
num_features(Φ::ConstantFeatureExtractor) = 1
get_features(Φ::ConstantFeatureExtractor, s) = [1.0]

