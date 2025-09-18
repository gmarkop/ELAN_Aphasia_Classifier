import pandas as pd

def create_features_dataframe_from_dict(features_dict):
    """
    Convert a features dictionary to a DataFrame suitable for the classifier.
    
    Args:
        features_dict (dict): Dictionary containing extracted features
        
    Returns:
        pd.DataFrame: DataFrame with a single row containing the features needed for classification
    """
    # Create a DataFrame with the features needed for classification
    feature_names = [
        'words_per_minute',
        'total_pauses_per_minute',
        'grammaticality_ratio',
        'mean_pause_duration',
        'filled_pause_ratio'
    ]
    
    features_df = pd.DataFrame([[
        features_dict['words_per_minute'],
        features_dict['total_pauses_per_minute'],
        features_dict['grammaticality_ratio'],
        features_dict['mean_pause_duration'],
        features_dict['filled_pause_ratio']
    ]], columns=feature_names)
    
    return features_df
