def add(a, b):
    return a + b

    
def engineer_iris_features(df):
    """
    Create ratio features for Iris dataset.
    
    Args:
        df: DataFrame with iris features
        
    Returns:
        DataFrame with added ratio features
    """
    df = df.copy()
    df['petal_ratio'] = df['petal length (cm)'] / df['petal width (cm)']
    df['sepal_ratio'] = df['sepal length (cm)'] / df['sepal width (cm)']
    return df